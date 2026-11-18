"""Occam's stable public API: encode, decode, and predict perturbations.

WHAT THIS IS FOR
    Everything else in `occam` is training and benchmarking machinery whose signatures
    move with the campaign. This module is the surface for someone who just wants to USE
    a checkpoint, and it is deliberately small: three loaders, three objects, one helper.

    It is a façade. It adds no modelling of its own -- it wraps
    `occam.inference.loader`, `occam.decoder.external` and
    `occam.vcell.pipeline.VirtualCellTrainer` and takes care of the two things that are
    easy to get silently wrong:

    1. **Gene alignment.** The checkpoint's vocabulary is the truth, not your file's
       column order. Genes it does not know are dropped, genes it knows that your file
       lacks are zero-filled, and `GeneSpace.prepare` REFUSES when the zero-filled
       fraction is implausible -- because a wrong gene column produces a nearly empty
       input that predicts happily and silently.
    2. **Preprocessing.** Library-normalise to the checkpoint's own `target_sum`, then
       log1p, honouring the `skip_normalize` / `skip_log1p` flags the checkpoint records.
       Passing already-log1p data to a raw-count encoder is silent too.

THE THREE OBJECTS

    >>> from occam.api import load_encoder, load_decoder, load_virtual_cell
    >>> enc = load_encoder("checkpoints/release/occam_small_encoder_decoder.pth")
    >>> z = enc.encode(adata, gene_col="ensembl_gene_id")        # (n_cells, 512)

    >>> dec = load_decoder("checkpoints/release/occam_small_encoder_decoder.pth")
    >>> x_hat = dec.decode(z)                                    # (n_cells, 44728) log1p
    >>> x_hat = dec.reconstruct(adata, gene_col="ensembl_gene_id")   # encode then decode

    >>> vc = load_virtual_cell("checkpoints/vcell/norman_combo_NQ02/best_model.pth")
    >>> vc.perturbations[:3]
    >>> pred = vc.predict(controls, ["FOXA1", "CBL+CNN1"], gene_col="gene_id")
    >>> delta = vc.delta(controls, "CBL+CNN1", gene_col="gene_id")

    `load_virtual_cell` works on ANY vcell checkpoint -- the two published ones and any
    you train yourself with `python -m occam.train --config <your config> --stages vcell`.
    Nothing here is specialised to the published pair.

⚠️ OUTPUT SPACE. `decode`, `reconstruct` and `predict` return **log1p** values at the
   checkpoint's `target_sum`, which is the space the models were trained in and the space
   every reported metric is computed in. `to_counts()` inverts it when you need counts.

⚠️ `predict` returns a profile, not a delta. For perturbation EFFECT, use `delta()`,
   which subtracts a baseline -- and read its docstring, because which baseline you
   subtract changes the number materially (`docs/norman_results.md`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "GeneSpace",
    "AlignmentReport",
    "OccamEncoder",
    "OccamDecoder",
    "OccamVirtualCell",
    "load_encoder",
    "load_decoder",
    "load_virtual_cell",
    "to_counts",
]

# A zero-fill fraction above this aborts `prepare`. It is high on purpose: a legitimately
# narrow panel (2,000 HVGs against a 44,728-gene vocabulary) is 95.5% zero-filled and must
# still work, while a wrong gene column is ~100%.
DEFAULT_MAX_ZERO_FILL = 0.99


@dataclass(frozen=True)
class AlignmentReport:
    """What `GeneSpace.prepare` did to your matrix. Printed by `describe()`."""

    n_cells: int
    n_input_genes: int
    n_matched: int
    n_input_dropped: int
    n_zero_filled: int
    n_vocab: int
    target_sum: float
    normalized: bool
    log1p: bool

    @property
    def zero_fill_fraction(self) -> float:
        return self.n_zero_filled / max(self.n_vocab, 1)

    def describe(self) -> str:
        return (f"{self.n_cells:,} cell(s) · {self.n_matched:,}/{self.n_input_genes:,} "
                f"input genes matched · {self.n_input_dropped:,} dropped · "
                f"{self.n_zero_filled:,}/{self.n_vocab:,} vocabulary genes zero-filled "
                f"({self.zero_fill_fraction:.1%}) · normalize={self.normalized} "
                f"(target_sum {self.target_sum:g}) log1p={self.log1p}")


class GeneSpace:
    """The checkpoint's gene vocabulary and preprocessing, as one reusable object.

    This is the piece worth understanding: a prediction is only meaningful if the input
    sits in the same space the model was trained in, and that space is defined by the
    CHECKPOINT (its ordered `gene_names`, its `target_sum`, its skip flags), never by the
    file you happen to be holding.
    """

    def __init__(self, gene_names: Sequence[str], target_sum: float = 2000.0,
                 skip_normalize: bool = False, skip_log1p: bool = False,
                 symbol_map: str = ""):
        self.gene_names: List[str] = [str(g) for g in gene_names]
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError(
                "the checkpoint's gene vocabulary contains duplicates, so a column "
                "cannot be assigned unambiguously")
        self.target_sum = float(target_sum)
        self.skip_normalize = bool(skip_normalize)
        self.skip_log1p = bool(skip_log1p)
        self.symbol_map = symbol_map
        self._index = {g: i for i, g in enumerate(self.gene_names)}

    def __len__(self) -> int:
        return len(self.gene_names)

    def __repr__(self) -> str:
        return (f"GeneSpace({len(self):,} genes, target_sum={self.target_sum:g}, "
                f"skip_normalize={self.skip_normalize}, skip_log1p={self.skip_log1p})")

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any], symbol_map: str = "") -> "GeneSpace":
        """Read the vocabulary and preprocessing out of a loaded checkpoint dict."""
        genes = ckpt.get("gene_names")
        if not genes:
            raise ValueError(
                "this checkpoint records no `gene_names`, so its input space is unknown "
                "and nothing can be aligned to it")
        cfg = ckpt.get("training_config") or ckpt.get("config") or {}
        if not isinstance(cfg, dict):
            cfg = getattr(cfg, "__dict__", {}) or {}
        return cls(genes,
                   target_sum=float(ckpt.get("target_sum", cfg.get("target_sum", 2000.0))),
                   skip_normalize=bool(cfg.get("skip_normalize", False)),
                   skip_log1p=bool(cfg.get("skip_log1p", False)),
                   symbol_map=symbol_map)

    # ---- the work -----------------------------------------------------------------
    def align(self, input_genes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """`(vocab_columns, input_columns)` -- the 1:1 matches, in input order.

        Falls back to a symbol->Ensembl map when a name is not in the vocabulary, which
        is how a symbol-keyed file reaches an Ensembl vocabulary.
        """
        sym2ens: Dict[str, str] = {}
        p = str(self.symbol_map or "").strip()
        if p and os.path.isfile(p):
            import pandas as pd
            m = pd.read_csv(p)
            sym2ens = dict(zip(m.gene_symbol.astype(str), m.ensembl_gene_id.astype(str)))
        vocab_cols, in_cols = [], []
        for j, g in enumerate(input_genes):
            col = self._index.get(str(g))
            if col is None and sym2ens:
                col = self._index.get(sym2ens.get(str(g), "\0"))
            if col is not None:
                vocab_cols.append(col)
                in_cols.append(j)
        if len(vocab_cols) != len(set(vocab_cols)):
            raise ValueError(
                "two input genes map to one vocabulary column, so the matrix cannot be "
                "scattered unambiguously. Deduplicate the input, or drop the symbol map.")
        return np.asarray(vocab_cols, np.int64), np.asarray(in_cols, np.int64)

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Library-normalise to `target_sum` then log1p, honouring the skip flags."""
        x = np.asarray(raw, dtype=np.float32)
        if not self.skip_normalize:
            s = x.sum(axis=1, keepdims=True)
            s = np.where(s > 0, s, 1.0)
            x = x / s * self.target_sum
        if not self.skip_log1p:
            x = np.log1p(x)
        return x.astype(np.float32, copy=False)

    def prepare(self, data, gene_col: Optional[str] = None,
                max_zero_fill: float = DEFAULT_MAX_ZERO_FILL,
                ) -> Tuple[np.ndarray, np.ndarray, AlignmentReport]:
        """`(X, library_sizes, report)` in the checkpoint's space, ready for the model.

        `data` is an AnnData or a 2-D array. For an array, its columns are assumed to be
        the vocabulary already, in order -- the only case where alignment is skipped.

        ⚠️ The library size is computed over the MATCHED columns and BEFORE scattering,
        because normalising after the scatter would divide by a sum taken over
        zero-filled columns.
        """
        if hasattr(data, "var") and hasattr(data, "X"):
            if gene_col:
                if gene_col not in data.var.columns:
                    raise ValueError(
                        f"gene_col {gene_col!r} is not a column of .var "
                        f"({list(data.var.columns)[:8]})")
                input_genes = [str(g) for g in data.var[gene_col]]
            else:
                input_genes = [str(g) for g in data.var_names]
            raw = data.X
        else:
            raw = np.asarray(data)
            if raw.ndim != 2:
                raise ValueError(f"expected a 2-D matrix, got shape {raw.shape}")
            if raw.shape[1] != len(self):
                raise ValueError(
                    f"a bare array must already be in the checkpoint's space: got "
                    f"{raw.shape[1]:,} columns, vocabulary is {len(self):,}. Pass an "
                    f"AnnData (with `gene_col`) to have the columns aligned for you.")
            input_genes = list(self.gene_names)
        raw = np.asarray(raw.todense() if hasattr(raw, "todense") else raw,
                         dtype=np.float32)

        vocab_cols, in_cols = self.align(input_genes)
        if vocab_cols.size == 0:
            raise ValueError(
                "not one input gene is in the checkpoint's vocabulary. The gene column "
                "is wrong: pass `gene_col=` naming the right .var column (the released "
                "checkpoints use Ensembl ids), or give a `symbol_map`.")
        lib = raw[:, in_cols].sum(axis=1, keepdims=True)
        proc = self.preprocess(raw[:, in_cols])
        X = np.zeros((raw.shape[0], len(self)), dtype=np.float32)
        X[:, vocab_cols] = proc

        n_zero = len(self) - vocab_cols.size
        report = AlignmentReport(
            n_cells=raw.shape[0], n_input_genes=len(input_genes),
            n_matched=int(vocab_cols.size),
            n_input_dropped=len(input_genes) - int(in_cols.size),
            n_zero_filled=int(n_zero), n_vocab=len(self),
            target_sum=self.target_sum, normalized=not self.skip_normalize,
            log1p=not self.skip_log1p)
        if max_zero_fill and report.zero_fill_fraction > max_zero_fill:
            raise ValueError(
                f"{report.zero_fill_fraction:.1%} of the vocabulary had to be "
                f"zero-filled, above max_zero_fill={max_zero_fill:.0%}. The input is "
                f"being read on the wrong axis -- check `gene_col` (the released "
                f"checkpoints use Ensembl ids) or pass a `symbol_map`. Raise "
                f"max_zero_fill if this is genuinely intended.")
        return X, lib.astype(np.float32), report


def to_counts(x_log1p: np.ndarray, library_sizes: np.ndarray) -> np.ndarray:
    """Invert log1p + library normalisation: back to counts at the given library sizes.

    ⚠️ Lossy, and not the inverse of a round trip: the model predicts in log1p space, so
    this recovers counts CONSISTENT with the prediction, not the counts you started from.
    """
    x = np.expm1(np.asarray(x_log1p, dtype=np.float32))
    s = x.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    lib = np.asarray(library_sizes, dtype=np.float32).reshape(-1, 1)
    return (x / s * lib).astype(np.float32)


class _Base:
    """Shared plumbing: the gene space, the device, and batched forward passes."""

    def __init__(self, gene_space: GeneSpace, device: str, batch_size: int = 256):
        self.gene_space = gene_space
        self.device = device
        self.batch_size = int(batch_size)

    @property
    def gene_names(self) -> List[str]:
        return list(self.gene_space.gene_names)

    @property
    def target_sum(self) -> float:
        return self.gene_space.target_sum

    def _batches(self, X: np.ndarray):
        import torch
        for i in range(0, X.shape[0], self.batch_size):
            yield i, torch.from_numpy(X[i:i + self.batch_size]).to(self.device)


class OccamEncoder(_Base):
    """Cells in, 512-dimensional embeddings out."""

    def __init__(self, model, gene_space: GeneSpace, device: str, batch_size: int = 256):
        super().__init__(gene_space, device, batch_size)
        self.model = model

    def __repr__(self) -> str:
        return f"OccamEncoder({self.gene_space!r}, device={self.device!r})"

    def encode(self, data, gene_col: Optional[str] = None, verbose: bool = False,
               max_zero_fill: float = DEFAULT_MAX_ZERO_FILL) -> np.ndarray:
        """`(n_cells, hidden)` embeddings. `data` is an AnnData or a 2-D array."""
        import torch
        X, _, report = self.gene_space.prepare(data, gene_col, max_zero_fill)
        if verbose:
            print(f"[occam] {report.describe()}")
        out = []
        self.model.eval()
        with torch.no_grad():
            for _, xb in self._batches(X):
                out.append(self._feature(xb).float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def _feature(self, xb):
        m = self.model
        for name in ("feature", "encode"):
            fn = getattr(m, name, None)
            if callable(fn):
                return fn(xb)
        enc = getattr(m, "encoder", None)
        if enc is None:
            raise AttributeError(
                "this model exposes neither `feature`, `encode` nor `.encoder`, so the "
                "embedding cannot be taken without guessing")
        return enc(xb)


class OccamDecoder(_Base):
    """Embeddings in, log1p expression out. Also does the full encode->decode."""

    def __init__(self, decoder, gene_space: GeneSpace, device: str,
                 encoder: Optional[OccamEncoder] = None, batch_size: int = 256):
        super().__init__(gene_space, device, batch_size)
        self.decoder = decoder
        self.encoder = encoder

    def __repr__(self) -> str:
        return (f"OccamDecoder({self.gene_space!r}, device={self.device!r}, "
                f"encoder={'yes' if self.encoder else 'no'})")

    def decode(self, z: np.ndarray) -> np.ndarray:
        """`(n_cells, n_genes)` in LOG1P space at the checkpoint's `target_sum`."""
        import torch
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"expected (n_cells, hidden), got shape {z.shape}")
        out = []
        self.decoder.eval()
        with torch.no_grad():
            for i in range(0, z.shape[0], self.batch_size):
                zb = torch.from_numpy(z[i:i + self.batch_size]).to(self.device)
                y = self.decoder(zb)
                if isinstance(y, dict):
                    y = y.get("prediction", next(iter(y.values())))
                out.append(y.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def reconstruct(self, data, gene_col: Optional[str] = None,
                    counts: bool = False, verbose: bool = False) -> np.ndarray:
        """Encode then decode. `counts=True` inverts back to count space."""
        if self.encoder is None:
            raise RuntimeError(
                "this decoder was loaded without an encoder, so it cannot reconstruct "
                "from expression. Use `load_decoder(..., with_encoder=True)`, or call "
                "`decode()` on embeddings you already have.")
        _, lib, report = self.gene_space.prepare(data, gene_col)
        if verbose:
            print(f"[occam] {report.describe()}")
        z = self.encoder.encode(data, gene_col=gene_col)
        x = self.decode(z)
        return to_counts(x, lib) if counts else x


class OccamVirtualCell(_Base):
    """Predict a perturbed profile from a control cell. Any vcell checkpoint.

    Works on the two published checkpoints and on anything trained with
    `python -m occam.train --config <cfg> --stages vcell`; nothing here is specialised.
    """

    def __init__(self, trainer, gene_space: GeneSpace, device: str,
                 batch_size: int = 256, checkpoint: Optional[Dict[str, Any]] = None):
        super().__init__(gene_space, device, batch_size)
        self.trainer = trainer
        self.model = trainer.model
        self.pert2id = dict(trainer.pert2id)
        self._ckpt = checkpoint or {}
        # `"A+B"` is not a graph node; without this every pair resolves to nothing and is
        # silently dropped, which once emptied a whole validation split.
        from occam.vcell.dataset import set_combo_decomposition
        set_combo_decomposition(True)

    def __repr__(self) -> str:
        b = (self._ckpt.get("benchmark") or {}).get("id", "?")
        return (f"OccamVirtualCell(benchmark={b!r}, {len(self.pert2id):,} perturbations, "
                f"epoch={self._ckpt.get('epoch', '?')}, device={self.device!r})")

    # ---- what it knows ------------------------------------------------------------
    @property
    def perturbations(self) -> List[str]:
        """The perturbation vocabulary, sorted. A pair is `"A+B"` with both in here."""
        return sorted(self.pert2id)

    @property
    def compose_modes(self) -> Tuple[str, ...]:
        """The multi-perturbation composition modes, `decode_seq_mean` being default."""
        from occam.vcell.model import VirtualCellModel
        return tuple(VirtualCellModel.COMPOSE_MODES)

    def resolve(self, label: str) -> Optional[List[int]]:
        """Perturbation ids for a label, or None when it cannot be resolved."""
        from occam.vcell.dataset import resolve_pert_indices
        return resolve_pert_indices(str(label), self.pert2id)

    def _resolve_all(self, labels: Sequence[str]) -> List[Tuple[str, List[int]]]:
        out, bad = [], []
        for lab in labels:
            idx = self.resolve(lab)
            (out if idx is not None else bad).append((str(lab), idx))
        if bad:
            detail = []
            for lab, _ in bad[:10]:
                missing = [p for p in str(lab).split("+") if p and p not in self.pert2id]
                detail.append(f"{lab} (unresolved: {missing or [lab]})")
            raise KeyError(
                f"{len(bad)} label(s) are not in this checkpoint's vocabulary of "
                f"{len(self.pert2id):,}: " + "; ".join(detail)
                + ". `.perturbations` lists what it knows.")
        return out

    # ---- prediction ---------------------------------------------------------------
    def predict(self, controls, perturbations: Union[str, Sequence[str]],
                gene_col: Optional[str] = None, mode: str = "decode_seq_mean",
                counts: bool = False, verbose: bool = False):
        """One predicted profile per (control cell x perturbation), as an AnnData.

        `mode` only affects labels with 2+ genes; every reconstruction-based mode is
        identical on a single perturbation. Default `decode_seq_mean`
        (`docs/default_recipe.md`).
        """
        import anndata as ad
        import pandas as pd
        import torch

        labels = [perturbations] if isinstance(perturbations, str) else list(perturbations)
        resolved = self._resolve_all(labels)
        if mode not in self.compose_modes:
            raise ValueError(f"unknown mode {mode!r}; expected one of "
                             f"{list(self.compose_modes)}")
        X, lib, report = self.gene_space.prepare(controls, gene_col)
        if verbose:
            n_pairs = sum(1 for _, i in resolved if len(i) > 1)
            print(f"[occam] {report.describe()}")
            print(f"[occam] {len(resolved)} perturbation(s), {n_pairs} pair(s), "
                  f"mode {mode}")

        preds, obs_pert, obs_row = [], [], []
        self.model.eval()
        with torch.no_grad():
            for lab, idx in resolved:
                for start, xb in self._batches(X):
                    n = xb.shape[0]
                    out = self.model.forward_composed(xb, [list(idx)] * n, mode)
                    preds.append(out["prediction"].float().cpu().numpy())
                    obs_pert += [lab] * n
                    obs_row += list(range(start, start + n))
        P = np.concatenate(preds, axis=0)
        if counts:
            P = to_counts(P, np.tile(lib, (len(resolved), 1)))

        return ad.AnnData(
            X=P,
            obs=pd.DataFrame({"target_gene": obs_pert, "control_row": obs_row},
                             index=[f"{p}__{r}" for p, r in zip(obs_pert, obs_row)]),
            var=pd.DataFrame(index=self.gene_names),
            uns={"occam": {"mode": mode, "target_sum": self.target_sum,
                           "space": "counts" if counts else "log1p",
                           "benchmark": (self._ckpt.get("benchmark") or {}).get("id", ""),
                           "epoch": self._ckpt.get("epoch", -1)}},
        )

    def delta(self, controls, perturbations: Union[str, Sequence[str]],
              gene_col: Optional[str] = None, mode: str = "decode_seq_mean",
              baseline: str = "reconstructed", verbose: bool = False) -> np.ndarray:
        """`(n_perturbations, n_genes)` predicted EFFECT, mean over the control cells.

        ⚠️ `baseline` changes the number materially and both choices are defensible:

        `"reconstructed"` (default) subtracts the model's own unperturbed prediction, so
        the decoder's reconstruction error cancels. This is the `recon`-base convention
        of `docs/norman_results.md`.

        `"real"` subtracts the real (preprocessed) control cells, so the delta is
        relative to the measured baseline. This is the `injctrl` convention, and on
        `rhj_fewshot` F-recon1 measured the decoder's own control reconstruction error at
        ~5x the effect size -- which is exactly the quantity this choice includes or
        removes. Read the two families against each other, never mixed.
        """
        if baseline not in ("reconstructed", "real"):
            raise ValueError("baseline must be 'reconstructed' or 'real'")
        labels = [perturbations] if isinstance(perturbations, str) else list(perturbations)
        pred = self.predict(controls, labels, gene_col=gene_col, mode=mode,
                            verbose=verbose)
        if baseline == "real":
            X, _, _ = self.gene_space.prepare(controls, gene_col)
            base = X.mean(axis=0)
        else:
            base = self._reconstructed_control(controls, gene_col)
        tg = pred.obs["target_gene"].to_numpy()
        return np.stack([np.asarray(pred.X)[tg == lab].mean(axis=0) - base
                         for lab in labels], axis=0)

    def _reconstructed_control(self, controls, gene_col) -> np.ndarray:
        """The model's own prediction with NO perturbation applied, averaged."""
        import torch
        X, _, _ = self.gene_space.prepare(controls, gene_col)
        out = []
        self.model.eval()
        with torch.no_grad():
            for _, xb in self._batches(X):
                # `[-1]` is the no-perturbation slot every composition mode understands.
                res = self.model.forward_composed(xb, [[-1]] * xb.shape[0], "sum")
                out.append(res["prediction"].float().cpu().numpy())
        return np.concatenate(out, axis=0).mean(axis=0)


# ---------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------
def _resolve_ckpt(path: str) -> str:
    if os.path.isdir(path):
        for name in ("best_model.pth", "occam_small_encoder_decoder.pth",
                     "final_model.pth", "best_decoder.pth"):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"{path} is a directory with no recognised checkpoint in it (looked for "
            f"best_model.pth, occam_small_encoder_decoder.pth, final_model.pth, "
            f"best_decoder.pth)")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def load_encoder(path: str, device: str = "cuda", batch_size: int = 256,
                 symbol_map: str = "") -> OccamEncoder:
    """Load any Occam checkpoint for EMBEDDING. Accepts a file or a directory."""
    import torch
    from occam.inference.loader import load_encoder_wrapper

    p = _resolve_ckpt(path)
    device = _pick_device(device)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if ckpt.get("is_virtual_cell_checkpoint"):
        vc = load_virtual_cell(p, device=device, batch_size=batch_size,
                               symbol_map=symbol_map)
        return OccamEncoder(vc.model, vc.gene_space, device, batch_size)
    w = load_encoder_wrapper(p, device=device)
    gs = GeneSpace.from_checkpoint(ckpt, symbol_map=symbol_map)
    return OccamEncoder(w.model, gs, device, batch_size)


def load_decoder(path: str, device: str = "cuda", batch_size: int = 256,
                 with_encoder: bool = True, symbol_map: str = "") -> OccamDecoder:
    """Load the SIMPLE decoder (embedding -> log1p expression).

    `with_encoder=True` also loads the encoder, so `reconstruct()` works end to end.
    """
    import torch

    p = _resolve_ckpt(path)
    device = _pick_device(device)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    gs = GeneSpace.from_checkpoint(ckpt, symbol_map=symbol_map)

    if ckpt.get("is_virtual_cell_checkpoint"):
        vc = load_virtual_cell(p, device=device, batch_size=batch_size,
                               symbol_map=symbol_map)
        enc = OccamEncoder(vc.model, vc.gene_space, device, batch_size) \
            if with_encoder else None
        return OccamDecoder(vc.model.decoder, vc.gene_space, device, enc, batch_size)

    if "decoder_state_dict" not in ckpt:
        raise ValueError(
            f"{os.path.basename(p)} carries no `decoder_state_dict`. A plain MAE "
            f"checkpoint has only its own internal decoder; use the unified release "
            f"checkpoint, a decoder run's best_decoder.pth, or a vcell checkpoint.")
    from occam.decoder.external import load_decoder as _load_ext
    # ⚠️ It returns a 4-TUPLE `(model, gene_names, gene_to_index, config)`. Binding it to
    # one name gives a tuple that looks like a module until the first attribute access --
    # the same shape of bug that killed 135 array tasks on `from_vcell_checkpoint`.
    dec, dec_genes, _dec_index, dec_cfg = _load_ext(p, device=device)
    # The decoder's own recorded vocabulary and target_sum win over the outer ones: it is
    # the module that will actually be fed, and a unified checkpoint can carry a decoder
    # trained on a narrower axis than the encoder.
    if dec_genes:
        gs = GeneSpace(dec_genes,
                       target_sum=float(dec_cfg.get("target_sum") or gs.target_sum),
                       skip_normalize=gs.skip_normalize, skip_log1p=gs.skip_log1p,
                       symbol_map=symbol_map)
    enc = load_encoder(p, device=device, batch_size=batch_size,
                       symbol_map=symbol_map) if with_encoder else None
    return OccamDecoder(dec, gs, device, enc, batch_size)


def load_virtual_cell(path: str, device: str = "cuda", batch_size: int = 256,
                      output_activation: Optional[str] = None,
                      symbol_map: str = "") -> OccamVirtualCell:
    """Load a virtual-cell checkpoint -- published or your own.

    ⚠️ `output_activation` is read from the checkpoint. Checkpoints that predate it being
    saved default to `rectified_sigmoid`, which for a softplus-trained decoder is ~99%
    exact zeros -- a dead model that still loads cleanly. Pass it explicitly for those.
    """
    import torch

    from occam.vcell.pipeline import VirtualCellTrainer

    p = _resolve_ckpt(path)
    device = _pick_device(device)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not ckpt.get("is_virtual_cell_checkpoint"):
        raise ValueError(
            f"{os.path.basename(p)} is not a virtual-cell checkpoint (no "
            f"`is_virtual_cell_checkpoint`). Use `load_encoder` / `load_decoder` for the "
            f"released encoder+decoder, or train one with `python -m occam.train "
            f"--config <cfg> --stages vcell`.")
    act = output_activation or (ckpt.get("decoder_config") or {}).get("output_activation")
    trainer, loaded = VirtualCellTrainer.from_vcell_checkpoint(
        p, device=device, output_activation=act)
    gs = GeneSpace.from_checkpoint(loaded, symbol_map=symbol_map)
    return OccamVirtualCell(trainer, gs, trainer.device, batch_size, loaded)


def _pick_device(device: str) -> str:
    """Fall back to CPU with a notice rather than dying on a machine with no GPU."""
    if str(device).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("[occam] CUDA is not available; using CPU")
                return "cpu"
        except Exception:  # noqa: BLE001 -- torch import problems are not ours to fix
            return "cpu"
    return device
