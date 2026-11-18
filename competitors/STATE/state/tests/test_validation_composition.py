"""Validation predicts pairs the way the export does; singles are untouched.

WHY THIS FILE EXISTS
    `best_celleval.ckpt` is chosen on `val/celleval/selection_score`. On norman_combo the
    validation pass used to predict each val pair through the one-hot of the literal label
    `"A+B"` -- a column that never trains -- while the export predicts the same pair by
    sequential composition (`decode_seq_mean`). Selection was on an untrained input and
    misaligned with what is scored (audit 2026-09-23). These tests pin the fix:

      1. pair path: the "A+B" one-hot column is NEVER fed to the model; every second pass
         receives a first pass's OUTPUT as its control; and the validation prediction is
         bit-identical to the export's own rule (`compose_sequential` from
         competitors/shared, unwrapped) on the same rows.
      2. singles path: `_predict_validation` is bit-identical to the pre-fix callback.
      3. export refactor: the factored `predict_pure_sentences` is bit-identical to the
         loop that used to live inline in `export_occam.py`, on singles and on pairs.

    Tiny randomly initialised STATE model on CPU; runs in ~10 s.
    Run: PYTHONPATH=src .venv/bin/python tests/test_validation_composition.py [orig_callback.py orig_export.py]
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
SHARED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "shared"))
for p in (SRC, SHARED):
    if p not in sys.path:
        sys.path.insert(0, p)

from export_predictions import compose_sequential  # noqa: E402
from state.tx.callbacks.validation_celleval import ValidationCellEvalCallback  # noqa: E402
from state.tx.pure_sentence import make_composed_pair_predict, predict_pure_sentences  # noqa: E402

G, S, CT = 12, 8, "K562"
BATCHES = ["b0", "b1", "b2"]
CONTROL = "empty"


def onehot_map(keys):
    keys = sorted(set(keys))
    eye = torch.eye(len(keys))
    return {k: eye[i] for i, k in enumerate(keys)}


def tiny_model(pert_dim, batch_dim, seed=0):
    from state.tx.models.state_transition import StateTransitionPerturbationModel as M
    torch.manual_seed(seed)
    m = M(input_dim=G, hidden_dim=16, output_dim=G, pert_dim=pert_dim, batch_dim=batch_dim,
          gene_dim=G, embed_key="X_hvg", output_space="gene", cell_set_len=S,
          batch_encoder=True, n_encoder_layers=1, n_decoder_layers=1, loss="mse",
          transformer_backbone_key="llama",
          transformer_backbone_kwargs=dict(
              bidirectional_attention=True, max_position_embeddings=S, hidden_size=16,
              intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
              num_key_value_heads=2, head_dim=8, use_cache=False, attention_dropout=0.0,
              hidden_dropout=0.0, layer_norm_eps=1e-6, pad_token_id=0, bos_token_id=1,
              eos_token_id=2, tie_word_embeddings=False, rotary_dim=0,
              use_rotary_embeddings=False))
    return m.eval()


class _DS:
    def __init__(self, pert_str):
        self.pert_str = np.asarray(pert_str, dtype=object)


class StubDM:
    """Just the surface the callback reads. The loader mimics `val_dataloader_for_scoring`
    (test=True, batch_size=1): one PURE sentence per batch, true length, collate_fn keys."""

    pert_col, cell_type_key, embed_key = "gene_pert", "cell_type", "X_hvg"

    def __init__(self, labels_per_sentence, pert_keys, seed=1):
        self.cell_sentence_len = S
        self.pert_onehot_map = onehot_map(pert_keys)
        self.batch_onehot_map = onehot_map(BATCHES)
        self.cell_type_onehot_map = onehot_map([CT])
        rng = np.random.default_rng(seed)
        self._batches = []
        for lab, t in labels_per_sentence:
            bn = [BATCHES[i] for i in rng.integers(0, len(BATCHES), t)]
            self._batches.append({
                "pert_cell_emb": torch.from_numpy(rng.random((t, G), dtype=np.float32)),
                "ctrl_cell_emb": torch.from_numpy(rng.random((t, G), dtype=np.float32)),
                "pert_emb": torch.stack([self.pert_onehot_map[lab]] * t),
                "pert_name": [lab] * t,
                "cell_type": [CT] * t,
                "cell_type_onehot": torch.stack([self.cell_type_onehot_map[CT]] * t),
                "batch": torch.stack([self.batch_onehot_map[b] for b in bn]),
                "batch_name": bn,
                "dataset_name": ["K562"] * t,
            })
        self.val_datasets = [_DS([l for l, t in labels_per_sentence for _ in range(t)])]

    def val_dataloader_for_scoring(self):
        return iter([dict(b) for b in self._batches])

    def get_control_pert(self):
        return CONTROL

    def get_var_dims(self):
        return {"gene_names": [f"g{i}" for i in range(G)]}


class Recorder:
    """Wrap predict_step: record the one-hot column and the control input of every call."""

    def __init__(self, model):
        self.model, self.calls = model, []
        self._orig = model.predict_step

        def _ps(batch, batch_idx, padded=True, **kw):
            out = self._orig(batch, batch_idx, padded=padded, **kw)
            self.calls.append({
                "cols": batch["pert_emb"].reshape(-1, batch["pert_emb"].shape[-1])
                        .argmax(-1).tolist(),
                "ctrl": batch["ctrl_cell_emb"].reshape(-1, G).detach().cpu().numpy().copy(),
                "out": out["preds"].reshape(-1, G).detach().cpu().numpy().copy(),
            })
            return out
        model.predict_step = _ps

    def restore(self):
        self.model.predict_step = self._orig


# --------------------------------------------------------------------------- pairs

PAIR_SENTS = [("A+B", 8), ("A+B", 3), ("C+D", 5), ("A+C", 6), (CONTROL, 8), (CONTROL, 2),
              ("B+D", 4)]
PAIR_KEYS = ["A", "B", "C", "D", "A+B", "C+D", "A+C", "B+D", CONTROL]


def test_pairs_use_composition_never_the_pair_onehot():
    dm = StubDM(PAIR_SENTS, PAIR_KEYS)
    model = tiny_model(len(dm.pert_onehot_map), len(BATCHES))
    rec = Recorder(model)
    pred, real = ValidationCellEvalCallback()._predict_validation(dm, model)
    rec.restore()

    pair_cols = {i for k, i in ((k, int(v.argmax())) for k, v in dm.pert_onehot_map.items())
                 if "+" in k}
    used = {c for call in rec.calls for c in call["cols"]}
    assert not (used & pair_cols), f"pair one-hot columns fed to the model: {used & pair_cols}"
    for call in rec.calls:
        assert len(set(call["cols"])) == 1, "a sentence mixed perturbations"
        assert len(call["cols"]) <= S

    # the chain: every control input is either a real loader control or a previous pass's
    # output, and pair rows' second passes get first-pass outputs
    loader_ctrl = {r.tobytes() for b in dm._batches for r in b["ctrl_cell_emb"].numpy()}
    seen_out: set = set()
    n_fed_back = 0
    for call in rec.calls:
        for r in call["ctrl"]:
            k = r.tobytes()
            assert k in loader_ctrl or k in seen_out, "control input from nowhere"
            n_fed_back += k in seen_out
        seen_out |= {r.tobytes() for r in call["out"]}
    n_pair_rows = sum(t for l, t in PAIR_SENTS if "+" in l)
    # 2 chains x 1 fed-back pass each = 2 fed-back rows per pair row
    assert n_fed_back == 2 * n_pair_rows, (n_fed_back, n_pair_rows)

    # bit-identical to the export's own rule, applied UNWRAPPED on the same rows
    labels = np.asarray(pred.obs["gene_pert"], dtype=object)
    allb = dm._batches
    ctrl = np.concatenate([b["ctrl_cell_emb"].numpy() for b in allb])
    names = np.asarray([n for b in allb for n in b["pert_name"]], dtype=object)
    bo = torch.cat([b["batch"] for b in allb])
    bn = np.asarray([n for b in allb for n in b["batch_name"]], dtype=object)
    order = np.argsort(names.astype(str), kind="stable")
    ctrl, names, bo, bn = ctrl[order], names[order], bo[order], bn[order]
    assert list(labels) == list(names)
    ct = torch.stack([dm.cell_type_onehot_map[CT]] * len(names))

    pr = np.flatnonzero(["+" in n for n in names])
    calls_pair_onehot = []

    def single(c, labs):
        labs = list(labs)
        rows = pr
        if any("+" in x for x in labs):          # compose_sequential's pass-through call
            calls_pair_onehot.append(len(labs))
            return np.zeros_like(c)              # discarded by compose_sequential
        return predict_pure_sentences(
            model, c, labs, pert_onehot_map=dm.pert_onehot_map, batch_onehot=bo[rows],
            batch_names=bn[rows], cell_type_onehot=ct[rows], cell_type_names=[CT] * len(rows),
            sentence_len=S, device="cpu")

    want_pairs = compose_sequential(single, order="mean")(ctrl[pr], list(names[pr]))
    assert np.array_equal(pred.X[pr], want_pairs), "validation != export rule on pairs"
    # an explicit manual chain for one pair row set, to pin the arithmetic
    a = [n.split("+")[0] for n in names[pr]]
    b = [n.split("+")[1] for n in names[pr]]
    ab = single(single(ctrl[pr], a), b)
    ba = single(single(ctrl[pr], b), a)
    assert np.array_equal(pred.X[pr], (0.5 * (ab + ba)).astype(np.float32))
    assert np.array_equal(real.X, np.concatenate(
        [x["pert_cell_emb"].numpy() for x in allb])[order])
    print(f"  pairs: {len(rec.calls)} predict_step calls, {n_pair_rows} pair rows, "
          f"{n_fed_back} fed-back rows, pair columns used: none; == compose_sequential "
          f"(mean) bit-identically")


# --------------------------------------------------------------------------- singles

SINGLE_SENTS = [("GENE1", 8), ("GENE1", 5), ("GENE2", 7), (CONTROL, 8), ("GENE3", 3),
                (CONTROL, 1)]
SINGLE_KEYS = ["GENE1", "GENE2", "GENE3", "GENE4", CONTROL]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_singles_bit_identical_to_prefix(orig_callback_path):
    dm = StubDM(SINGLE_SENTS, SINGLE_KEYS)
    model = tiny_model(len(dm.pert_onehot_map), len(BATCHES))
    new_p, new_r = ValidationCellEvalCallback()._predict_validation(dm, model)
    old = _load(orig_callback_path, "orig_validation_celleval")
    old_p, old_r = old.ValidationCellEvalCallback()._predict_validation(dm, model)
    assert np.array_equal(new_p.X, old_p.X) and new_p.X.dtype == old_p.X.dtype
    assert np.array_equal(new_r.X, old_r.X)
    assert new_p.obs.equals(old_p.obs) and list(new_p.var_names) == list(old_p.var_names)
    print(f"  singles: {new_p.shape} pred bit-identical to the pre-fix callback "
          f"(max |diff| {float(np.abs(new_p.X - old_p.X).max())})")


# --------------------------------------------------------------------------- export

def _old_export_predict(orig_export_path, ns):
    """Exec the pre-fix inline `predict` from export_occam.py verbatim."""
    src = open(orig_export_path).read().split("\n")
    s = next(i for i, l in enumerate(src) if l.startswith("    def predict(ctrl: np.ndarray"))
    e = next(i for i, l in enumerate(src) if l == "        return out" and i > s)
    code = "\n".join(l[4:] for l in src[s:e + 1])
    exec(code, ns)
    return ns["predict"]


def test_export_refactor_bit_identical(orig_export_path):
    from typing import Sequence
    rng = np.random.default_rng(3)
    for kind, labels in (
        ("singles", ["GENE1"] * 11 + ["GENE2"] * 3 + ["GENE3"] * 9),
        ("pairs", ["A+B"] * 10 + ["A+C"] * 4 + ["C+D"] * 9),
    ):
        keys = SINGLE_KEYS if kind == "singles" else PAIR_KEYS
        pmap = onehot_map(keys)
        bmap = onehot_map(BATCHES)
        model = tiny_model(len(pmap), len(BATCHES), seed=5)
        n = len(labels)
        ctrl = rng.random((n, G), dtype=np.float32)
        batch_labels = np.asarray([BATCHES[i] for i in rng.integers(0, 3, n)], dtype=object)
        batch_onehot = torch.stack([bmap[k] for k in batch_labels])
        ct_onehot = torch.ones(1)

        class _A:
            cell_type, device = CT, "cpu"

        class _DM:
            pert_onehot_map = pmap

        ns = dict(np=np, torch=torch, Sequence=Sequence, dm=_DM, batch_labels=batch_labels,
                  batch_onehot=batch_onehot, ct_onehot=ct_onehot, args=_A, S=S, model=model)
        old = _old_export_predict(orig_export_path, ns)

        def new(c, names):
            m = len(names)
            return predict_pure_sentences(
                model, c, names, pert_onehot_map=pmap, batch_onehot=batch_onehot,
                batch_names=batch_labels, cell_type_onehot=ct_onehot.unsqueeze(0).expand(m, -1),
                cell_type_names=[CT] * m, sentence_len=S, device="cpu")

        if kind == "singles":
            a, b = old(ctrl, labels), new(ctrl, labels)
        else:
            a = compose_sequential(old, order="mean")(ctrl, labels)
            b = make_composed_pair_predict(new, n_rows=n, order="mean")(ctrl, labels)
        assert np.array_equal(a, b), kind
        print(f"  export {kind}: new == old inline export bit-identically {a.shape}")


if __name__ == "__main__":
    test_pairs_use_composition_never_the_pair_onehot()
    if len(sys.argv) >= 3:
        test_singles_bit_identical_to_prefix(sys.argv[1])
        test_export_refactor_bit_identical(sys.argv[2])
    else:
        print("  (pass the pre-fix callback and export paths to run the bit-identity checks)")
    print("OK")
