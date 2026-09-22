"""Phase A for TxPert: predict our panel's split from the drawn half-B control cells.

WHAT THIS ANSWERS
-----------------
Overall goal: predict a perturbation's effect in a context where that perturbation was never
measured, from a single control cell.

This script's job: run TxPert's selected checkpoint over one split of our frozen panel and
write `(18575, 6983)` float32 in OUR evaluation space, which `scripts/eval_competitor.py` then
scores through OUR cell-eval 0.6.0. It measures nothing itself; it is the transport.

HOW IT PREDICTS, AND THE ONE DELIBERATE DEVIATION
-------------------------------------------------
Everything goes through TxPert's own `XcelltypePerturbDataset` + `evaluate()`, deliberately
rather than assembling tensors by hand -- their `export_artifacts.py` docstring makes the point
that the basal input, the dose encoding and the `p` embedding must not be allowed to drift from
what the perturbed rows get during their own evaluation.

⚠️ EXCEPT FOR THE CONTROL PAIRING, WHICH IS OURS AND MUST BE. Their
`_get_control_data(idx)` picks a control cell by (cell_type, batch) -- randomly, or averaged
when `avg_cntr` is set -- because their protocol conditions on a control *population*. Our
split-half protocol conditions each predicted cell on ONE SPECIFIC drawn control cell, the same
cell our own arms and all four baselines see (`eval_control="single"`, `ctrl_avg_k=1`,
`rng = default_rng(42 + 1000 * split)`). `_PairedDataset` below overrides exactly that one
method to return row `idx`'s own control, and changes nothing else.

WHY THIS IS THE RIGHT CHOICE, not a convenience: the paired design is what buys the
0.001-0.003 split-to-split std on `pearson_delta`. If TxPert were fed averaged or randomly
drawn controls while `meanpert`, `transfer` and our own model were fed the drawn ones, the
comparison would stop being paired and the anchors would no longer be anchors. Recorded beside
TxPert's numbers in `docs/rhj_competitors.md`.

EMISSION SPACE
--------------
TxPert emits gene space directly, in the space it was trained on:
`log1p(counts / rowsum_FULL_PANEL * 10_000)`. Declared to `export_split` as
`from_target_sum=10_000, from_log1p=True`, and `assert_representable` refuses only what cannot be SCORED; library size and density are
recorded as diagnostics, because refusing on them would exclude weak arms.
Never let cell-eval decide: `guess_is_lognorm` reads 500 UNSEEDED cells and asks only whether
the row sums have fractional parts.

    python export_occam.py --seed 42 --split 0 --out-dir <dir> \
        --panel-dir data/benchmarks/rhj_fewshot/splithalf \
        --bench-dir data/benchmarks/rhj_fewshot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

ARM = "txpert"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_REPO, "competitors", "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _bench_tag(bench_dir: str) -> str:
    """`""` for rhj_fewshot, `"__<bench>"` otherwise -- the run-directory suffix.

    ⚠️ The training legs tag their run directory by benchmark and the export legs did
    NOT, so an export defaulted to the rhj_fewshot run and would have scored the WRONG
    CHECKPOINT on the right panel. CellFlow's emission-space guard caught its version of
    this ("the gene axis is not ours: 6983 vs 8000"); an arm whose axis happened to match
    would have produced a plausible number instead. Measured 2026-09-22, job 18328366.
    """
    import os as _os
    name = _os.path.basename(_os.path.normpath(str(bench_dir)))
    return "" if name == "rhj_fewshot" else f"__{name}"


def _resolve_checkpoint(seed: int, explicit: str, bench_dir: str = "") -> str:
    """The checkpoint TxPert itself selected on VALIDATION, never the last epoch.

    Its training leg writes `best_ckpt_path.txt` beside the checkpoints, chosen on
    `val/celleval/selection_score` = mean(pearson_delta, discrimination_score_l1), mode=max.
    Refuse rather than fall back to a glob: silently scoring the final epoch would make the
    number incomparable to every other arm, and nothing about the output would say so.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"no checkpoint at {explicit}")
        return explicit
    marker = os.path.join(_REPO, "runs", "competitors", "txpert", "ckpt",
                          f"occam_s{seed}{_bench_tag(bench_dir)}",
                          "best_ckpt_path.txt")
    if not os.path.isfile(marker):
        raise SystemExit(
            f"no selection marker at {marker}. Either training has not finished for seed "
            f"{seed}, or it finished without selecting a checkpoint. Refusing to guess: "
            f"scoring the final epoch instead of the selected one would be silently "
            f"incomparable to every other arm.")
    path = open(marker).read().strip()
    if not os.path.isfile(path):
        raise SystemExit(f"{marker} points at {path}, which does not exist")
    return path


def _paired_dataset_class():
    """`XcelltypePerturbDataset` with OUR control pairing. See the module docstring."""
    from gspp.data.datamodule import XcelltypePerturbDataset

    class _PairedDataset(XcelltypePerturbDataset):
        """One predicted cell per drawn control cell, row for row.

        The ONLY override is `_get_control_data`. Everything else -- the `p` embedding, the
        dose encoding, the condition-name parsing `evaluate()` relies on -- is inherited, so
        it cannot drift from what their own evaluation produces.
        """

        def _get_control_data(self, idx: int):
            return self.control_data[int(idx)]

    return _PairedDataset


def _resolved_kg_list(cfg) -> list:
    """The knowledge graphs this run actually used.

    `modify_config` CONSUMES `cfg.graph.kg_list`, so by export time the surviving copy is
    `cfg.model.pert_model.kg_list`. Refuse an empty list rather than record a manifest that
    says the arm ran with no graphs -- that claim was in the provenance of five reported
    splits before this was fixed.
    """
    for path in (("model", "pert_model", "kg_list"), ("graph", "kg_list"),
                 ("datamodule", "kg_list")):
        node = cfg
        for key in path:
            node = node.get(key) if hasattr(node, "get") else None
            if node is None:
                break
        if node:
            return [str(x) for x in node]
    raise SystemExit(
        "no non-empty kg_list found at model.pert_model / graph / datamodule. Recording an "
        "empty one would misstate which knowledge graphs produced this number.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--config-name", default="config-txpert-occam")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from scipy.sparse import csr_matrix
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader

    import anndata as sc_ad
    import gspp.constants as cs
    from gspp.data.datamodule import collate_fn
    from gspp.evaluation import evaluate
    from export_predictions import TRAIN_TARGET_SUM, export_split, load_panel_split

    ckpt_path = _resolve_checkpoint(args.seed, args.checkpoint, args.bench_dir)
    print(f"[{ARM}] checkpoint {ckpt_path}")

    split = load_panel_split(args.panel_dir, args.split, target_sum=TRAIN_TARGET_SUM)

    # The datamodule is instantiated for its VOCABULARY and GRAPH, not for its loaders:
    # pert2id / id2pert and the knowledge-graph embeddings must be exactly the ones the
    # checkpoint was trained with, or `pert_idxs` index a different perturbation.
    # ⚠️ TWO THINGS `compose()` DOES NOT GIVE YOU THAT `@hydra.main` DOES, both measured on
    # 2026-09-18 and each costing an export run:
    #
    # 1. `HydraConfig` is unset, so every `${hydra:runtime.cwd}` interpolation raises
    #    `InterpolationResolutionError` -- `datamodule.cache_dir` and the graph paths use it
    #    (job 18080877). `return_hydra_config=True` plus `HydraConfig.instance().set_config`
    #    is the documented way to make them resolve.
    # 2. `runtime.cwd` is the PROCESS cwd, and the training leg runs from the TxPert
    #    directory while this adapter is launched from the repo root. Those resolve to
    #    different cache and graph paths, so the export would read a different graph than
    #    the checkpoint trained on. chdir FIRST, so the interpolations match training.
    os.chdir(_HERE)
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=os.path.join(_HERE, "configs"), version_base=None):
        # ⚠️ THE BENCHMARK-DERIVED OVERRIDES ARE COMPUTED HERE, not passed in. The train
        # leg sets `hvg_number` and `test_cell_type` from the artifact; if the export does
        # not set the SAME values it rebuilds a datamodule pinned to the config's
        # `test_cell_type: hepg2`, and on a benchmark whose evaluated line is anything else
        # the validation mask is all-False -> "Validation split is empty." after the
        # training had already completed (measured: job 18328675, 5/5 splits lost).
        # Deriving them from `--bench-dir` here means the two legs cannot disagree and a
        # launcher cannot forget one.
        _bench = json.load(open(os.path.join(args.bench_dir, "benchmark.json")))
        _axis = _bench.get("gene_axis") or {}
        _n_genes = (_axis.get("n_genes") or _axis.get("n_axis_genes")
                    or _bench.get("n_axis_genes"))
        if _n_genes is None:
            with open(os.path.join(args.bench_dir, "gene_axis.csv")) as _fh:
                _n_genes = sum(1 for _ in _fh) - 1
        _line = _bench.get("cell_line") or ""
        if not _line:
            for _rec in (_bench.get("sources") or {}).values():
                if _rec.get("cell_line"):
                    _line = _rec["cell_line"]
                    break
        if not _line:
            _ecl = _bench.get("eval_cell_lines") or []
            _line = _ecl[0] if _ecl else "HEPG2"
        # TxPert's own OCCAM_CELL_LINE_RENAME lowercases HEPG2 and passes everything else.
        _test_ct = "hepg2" if _line == "HEPG2" else _line
        print(f"[export] benchmark-derived: hvg_number={int(_n_genes)} "
              f"test_cell_type={_test_ct!r} (evaluated line {_line!r})")
        cfg = compose(config_name=args.config_name, overrides=[
            f"datamodule.occam_bench_dir={args.bench_dir}",
            f"datamodule.hvg_number={int(_n_genes)}",
            f"datamodule.test_cell_type={_test_ct}",
            f"training.seed={args.seed}",
        ], return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    from omegaconf import OmegaConf

    # ⚠️ The module paths are main.py's, verified against it -- not guessed. The wrapper is
    # in `datamodule_hybrid`, not `datamodule`, and the graph is in `data.graphmodule`, not
    # `graph`. Both wrong guesses cost a 5-split export run (job 18080378).
    from gspp.data.datamodule_hybrid import HybridDataModuleWrapper
    from gspp.data.graphmodule import GSPGraph
    from gspp.predictor import PertPredictor
    from gspp.utils import modify_config, set_seed

    # Their own `mode=test` sequence from main.py, step for step. Reproducing it rather than
    # writing a shorter one matters: `modify_config` resolves interpolations the checkpoint's
    # `model_args` were built from, and `datamodule.setup("fit")` is what populates
    # `condition_group` and the predefined splits -- main.py's comment records that skipping
    # it raises AttributeError downstream.
    # ⚠️ `compose()` RETURNS A STRUCT-MODE CONFIG; `@hydra.main` does not. `modify_config`
    # WRITES new keys (`cfg.datamodule.added_data_args = {}` when added data is off), which
    # struct mode rejects with `ConfigAttributeError: Key 'added_data_args' is not in struct`.
    # main.py never hits this because it gets its config from the decorator. MEASURED
    # 2026-09-18, job 18080629 -- it cost a 5-split export run.
    OmegaConf.set_struct(cfg, False)
    set_seed(args.seed)
    cfg = modify_config(cfg)
    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)
    added_data_args = data_args.pop("added_data_args", {})
    dm = HybridDataModuleWrapper(base_dm_args=data_args, **added_data_args)
    dm.prepare_data()
    dm.setup("fit")

    graph = GSPGraph(pert2id=dm.pert2id, gene2id=dm.gene2id, **cfg.graph)
    model = PertPredictor.load_from_checkpoint(
        ckpt_path,
        input_dim=dm.input_dim,
        output_dim=dm.output_dim,
        adata_output_dim=dm.adata_output_dim,
        num_perts=len(dm.pert2id),
        graph=graph,
        pert_names=list(dm.pert2id.keys()),
        model_args=OmegaConf.to_container(cfg.model, resolve=True),
        device=args.device,
    ).to(args.device).eval()
    base_dm = getattr(dm, "base_dm", dm)
    # ⚠️ ASK THE BUILT GRAPH, NOT THE CONFIG. Two wrong sources cost a run each:
    #   * `cfg.datamodule.kg_list` -- wrong section, printed `graphs []` while the model was
    #     correctly built on ['go', 'string']. Cosmetic, but it reads like a disaster.
    #   * `cfg.graph.kg_list` AFTER `modify_config` -- that function CONSUMES the key while
    #     pruning `gnns_config` to the used graphs, so it reads empty even on a correct run.
    #     A guard built on it aborted a healthy export.
    # `GSPGraph.graph_dict` is populated at construction and is the only authoritative answer.
    _kgs = sorted((getattr(graph, "graph_dict", None) or {}).keys())
    print(f"[{ARM}] graphs {_kgs} · {len(dm.pert2id):,} perturbations · "
          f"{dm.adata_output_dim:,} output genes")
    if not _kgs:
        raise SystemExit(
            "GSPGraph built no graphs. TxPert on this benchmark runs GO + STRING; an empty "
            "graph_dict means the checkpoint's knowledge graphs are not being rebuilt, and "
            "the arm would be scored without the prior it trained on.")

    def _pert_ids(label):
        """The perturbation ids for a label, resolving `"A+B"` into its components.

        ⚠️ TxPert's `pert2id` is per SINGLE GENE -- `_extend_pert_vocabulary` splits every
        observed `"A+B"` on `+` and adds the components -- so a combinatorial label is
        NEVER a key. Looking one up whole made all 111 cohort pairs "absent from the
        checkpoint's vocabulary" (job 18363660, 5/5 splits lost) even though the model was
        trained on exactly them.

        Returning BOTH ids is also TxPert's NATIVE pair mechanism, not our invention:
        `gspp/models/gears_derivative.py:256-275` iterates the ids for a row and SUMS
        their embeddings into one latent shift before decoding. A GEARS condition is
        `"A+B"`; a single gene is `"A+ctrl"`. So a two-id list is what the model expects.
        """
        text = str(label)
        if text in base_dm.pert2id:
            return [base_dm.pert2id[text]]
        parts = [q for q in text.split("+") if q and q != "ctrl"]
        if parts and all(q in base_dm.pert2id for q in parts):
            return [base_dm.pert2id[q] for q in parts]
        return None

    missing = sorted({str(p) for p in split.labels if _pert_ids(p) is None})
    if missing:
        raise SystemExit(
            f"{len(missing)} cohort perturbation(s) cannot be resolved against the "
            f"checkpoint's {len(base_dm.pert2id):,}-symbol vocabulary, e.g. "
            f"{missing[:5]}. The frozen cohort is not negotiable; a partial arm would be "
            f"scored on a different panel.")
    _n_pairs = sum(1 for p in {str(q) for q in split.labels} if len(_pert_ids(p) or []) > 1)
    if _n_pairs:
        print(f"[{ARM}] {_n_pairs} multi-gene condition(s) resolve to 2+ ids; the model "
              f"SUMS their embeddings (its own additive mechanism)")

    paired_cls = _paired_dataset_class()
    cell_type = str(getattr(base_dm, "test_cell_type", "hepg2"))

    def predict(ctrl: np.ndarray, pert_names: Sequence[str]) -> np.ndarray:
        n = len(pert_names)
        # ⚠️ SPARSE, NOT DENSE. `XcelltypePerturbDataset._to_tensor` calls `data.toarray()`
        # unconditionally, so a dense ndarray dies with
        # `AttributeError: 'numpy.ndarray' object has no attribute 'toarray'`.
        # MEASURED 2026-09-18, the adapter's first execution.
        X = csr_matrix(np.asarray(ctrl, dtype=np.float32))
        obs = pd.DataFrame({
            cs.GENE_NAME: [str(p) for p in pert_names],
            cs.IS_CONTROL: np.zeros(n, dtype=bool),
            cs.CELL_TYPE: np.full(n, cell_type),
            # ⚠️ BOTH batch columns. The dataset reads `obs[BATCH]` ("batch"), while the
            # datamodule's own loader writes `BATCH_RAW` ("batch_num") -- setting only the
            # latter is a KeyError inside `_initialize_all_treatment_data`. They are
            # DIFFERENT constants, which is easy to miss because the names are so close.
            cs.BATCH: np.full(n, "occam_panel"),
            cs.BATCH_RAW: np.full(n, "occam_panel"),
            # `evaluate` parses this as "{cell_type}_{pert}+ctrl_{dose}" by splitting on "_",
            # taking [1] for the perturbation and [-1] for the dose. Build it the same way.
            # ⚠️ A GEARS condition name is `{cell_type}_{A}+{B}_{dose}`; a SINGLE gene
            # fills the unused second slot with `ctrl`. Appending `+ctrl` to a pair would
            # make `A+B+ctrl` -- three slots in a two-slot format -- which is the same
            # mistake `data_utils.py` made globally.
            "condition_name": [
                f"{cell_type}_{p}_1+1" if "+" in str(p)
                else f"{cell_type}_{p}+ctrl_1+1" for p in pert_names],
        }, index=[str(i) for i in range(n)])
        ad = sc_ad.AnnData(X=X, obs=obs)
        conditions = pd.Series([_pert_ids(p) for p in pert_names], index=obs.index)

        ds = paired_cls(
            ad, ad, conditions, obs["condition_name"], args.device,
            train_cond_names=None,          # do NOT re-inject controls as extra rows
            match_cntr=False, avg_cntr=False,   # ⚠️ ours: one drawn control per row
            smiles2vec=None, unique_doses=getattr(base_dm, "unique_doses", None),
            obsm_key=cs.ObsmKey.RAW,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)
        with torch.no_grad():
            res = evaluate(loader, model, args.device, ad, base_dm.id2pert)

        out = np.asarray(res[cs.RES_PRED], dtype=np.float32)
        got = [str(p).replace("+ctrl", "") for p in res[cs.RES_PERT_CAT]]
        want = [str(p) for p in pert_names]
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), 0)
            raise RuntimeError(
                f"evaluate() returned rows in a different order than they were given "
                f"(first mismatch at {first}: {got[first]!r} vs {want[first]!r}). A "
                f"positional prediction matrix would then be silently wrong.")
        if out.shape != (n, X.shape[1]):
            raise RuntimeError(f"prediction is {out.shape}, expected {(n, X.shape[1])}")
        return out

    # ⚠️ TXPERT EMITS NEGATIVE VALUES. Measured on the epoch-20 checkpoint, split 0:
    # raw log1p-space range [-0.3274, 5.9249]. It has no non-negativity constraint on its
    # output head, so `to_eval_space` REFUSED the matrix outright -- "a library-normalised
    # expression matrix cannot be negative".
    #
    # Clipping at 0 is declared HERE, in the arm's own adapter, rather than loosened in the
    # shared guard: negative expression is not a prediction, it is an artifact of an
    # unconstrained head, and each arm should own its emission handling. CellFlow's decode
    # already does the same (`clip_min=0.0`), and TxPert's own scorer clips to [0, 14].
    #
    # ⚠️ The clip is MEASURED and printed every run, because "how much was clipped" is the
    # difference between a rounding detail and a changed arm.
    def predict_clipped(ctrl, pert_names):
        out = np.asarray(predict(ctrl, pert_names), dtype=np.float32)
        neg = out < 0.0
        n_neg = int(neg.sum())
        if n_neg:
            print(f"[{ARM}]   clipped {n_neg:,}/{out.size:,} values "
                  f"({100.0 * n_neg / out.size:.3f}%) at 0; min was {float(out.min()):.4f}",
                  flush=True)
        return np.clip(out, 0.0, None)

    export_split(predict_clipped, split, args.out_dir, arm=ARM,
                 from_target_sum=TRAIN_TARGET_SUM, from_log1p=True,
                 batch_size=max(args.batch_size, 512),
                 extra_manifest={
                     "checkpoint": ckpt_path,
                     "seed": args.seed,
                     "config_name": args.config_name,
                     # ⚠️ `model.pert_model.kg_list`, NOT `cfg.datamodule.kg_list`. The
                     # latter does not exist, so `.get(..., [])` silently recorded
                     # `"kg_list": []` in the frozen manifest of every reported split --
                     # i.e. the provenance of the published number claimed the arm ran with
                     # NO knowledge graphs, when it ran with GO + STRING. The adapter already
                     # documented this exact trap above and then fell into it for the
                     # manifest. Read it from where it lives, and refuse if it is empty.
                     "kg_list": _resolved_kg_list(cfg),
                     "emission_clip": "negatives clipped at 0 in the adapter; TxPert's head "
                                      "is unconstrained and emitted down to -0.3274 in log1p "
                                      "space on the epoch-20 checkpoint",
                     "control_pairing": "one drawn half-B control cell per predicted row "
                                        "(match_cntr=False, avg_cntr=False); OUR protocol, "
                                        "not their population convention",
                 })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
