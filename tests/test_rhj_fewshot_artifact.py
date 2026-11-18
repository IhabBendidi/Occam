"""The frozen benchmark definition must be internally consistent and unedited.

A benchmark whose definition can drift is not a benchmark. Every artifact is hashed into
`benchmark.json`, and these tests re-derive the invariants rather than trusting the file:
hand-editing `splits.json` to "just add one perturbation" fails here.

The subtle one is `train ∩ val != {}`, which is DELIBERATE: `train` is the global
perturbation-name set the streaming datasets filter on, and every held-out symbol legitimately
trains from the other two cell lines. `exclusions.json` is what holds out the cells.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "data", "benchmarks", "rhj_fewshot")

N_AXIS, N_VAL, N_TEST, N_HEPG2_TRAIN = 6983, 103, 621, 1657
HOLDOUT = "nadig_2024_hepg2.h5ad"

_reason = "benchmark not built; run scripts/build_rhj_fewshot.py"
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(BENCH, "benchmark.json")), reason=_reason)


def _load(name):
    with open(os.path.join(BENCH, name)) as fh:
        return json.load(fh)


def _sha256_file(name):
    h = hashlib.sha256()
    with open(os.path.join(BENCH, name), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_every_artifact_matches_its_recorded_hash():
    bench = _load("benchmark.json")
    for rel, want in bench["artifact_sha256"].items():
        assert _sha256_file(rel) == want, f"{rel} has been edited since it was built"


def test_the_gene_axis_is_the_declared_size_and_hash():
    bench = _load("benchmark.json")
    rows = open(os.path.join(BENCH, "gene_axis.csv")).read().splitlines()[1:]
    ids = [r.split(",")[0] for r in rows]

    assert len(ids) == N_AXIS
    assert len(set(ids)) == N_AXIS, "duplicate Ensembl ID on the axis"
    assert all(g.startswith("ENSG") for g in ids)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    assert digest == bench["gene_axis"]["sha256"]
    # HSPA14 is dropped, not deduplicated: an ambiguous symbol must not reach the scorer.
    assert "HSPA14" in bench["gene_axis"]["ambiguous_symbols_dropped"]
    syms = [r.split(",")[1] for r in rows]
    assert len(set(syms)) == len(syms), "a symbol appears twice on the axis"


def test_the_split_sizes_are_frozen():
    sp = _load("splits.json")
    assert len(sp["val"]) == N_VAL
    assert len(sp["test_unseen_pert"]) == N_TEST
    assert not set(sp["val"]) & set(sp["test_unseen_pert"])
    assert sp["test_unseen_cell"] == [], "this benchmark holds out perturbations, not a line"
    assert sp["holdout_cell_line"] is None
    assert sorted(sp["cell_lines_train"]) == ["HEPG2", "RPE1", "jurkat"]


def test_train_deliberately_overlaps_val_and_test():
    """The property that reads as leakage and is not. Pinned so nobody 'fixes' it."""
    sp = _load("splits.json")
    held = set(sp["val"]) | set(sp["test_unseen_pert"])
    assert held <= set(sp["train"]), (
        "every held-out symbol must be in the global `train` name set, because it trains "
        "from Jurkat and RPE1; the per-file exclusion is what holds out the HepG2 cells"
    )
    assert "_note" in sp, "the overlap must be explained in the file itself"


def test_the_exclusion_is_exactly_val_plus_test_on_the_holdout_file_only():
    sp, ex = _load("splits.json"), _load("exclusions.json")
    assert set(ex["by_file"]) == {HOLDOUT}, "only HepG2 may have training exclusions"
    excluded = set(ex["by_file"][HOLDOUT])
    assert excluded == set(sp["val"]) | set(sp["test_unseen_pert"])
    assert len(excluded) == N_VAL + N_TEST == 724
    assert ex["eval_cell_lines"]["val"] == ["HEPG2"]
    assert ex["eval_cell_lines"]["test_unseen_pert"] == ["HEPG2"]


def test_every_held_out_symbol_still_trains_from_the_other_two_contexts():
    """The positive half of the invariant: a global exclusion would fail this."""
    ex = _load("exclusions.json")
    excluded = set(ex["by_file"][HOLDOUT])
    for name in ("nadig_2024_jurkat.json", "replogle_2022_rpe1_e.json"):
        rec = _load(os.path.join("files", name))
        have = set(rec["per_pert_cells"])
        missing = sorted(excluded - have)
        assert not missing, (
            f"{name}: {len(missing)} held-out symbols absent here ({missing[:5]}) — they "
            f"could not train from this context, so the benchmark is not few-shot"
        )


def test_the_hepg2_training_set_is_the_complement():
    sp, ex = _load("splits.json"), _load("exclusions.json")
    bench = _load("benchmark.json")
    rec = _load(os.path.join("files", HOLDOUT.replace(".h5ad", ".json")))

    eligible = set(sp["train"])
    excluded = set(ex["by_file"][HOLDOUT])
    hepg2_train = (set(rec["per_pert_cells"]) & eligible) - excluded
    assert len(hepg2_train) == N_HEPG2_TRAIN == bench["counts"]["hepg2_train"]
    assert not hepg2_train & excluded


def test_the_unresolvable_perturbations_are_recorded_not_silently_gone():
    bench = _load("benchmark.json")
    counts = bench["counts"]
    assert sorted(counts["dropped_from_val"]) == ["CCDC144NL"]
    assert sorted(counts["dropped_from_test"]) == ["ALG1L", "C14orf178", "NEDD8-MDP1"]
    assert len(counts["unresolvable"]) == 10
    # and the requested lists are kept, so the difference stays visible
    req = _load("REQUESTED.json")
    assert req["n_val"] == 104 and req["n_test"] == 624


def test_the_axis_lives_inside_the_encoder_vocabulary():
    """Asserted here too, because a checkpoint scored off-axis fails silently."""
    torch = pytest.importorskip("torch")
    ckpt = os.path.join(REPO, "checkpoints", "release", "occam_small_encoder_decoder.pth")
    if not os.path.isfile(ckpt):
        pytest.skip("release checkpoint not bootstrapped")

    bench = _load("benchmark.json")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert bench["encoder_n_genes"] == len(ck["gene_names"]) == 44728
    vocab = {str(g) for g in ck["gene_names"]}
    ids = [r.split(",")[0]
           for r in open(os.path.join(BENCH, "gene_axis.csv")).read().splitlines()[1:]]
    assert set(ids) <= vocab
