"""The two mechanisms the rhj_fewshot benchmark is built on.

**A frozen gene axis.** Every context must present the encoder with the *same* non-zero
columns and the same zeros, so cell line is the only thing that differs between them. A file
that cannot supply the whole axis is an error: a silently truncated axis changes the pooled
loss normalisation and makes two runs incomparable.

**Per-(file, perturbation) training exclusion.** The benchmark holds 724 perturbations out of
HepG2 while the *same symbols* keep training from Jurkat and RPE1, so a global name-level
exclusion would delete two contexts of training data. And an exclusion naming a file that is
not loaded must raise, because that is the one failure that leaks the held-out perturbations
into training with no error and a flattering score.
"""

from __future__ import annotations

import os
import tempfile

import pytest

N_GENES = 40
AXIS = [f"ENSG{i:08d}" for i in range(24)]      # the frozen subset


def _write(path, ensembl_ids, perts, cell_line, np, ad, pd):
    """A tiny h5ad with the obs/var schema the loader expects."""
    n = len(perts)
    X = np.asarray(np.arange(n * len(ensembl_ids)), dtype="float32").reshape(n, -1) + 1.0
    obs = pd.DataFrame(
        {
            "gene_name": pd.Categorical(perts),
            "is_control": np.array([p == "non-targeting" for p in perts]),
            "cell_line": pd.Categorical([cell_line] * n),
            "dataset_batch_num": pd.Categorical([f"{cell_line}-1"] * n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame({"ensembl_gene_id": ensembl_ids,
                        "gene_name": [f"SYM{i}" for i in range(len(ensembl_ids))]},
                       index=[str(i) for i in range(len(ensembl_ids))])
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)
    return path


@pytest.fixture(scope="module")
def corpus():
    np = pytest.importorskip("numpy")
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")

    tmp = tempfile.mkdtemp(prefix="occam_bench_")
    all_ids = [f"ENSG{i:08d}" for i in range(N_GENES)]
    perts = ["AAA", "BBB", "CCC", "DDD", "non-targeting", "non-targeting"]
    files = {
        "wide.h5ad": _write(os.path.join(tmp, "wide.h5ad"), all_ids, perts, "L1", np, ad, pd),
        # a different var ORDER, same content: the masks must still come out identical
        "shuffled.h5ad": _write(os.path.join(tmp, "shuffled.h5ad"), all_ids[::-1], perts,
                                "L2", np, ad, pd),
        # missing two axis genes -> must raise under the frozen axis
        "short.h5ad": _write(os.path.join(tmp, "short.h5ad"),
                             [g for g in all_ids if g not in (AXIS[0], AXIS[1])],
                             perts, "L3", np, ad, pd),
    }
    return tmp, files, all_ids


def _load(path, encoder_gene_names, restrict=None):
    from occam.vcell.dataset import _load_file_metadata

    idx = {g: i for i, g in enumerate(encoder_gene_names)}
    return _load_file_metadata(path, encoder_gene_names, idx, restrict_gene_ids=restrict)


def test_the_frozen_axis_gives_every_context_an_identical_mask(corpus):
    pytest.importorskip("torch")
    tmp, files, all_ids = corpus

    wide = _load(files["wide.h5ad"], all_ids, restrict=set(AXIS))
    shuf = _load(files["shuffled.h5ad"], all_ids, restrict=set(AXIS))

    assert int(wide.gene_mask.sum()) == len(AXIS)
    assert int(shuf.gene_mask.sum()) == len(AXIS)
    # Bitwise identical despite a reversed var order — this is the property the whole
    # cross-context comparison rests on.
    import torch
    assert torch.equal(wide.gene_mask, shuf.gene_mask)


def test_without_the_axis_the_masks_differ(corpus):
    """The unrestricted behaviour, so the test above is known to be doing something."""
    pytest.importorskip("torch")
    tmp, files, all_ids = corpus

    wide = _load(files["wide.h5ad"], all_ids)
    short = _load(files["short.h5ad"], all_ids)
    assert int(wide.gene_mask.sum()) == N_GENES
    assert int(short.gene_mask.sum()) == N_GENES - 2


def test_a_file_that_cannot_supply_the_axis_raises(corpus):
    pytest.importorskip("torch")
    tmp, files, all_ids = corpus

    with pytest.raises(ValueError) as exc:
        _load(files["short.h5ad"], all_ids, restrict=set(AXIS))
    msg = str(exc.value)
    assert "frozen benchmark axis" in msg
    assert AXIS[0] in msg or AXIS[1] in msg, "the message must name what is missing"


def test_the_axis_restricts_the_input_scatter_too(corpus):
    """Restricting the loss but not the input would leave a per-context zero-fill pattern."""
    pytest.importorskip("torch")
    tmp, files, all_ids = corpus

    fm = _load(files["wide.h5ad"], all_ids, restrict=set(AXIS))
    assert len(fm.src_indices) == len(AXIS) == len(fm.dst_indices)


def test_excluded_perts_defaults_to_none_and_is_a_frozenset_when_set(corpus):
    pytest.importorskip("torch")
    tmp, files, all_ids = corpus

    fm = _load(files["wide.h5ad"], all_ids)
    assert fm.excluded_perts is None, "must be inert unless a benchmark sets it"

    fm.excluded_perts = frozenset({"AAA", "BBB"})
    import numpy as np
    keep = ~np.isin(np.asarray(fm.pert_gene_names, dtype=str), list(fm.excluded_perts))
    # 4 perturbed + 2 controls; excluding two perturbations leaves 2 perturbed + 2 controls
    assert int(keep.sum()) == 4


def test_an_exclusion_naming_an_unloaded_file_raises(corpus):
    """The highest-consequence silent failure: a renamed file makes the exclusion a no-op."""
    pytest.importorskip("torch")
    from occam.vcell.dataset import prepare_virtual_cell_data

    tmp, files, all_ids = corpus
    with pytest.raises(ValueError) as exc:
        prepare_virtual_cell_data(
            data_paths=[files["wide.h5ad"]],
            encoder_gene_names=all_ids,
            pert2id={"AAA": 0, "BBB": 1, "CCC": 2, "DDD": 3},
            exclude_file_perts={"a_file_that_moved.h5ad": {"AAA"}},
        )
    msg = str(exc.value)
    assert "not in data_paths" in msg
    assert "a_file_that_moved.h5ad" in msg


def test_the_control_cache_key_covers_the_gene_axis():
    """The field whose absence silently reused embeddings from a different input width."""
    np = pytest.importorskip("numpy")
    from occam.vcell.stream_v2 import control_cache_key, control_cache_mismatch

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as fh:
        fh.write(b"x" * 512)
        path = fh.name
    try:
        base = dict(path=path, n_obs=64, n_genes_encoder=44728,
                    control_rows=np.arange(8), target_sum=2000.0,
                    skip_normalize=False, skip_log1p=False, encoder_key="enc")
        wide = np.arange(100)
        narrow = np.where(np.arange(100) < 40, np.arange(100), -1)

        k_wide, d_wide = control_cache_key(col_map=wide, axis_sha256="A", **base)
        k_narrow, d_narrow = control_cache_key(col_map=narrow, axis_sha256="B", **base)
        k_again, _ = control_cache_key(col_map=wide, axis_sha256="A", **base)

        assert d_wide["n_genes_axis"] == 100 and d_narrow["n_genes_axis"] == 40
        assert k_wide != k_narrow, "a different gene axis must not hit the same cache entry"
        assert k_wide == k_again, "identical inputs must be reproducible"
        diff = control_cache_mismatch(d_wide, d_narrow)
        assert "col_map_sha256" in diff and "n_genes_axis" in diff
        assert d_wide["cache_schema"] == 2, "bumping the schema invalidates old caches"
    finally:
        os.unlink(path)


def test_the_seen_sample_is_depth_matched_to_the_heldout_set():
    """The metric-side exclusion gate must not confound "trained on" with "how deep".

    `pearson_delta` at these cell counts is depth-sensitive, and a uniform draw over 1,657
    training perturbations has a different depth profile from the 103 held-out ones. An
    unmatched sample would make a real generalisation gap and a pure depth difference look
    identical, which is exactly the confound this gate exists to avoid.
    """
    import importlib.util
    import os

    import numpy as np

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "verify_by_metric",
        os.path.join(repo, "scripts", "verify_rhj_exclusion_by_metric.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    counts = {f"S{i}": i for i in range(1, 201)}
    counts.update({f"H{i}": i * 2 for i in range(1, 11)})
    target = [f"H{i}" for i in range(1, 11)]
    seen = [f"S{i}" for i in range(1, 201)]

    picked = mod._matched_sample(seen, target, counts, np.random.default_rng(0))
    assert len(picked) == len(target), "one seen perturbation per held-out one"
    assert len(set(picked)) == len(picked), "sampled without replacement"
    assert sorted(counts[p] for p in picked) == sorted(counts[p] for p in target)

    # and when the pool cannot match exactly it still returns one per target, nearest-first
    thin = {"S1": 5, "S2": 6}
    picked2 = mod._matched_sample(["S1", "S2"], ["H1", "H2"],
                                  {**thin, "H1": 100, "H2": 1},
                                  np.random.default_rng(0))
    assert sorted(picked2) == ["S1", "S2"]
