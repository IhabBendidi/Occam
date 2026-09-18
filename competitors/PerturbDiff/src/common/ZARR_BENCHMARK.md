# `src/common/` — shared utilities

Companion to [../../ZARR_BENCHMARK.md](../../ZARR_BENCHMARK.md).

| File | Contents |
|---|---|
| `utils.py` | `get_short_dsname`, `get_cosine_schedule_with_warmup`, `setup_loggings`, `safe_decode_array`, tensor helpers |
| `paths.py` | module-level path constants read from `configs/path/trixie_path.yaml` |
| `gene_id_map.py` | **new** — Ensembl ↔ gene symbol, chaining `configs/metadata/vocab_id2name.csv` (upstream, precedence) then `gene_id_map.csv` (mygene) |

## `gene_id_map.py`

The pretrained embedding tables (`genept`, `llm`, per-gene) are keyed by **gene symbol**; the zarr
benchmark identifies perturbations by **Ensembl id**. `var.parquet` cannot bridge that: it covers
the 2000 *measured* genes, which overlap the 2023 *perturbed* genes by only 20.1%.

Two tables are chained, earlier wins, because they are complementary — measured against the real
genept pickle on the benchmark's perturbations:

| mapping | resolves in genept |
|---|---|
| mygene only (53,653 pairs) | 1965 (97.1%) |
| upstream `vocab_id2name` only (25,133) | 1979 (97.8%) |
| **chained** | **1980 (97.9%)** |

mygene returns current HGNC symbols (`TIMCC`, `PRP4K`) where genept was built with the older ones
(`FAM136A`, `PRPF4B`); mygene alone resolves `CASP8AP2` and 31,401 ids upstream lacks. Tie-breaks
are deterministic (lowest-numbered Ensembl id per symbol; alphabetically first symbol per id).
Regenerate the mygene half with `tools/build_gene_id_map.py` (`uv sync --extra assets`).

---

## `utils.py`

### `get_short_dsname` (34-45) — the naming constraint

```python
if "tahoe"     in ds_name.lower(): return "tahoe100m"
if "pbmc"      in ds_name.lower(): return "pbmc"
if "replogle"  in ds_name.lower(): return "replogle"
if "cellxgene" in ds_name.lower(): return "cellxgene"
logging.getLogger(__name__).error("Unknown dataset name: %s", ds_name)
assert 0
```

Three call sites depend on this: the per-dataset embedder dict
(`../models/cross_dit/cross_dit_main.py:88-93, 240-244`), the per-dataset loss keys
(`../models/lightning/lightning_module.py:247-248`), and `Cross_DiT.forward`.

**For the zarr benchmark:** the manifest keys `ihabxcell_replogle_nadig_{hepg2,jurkat,k562,rpe1}`
all contain `replogle`, so they resolve — and all four collapse into one bucket. If you want
per-cell-line embedders, add explicit cases here (e.g. `hepg2`/`jurkat`/`k562`/`rpe1`) rather than
renaming datasets, since the name also feeds `dataset_path_map` keys and the pickle cache
filenames in `../data/dataset/dataset_grouping.py:146-147`.

### `safe_decode_array` (139-155)

Decodes HDF5 byte-string arrays to `str`. Used throughout the h5 metadata path
(`../data/metadata_cache.py`, `../data/data_module/data_module_setup.py`,
`../data/dataset/dataset_grouping.py:233-244`). **Not needed on the zarr path** — polars returns
Python `str` directly from parquet. Leave it in place for the h5ad path.

### `setup_loggings` (102-137)

Called first in both entrypoints. Fills `cfg.save_dir_path` (from
`trainer.default_root_dir` + `run_name`) and configures the python logger. Any new
`configs/path/*.yaml` must therefore still define `diffusion.save_dir`.

### `get_cosine_schedule_with_warmup` (47-100)

Hydra target of `configs/optimization/scheduler/cosine.yaml`. Format-agnostic.

### Tensor helpers

`_extract_into_tensor` (157), `maybe_add_mask` (172), `reshape_concat_to_tokens` /
`reshape_tokens_to_concat` (183, 199) — the last two default to `gene_dim=2000`, consistent with
the benchmark's 2000-gene panel.

---

## `paths.py` — a CWD trap

```python
_cfg_candidates = [Path("configs/path/trixie_path.yaml"), Path("configs/path/path.yaml")]
_cfg_path = next((p for p in _cfg_candidates if p.exists()), None)
```

These are **relative paths**, resolved against the *current working directory* at import time. Run
from anywhere other than the repo root and `_paths` silently becomes `{}` (no error), so
`WANDB_LOGGING_DIR` becomes `None`.

Two further notes:

* It hard-codes `trixie_path.yaml` by name. A new `configs/path/upstream_path.yaml` selected via
  the hydra CLI (`path=upstream_path`) will **not** be picked up here — the hydra config and this
  module read different files. In practice only `WANDB_LOGGING_DIR` comes from here, so it is
  cosmetic, but worth knowing when the wandb directory looks wrong.
* `CELLFLOW_*` / `STATE_*` constants (22-37) are all `None` placeholders; the `STATE_*` block is
  dead code referring to the STATE repo's ESM2 protein embeddings.

**Zarr impact: none required.** Optionally make the candidate list absolute relative to
`Path(__file__).parents[2]` so the constants resolve regardless of CWD.
