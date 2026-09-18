"""Setup helpers extracted from data_module.py (logic-preserving)."""

import os
import pickle
import tomllib
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.common.utils import get_short_dsname, safe_decode_array
from src.data.dataset.dataset_core import H5adSentenceDataset
from src.data.metadata_cache import GlobalH5MetadataCache


# --------------------------------------------------------------------------- #
# the upstream benchmark canonical zarr path
# --------------------------------------------------------------------------- #
def read_zarr_manifest(toml_path):
    """Parse the dataset manifest -- the same TOML format STATE consumes.

    ``[datasets]``   label -> directory holding X.zarr / obs.parquet / var.parquet
    ``[training]``   label -> "train" (contributes to the train split)
    ``[splits]``     label -> index-based split JSON (optional)
    ``[split_fold]`` label -> fold index (optional, default 0)

    Using STATE's file verbatim is the cheapest guarantee that both models see
    the same datasets, split file and fold.
    """
    with open(toml_path, "rb") as fin:
        doc = tomllib.load(fin)
    datasets = dict(doc.get("datasets", {}))
    if not datasets:
        raise ValueError(f"{toml_path} has no [datasets] entries")
    return (
        datasets,
        dict(doc.get("training", {})),
        dict(doc.get("splits", {})),
        dict(doc.get("split_fold", {})),
    )


def _zarr_dataset_name(label, path):
    """Internal dataset name for a manifest entry.

    We use the **directory basename**, not the manifest label, because
    ``get_short_dsname`` (src/common/utils.py) matches on the substrings
    tahoe / pbmc / replogle / cellxgene and ``assert 0`` otherwise. STATE's
    manifest labels the benchmark datasets ``ihabxcell_hepg2`` etc., which would
    fail that check, while the directories are ``ihabxcell_replogle_nadig_hepg2``
    and resolve to "replogle". Deriving the name keeps the manifest byte-identical
    to STATE's instead of forcing a rename.
    """
    name = Path(path).name
    get_short_dsname(name)  # fail loudly here rather than deep inside the model
    return name


def zarr_setup(dm, stage=None):
    """Build the global covariate vocabularies from obs.parquet.

    Zarr counterpart to ``pretraining_setup`` / ``perturbation_pretraining_setup``,
    which open every file with h5py to read ``obs/<col>/categories``.
    """
    cfg = dm.data_args
    datasets, training, splits, split_folds = read_zarr_manifest(cfg.toml_config_path)

    dm.dataset_name_list, dm.file_list, dm.key_info_list = [], [], []
    label_of = {}
    for label, path in datasets.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"[{label}] dataset directory does not exist: {path}")
        name = _zarr_dataset_name(label, path)
        if name in label_of:
            raise ValueError(
                f"two manifest entries resolve to the same dataset name {name!r}: "
                f"{label_of[name]!r} and {label!r}"
            )
        label_of[name] = label
        dm.dataset_name_list.append(name)
        dm.file_list.append(str(path))
        dm.key_info_list.append(
            {
                "control_pert": cfg.control_pert,
                "cell_type_key": cfg.cell_line_key,
                "batch_col": cfg.perturbseq_batch_col,
                "split_file": splits.get(label),
                "split_fold": int(split_folds.get(label, getattr(cfg, "split_idx", 0))),
                "is_training": training.get(label) == "train",
            }
        )
        if label != name:
            dm.py_logger.info("manifest '%s' -> dataset name '%s'", label, name)

    n = len(dm.dataset_name_list)
    dm.data_type_list = ["Perturb-seq"] * n
    # The gene panel comes from data.selected_gene_file, shared by all datasets.
    dm.selected_genes_list = [None] * n
    dm.original_dataset_name_list = list(dm.dataset_name_list)

    meta_cache = GlobalH5MetadataCache()
    all_perts, all_batches, all_celltypes = set(), set(), set()
    for name, path, key_info in zip(dm.dataset_name_list, dm.file_list, dm.key_info_list):
        cache = meta_cache.get_zarr_cache(
            path,
            cell_type_key=key_info["cell_type_key"],
            batch_col=key_info["batch_col"],
            control_pert=key_info["control_pert"],
        )
        all_perts.update(cache.pert_categories.tolist())
        all_celltypes.update(cache.cell_type_categories.tolist())
        # Batch labels are namespaced by dataset, matching
        # dataset_core.get_covarates and sampling's collect_batch_covariates.
        all_batches.update(f"{name}_{b}" for b in cache.batch_categories.tolist())

    dm.pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    dm.batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    dm.cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.py_logger.info(
        f"#datasets: {n}, #perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, #batch: {len(all_batches)}"
    )


def zarr_setup_dataset(dm, stage=None):
    """Split each dataset by its index JSON and build the per-split datasets."""
    from src.data.split_strategy import split_index_json

    meta_cache = GlobalH5MetadataCache()
    meta_cache.register_covariate_dict(dm.pert_dict, dm.batch_dict, dm.cell_type_dict)

    for split_stage in ["train"] + dm.all_split_names:
        setattr(dm, f"{split_stage}_num_cell", {})
        setattr(dm, f"{split_stage}_indices", {})

    dataset_path_map = dict(zip(dm.dataset_name_list, dm.file_list))
    selected_genes_list = dict(zip(dm.dataset_name_list, dm.selected_genes_list))
    control_type = {name: dm.data_args.control_pert for name in dm.dataset_name_list}

    # Splits are read straight from JSON, so they are not cached: it is cheap,
    # and it removes a stale-cache surface (the grouping cache in
    # dataset_grouping.py is content-hashed for the same reason).
    for name, path, key_info in zip(dm.dataset_name_list, dm.file_list, dm.key_info_list):
        cache = meta_cache.get_zarr_cache(
            path,
            cell_type_key=key_info["cell_type_key"],
            batch_col=key_info["batch_col"],
            control_pert=key_info["control_pert"],
        )
        dm.py_logger.info("[%s] splitting ...", name)
        split_indices = split_index_json(dm, cache, "pert", key_info)
        for split_stage, indices in split_indices.items():
            getattr(dm, f"{split_stage}_indices")[name] = indices
            getattr(dm, f"{split_stage}_num_cell")[name] = len(indices)

    for split_stage in ["train"] + dm.all_split_names:
        total = sum(getattr(dm, f"{split_stage}_num_cell").values())
        dm.py_logger.info(">>>>> stage: %s (%d cells)", split_stage, total)
        setattr(
            dm,
            f"{split_stage}_dataset",
            H5adSentenceDataset(
                split_stage,
                meta_cache,
                dataset_path_map,
                selected_genes_list,
                getattr(dm, f"{split_stage}_indices"),
                getattr(dm, f"{split_stage}_num_cell"),
                control_type,
                dm.data_args,
                dm.py_logger,
            ),
        )

    # Preload expression in the MAIN process so forked workers share it
    # copy-on-write instead of each re-reading gigabytes.
    if getattr(dm.data_args, "feature_loading", "ram") in ("ram", "npy"):
        getattr(dm, "train_dataset").store.preload()


def occam_setup(dm, stage=None):
    """Build the global covariate vocabularies from Occam's `rhj_fewshot` benchmark.

    Occam counterpart to ``zarr_setup``. One dataset is one CELL LINE: ``[datasets]`` names our
    three h5ads and each is read in place through ``competitors/shared/occam_data.py``.

    ⚠️ THERE IS NO ``[splits]`` TABLE, deliberately. The frozen benchmark owns the split, and
    ``split_index_occam`` computes it live, so there is no row-index JSON that can go stale.
    ``key_info`` carries ``occam_bench_dir`` instead of ``split_file``.

    ⚠️ ASYMMETRY BY DESIGN: HepG2 is the evaluated line and splits three ways; jurkat and RPE1
    are wholly training data. The very same perturbation held out of HepG2 still TRAINS from
    them, which is what makes the task few-shot-in-a-new-context rather than
    never-seen-at-all.
    """
    from src.common.utils import get_short_dsname
    from src.data.occam_bridge import occam_line_from_path, read_occam_line

    cfg = dm.data_args
    bench_dir = str(cfg.occam_bench_dir)
    datasets, training, splits, split_folds = read_zarr_manifest(cfg.toml_config_path)
    if splits:
        raise ValueError(
            f"the manifest has a [splits] table ({sorted(splits)}) but source_format='occam' "
            f"reads the split from the benchmark itself. A second split file could only "
            f"disagree with it silently; remove the table.")

    dm.dataset_name_list, dm.file_list, dm.key_info_list = [], [], []
    seen = {}
    for label, path in datasets.items():
        line = occam_line_from_path(path)
        if line in seen:
            raise ValueError(
                f"two manifest entries resolve to the same cell line {line!r}: "
                f"{seen[line]!r} and {label!r}")
        seen[line] = label
        # ⚠️ THE DATASET NAME MUST CONTAIN A FAMILY `get_short_dsname` KNOWS.
        # `src/common/utils.py::get_short_dsname` matches the substrings tahoe / pbmc /
        # replogle / cellxgene and `assert 0` otherwise, and `_compute_loss` calls it on every
        # batch's `ds_name`. Naming our datasets after the cell line alone ("HEPG2", "jurkat",
        # "RPE1") matched nothing and killed training ~4 min in, inside the loss -- measured
        # 2026-09-18, jobs 18072825 and 18072187.
        #
        # `replogle_nadig_` is the honest prefix: these ARE the Replogle/Nadig sources
        # (nadig_2024_hepg2, nadig_2024_jurkat, replogle_2022_rpe1_e), so the family the loss
        # branches on is the one it would have picked for the same data as a zarr dump. Their
        # own `_zarr_dataset_name` solves this the same way, by deriving a name that contains
        # "replogle" rather than using the manifest label.
        name = f"replogle_nadig_{line}"
        get_short_dsname(name)      # fail loudly HERE, not deep inside the model
        dm.dataset_name_list.append(name)
        dm.file_list.append(str(path))
        dm.key_info_list.append(
            {
                "control_pert": cfg.control_pert,
                "cell_type_key": cfg.cell_line_key,
                "batch_col": cfg.perturbseq_batch_col,
                "occam_bench_dir": bench_dir,
                "occam_line": line,
                "occam_dataset_name": name,
                "split_file": None,
                "split_fold": 0,
                "is_training": training.get(label, "train") == "train",
            }
        )
        dm.py_logger.info("manifest '%s' -> cell line '%s' -> dataset '%s'",
                          label, line, name)

    n = len(dm.dataset_name_list)
    dm.data_type_list = ["Perturb-seq"] * n
    # The gene panel IS the frozen axis; there is nothing to select.
    dm.selected_genes_list = [None] * n
    dm.original_dataset_name_list = list(dm.dataset_name_list)

    meta_cache = GlobalH5MetadataCache()
    all_perts, all_batches, all_celltypes = set(), set(), set()
    for name, path, key_info in zip(dm.dataset_name_list, dm.file_list, dm.key_info_list):
        cache = meta_cache.get_occam_cache(
            path,
            bench_dir=bench_dir,
            cell_type_key=key_info["cell_type_key"],
            batch_col=key_info["batch_col"],
            control_pert=key_info["control_pert"],
            target_sum=float(getattr(cfg, "occam_target_sum", 10_000.0)),
            log1p=bool(getattr(cfg, "occam_log1p", True)),
            verify_sources=bool(getattr(cfg, "occam_verify_sources", True)),
            max_cells=getattr(cfg, "occam_max_cells_per_line", None),
        )
        all_perts.update(cache.pert_categories.tolist())
        all_celltypes.update(cache.cell_type_categories.tolist())
        # Batch labels are namespaced by dataset, matching dataset_core.get_covarates and
        # sampling's collect_batch_covariates.
        all_batches.update(f"{name}_{b}" for b in cache.batch_categories.tolist())

    dm.pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    dm.batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    dm.cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.py_logger.info(
        f"#datasets: {n}, #perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, #batch: {len(all_batches)}"
    )

    # ⛔ FAIL AT SECOND ZERO, NOT 43 MINUTES IN.
    #
    # The validation cell-eval callback CATCHES its own exceptions, logs "not publishing a
    # score" and continues -- so anything it needs and does not have becomes a
    # ModelCheckpoint failure on a missing `validation/celleval/selection_score` metric at the
    # FIRST validation, which on this recipe is step 16,000. Two separate defects were found
    # that way on 2026-09-18, each costing a full run: the dataset-family name
    # (`get_short_dsname`) and the gene panel (`load_selected_genes` on a null path).
    #
    # So exercise the validation path's own prerequisites here, where a failure is free.
    try:
        from src.apps.sampling.sampling_generation_helpers import load_selected_genes

        class _Probe:
            pass

        probe = _Probe()
        probe.data = cfg
        genes = load_selected_genes(probe)
    except Exception as exc:
        raise RuntimeError(
            f"the validation cell-eval path cannot resolve its gene panel: "
            f"{type(exc).__name__}: {exc}\n"
            f"That callback swallows its own errors, so this would surface as a "
            f"ModelCheckpoint failure on a missing selection_score at step "
            f"{getattr(cfg, 'val_check_interval', 16000)}, not here.") from exc
    if len(genes) != int(cfg.pad_length):
        raise RuntimeError(
            f"the validation gene panel has {len(genes)} genes but data.pad_length is "
            f"{cfg.pad_length}; the emitted h5ad and the model width must agree.")
    dm.py_logger.info("validation cell-eval prerequisites OK: %d gene symbols", len(genes))


def occam_setup_dataset(dm, stage=None):
    """Split each cell line live from the benchmark and build the per-split datasets."""
    from src.data.occam_bridge import split_index_occam

    cfg = dm.data_args
    bench_dir = str(cfg.occam_bench_dir)

    meta_cache = GlobalH5MetadataCache()
    meta_cache.register_covariate_dict(dm.pert_dict, dm.batch_dict, dm.cell_type_dict)

    for split_stage in ["train"] + dm.all_split_names:
        setattr(dm, f"{split_stage}_num_cell", {})
        setattr(dm, f"{split_stage}_indices", {})

    dataset_path_map = dict(zip(dm.dataset_name_list, dm.file_list))
    selected_genes_list = dict(zip(dm.dataset_name_list, dm.selected_genes_list))
    control_type = {name: cfg.control_pert for name in dm.dataset_name_list}

    for name, path, key_info in zip(dm.dataset_name_list, dm.file_list, dm.key_info_list):
        cache = meta_cache.get_occam_cache(
            path,
            bench_dir=bench_dir,
            cell_type_key=key_info["cell_type_key"],
            batch_col=key_info["batch_col"],
            control_pert=key_info["control_pert"],
            target_sum=float(getattr(cfg, "occam_target_sum", 10_000.0)),
            log1p=bool(getattr(cfg, "occam_log1p", True)),
            verify_sources=False,
            max_cells=getattr(cfg, "occam_max_cells_per_line", None),
        )
        dm.py_logger.info("[%s] splitting ...", name)
        split_indices = split_index_occam(dm, cache, "pert", key_info)
        for split_stage, indices in split_indices.items():
            getattr(dm, f"{split_stage}_indices")[name] = indices
            getattr(dm, f"{split_stage}_num_cell")[name] = len(indices)

    for split_stage in ["train"] + dm.all_split_names:
        total = sum(getattr(dm, f"{split_stage}_num_cell").values())
        dm.py_logger.info(">>>>> stage: %s (%d cells)", split_stage, total)
        setattr(
            dm,
            f"{split_stage}_dataset",
            H5adSentenceDataset(
                split_stage,
                meta_cache,
                dataset_path_map,
                selected_genes_list,
                getattr(dm, f"{split_stage}_indices"),
                getattr(dm, f"{split_stage}_num_cell"),
                control_type,
                cfg,
                dm.py_logger,
            ),
        )

    # Preload expression in the MAIN process so forked workers share it copy-on-write
    # instead of each re-reading gigabytes.
    getattr(dm, "train_dataset").store.preload()


def pretraining_setup_dataset(dm, stage=None):
    """
    Set up datasets by looping through all processed cellxgene / tahoe100m datasets.
    Split by different strategies: random and holdout
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    meta_cache = GlobalH5MetadataCache()
    meta_cache.register_covariate_dict(
        dm.pert_dict,
        dm.batch_dict,
        dm.cell_type_dict,
    )
    for split_stage in ["train"] + dm.all_split_names:
        setattr(dm, f"{split_stage}_num_cell", {})
        setattr(dm, f"{split_stage}_indices", {})

    dataset_path_map = {x: y for x, y in zip(dm.dataset_name_list, dm.file_list)}
    selected_genes_list = {x: y for x, y in zip(dm.dataset_name_list, dm.selected_genes_list)}
    control_type = {
        x: None if y == "RNA-seq" else dm.data_args.control_pert
        for x, y in zip(dm.dataset_name_list, dm.data_type_list)
    }

    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Splitting into train/validation/test data (only for indices) ...",
    ):
        # RNA-seq from cellxgene only contains cell types
        # Perturb-seq from tahoe100m only contains perturbation conditions
        if data_type == "RNA-seq":
            assert 0
        elif data_type == "Perturb-seq":
            cache = meta_cache.get_cache(
                dataset_path,
                batch_col=dm.data_args.perturbseq_batch_col,
                pert_col=dm.data_args.pert_col,
                cell_type_key=dm.data_args.cell_line_key,
            )
            split_indices = dm._proceed_data_split(
                cache, "pert", key_info, "holdout_pert", "random_Perturbseq"
            )
        else:
            raise NotImplementedError

        for split_stage in split_indices:
            tmp = getattr(dm, f"{split_stage}_indices")
            tmp[dataset_name] = split_indices[split_stage]

            tmp = getattr(dm, f"{split_stage}_num_cell")
            tmp[dataset_name] = len(split_indices[split_stage])

            dm.py_logger.info(f"Processed {split_stage}: {len(split_indices[split_stage])}")

    for split_stage in ["train"] + dm.all_split_names:
        dm.py_logger.info(">>>>> stage: %s", split_stage)
        setattr(
            dm,
            f"{split_stage}_dataset",
            H5adSentenceDataset(
                split_stage,
                meta_cache,
                dataset_path_map,
                selected_genes_list,
                getattr(dm, f"{split_stage}_indices"),
                getattr(dm, f"{split_stage}_num_cell"),
                control_type,
                dm.data_args,
                dm.py_logger,
            ),
        )


def pretraining_setup(dm, stage=None):
    """
    Pretraining setup.

    :param dm: Input `dm` value.
    :param stage: Input `stage` value.
    :return: Computed output(s) for this function.
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    all_perts, all_batches, all_celltypes = set(), set(), set()

    # search through data directory for all processed h5 / h5ad files
    (
        dm.dataset_name_list,
        dm.file_list,
        dm.data_type_list,
        dm.key_info_list,
        dm.selected_genes_list,
    ) = dm.get_dataset_names_and_paths(dm.data_args, dm.data_args.data_name)

    dm.original_dataset_name_list = list(dm.dataset_name_list)

    # process to get all perturbation / cell type categories
    for dataset_name, dataset_path, data_type in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list), "Processing meta data ..."
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                assert 0

            elif data_type == "Perturb-seq":
                # cell line
                try:
                    celltype_arr = f[f"obs/{dm.data_args.cell_line_key}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{dm.data_args.cell_line_key}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # perturbation
                pert_arr = f[f"obs/{dm.data_args.pert_col}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{dm.data_args.perturbseq_batch_col}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{dm.data_args.perturbseq_batch_col}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

    dm.pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    dm.batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    dm.cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.original_dataset_name_list = dm.dataset_name_list

    dm.py_logger.info(
        f"#perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, "
        f"#batch: {len(all_batches)}"
    )


def perturbation_pretraining_setup_dataset(dm, stage=None):
    """
    Set up datasets by looping through all processed cellxgene / tahoe100m datasets.
    Split by different strategies: random and holdout
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    meta_cache = GlobalH5MetadataCache()
    meta_cache.register_covariate_dict(
        dm.pert_dict,
        dm.batch_dict,
        dm.cell_type_dict,
    )
    for split_stage in ["train"] + dm.all_split_names:
        setattr(dm, f"{split_stage}_num_cell", {})
        setattr(dm, f"{split_stage}_indices", {})

    dataset_path_map = {x: y for x, y in zip(dm.dataset_name_list, dm.file_list)}
    selected_genes_list = {x: y for x, y in zip(dm.dataset_name_list, dm.selected_genes_list)}
    control_type = {
        x: None if y == "RNA-seq" else z["control_pert"]
        for x, y, z in zip(dm.dataset_name_list, dm.data_type_list, dm.key_info_list)
    }

    exist_flag = False
    if dm.data_args.data_name == "Tahoe100mPBMCReplogleCellxGenePretrain":
        exist_flag = True
        for split_stage in ["train"] + dm.all_split_names:
            split_indices_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_indices_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            split_num_cell_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_num_cell_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            if (not os.path.exists(split_indices_cache_file)) or (
                not os.path.exists(split_num_cell_cache_file)
            ):
                exist_flag = False
                break

    if not exist_flag or dm.data_args.skip_cached_indices:
        for dataset_name, dataset_path, data_type, key_info in tqdm(
            zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
            "Splitting into train/validation/test data (only for indices) ...",
        ):
            # RNA-seq from cellxgene only contains cell types
            # Perturb-seq from tahoe100m only contains perturbation conditions
            if data_type == "RNA-seq":
                cache = meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["rnaseq_batch_col"],
                    cell_type_key=key_info["cell_type_key"],
                )
                split_indices = dm._proceed_data_split_cellxgene(
                    cache, "celltype", key_info, "holdout_celltype", "random_RNAseq"
                )

            elif data_type == "Perturb-seq":
                cache = meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["perturbseq_batch_col"],
                    pert_col=key_info["pert_col"],
                    cell_type_key=key_info["cell_line_key"],
                )
                if "pbmc" in dataset_path.lower():
                    split_indices = dm._proceed_data_split_pbmc(
                        cache, "pert", key_info, "holdout_pert", "random_Perturbseq"
                    )
                elif "tahoe100m" in dataset_path.lower() or "replogle" in dataset_path.lower():
                    split_indices = dm._proceed_data_split_tahoe100m(
                        cache, "pert", key_info, "holdout_pert", "random_Perturbseq"
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            for split_stage in split_indices:
                tmp = getattr(dm, f"{split_stage}_indices")
                tmp[dataset_name] = split_indices[split_stage]

                tmp = getattr(dm, f"{split_stage}_num_cell")
                tmp[dataset_name] = len(split_indices[split_stage])

                dm.py_logger.info(f"Processed {split_stage}: {len(split_indices[split_stage])}")

        if not exist_flag and not dm.data_args.skip_cached_indices:
            for split_stage in ["train"] + dm.all_split_names:
                split_indices_cache_file = os.path.join(
                    dm.data_args.indices_cache_dir, f"split_indices_{dm.data_args.data_name}_{split_stage}.pkl"
                )
                split_num_cell_cache_file = os.path.join(
                    dm.data_args.indices_cache_dir, f"split_num_cell_{dm.data_args.data_name}_{split_stage}.pkl"
                )
                with open(split_indices_cache_file, "wb") as fout:
                    pickle.dump(getattr(dm, f"{split_stage}_indices"), fout)
                with open(split_num_cell_cache_file, "wb") as fout:
                    pickle.dump(getattr(dm, f"{split_stage}_num_cell"), fout)
                dm.py_logger.info(f"Processed {split_stage}: writing split indices to caches")
    else:
        # store cache still
        for dataset_name, dataset_path, data_type, key_info in tqdm(
            zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
            "Splitting into train/validation/test data (only for indices) ...",
        ):
            # RNA-seq from cellxgene only contains cell types
            # Perturb-seq from tahoe100m only contains perturbation conditions
            if data_type == "RNA-seq":
                meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["rnaseq_batch_col"],
                    cell_type_key=key_info["cell_type_key"],
                )
            elif data_type == "Perturb-seq":
                meta_cache.get_cache(
                    dataset_path,
                    batch_col=key_info["perturbseq_batch_col"],
                    pert_col=key_info["pert_col"],
                    cell_type_key=key_info["cell_line_key"],
                )
            else:
                raise NotImplementedError

        for split_stage in ["train"] + dm.all_split_names:
            split_indices_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_indices_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            split_num_cell_cache_file = os.path.join(
                dm.data_args.indices_cache_dir, f"split_num_cell_{dm.data_args.data_name}_{split_stage}.pkl"
            )
            dm.py_logger.info(f"Processed {split_stage}: loading from cached split indices")
            with open(split_indices_cache_file, "rb") as fin:
                split_indices = pickle.load(fin)
            setattr(dm, f"{split_stage}_indices", split_indices)
            with open(split_num_cell_cache_file, "rb") as fin:
                split_num_cell = pickle.load(fin)
            setattr(dm, f"{split_stage}_num_cell", split_num_cell)

    for split_stage in ["train"] + dm.all_split_names:
        dm.py_logger.info(">>>>> stage: %s", split_stage)
        setattr(
            dm,
            f"{split_stage}_dataset",
            H5adSentenceDataset(
                split_stage,
                meta_cache,
                dataset_path_map,
                selected_genes_list,
                getattr(dm, f"{split_stage}_indices"),
                getattr(dm, f"{split_stage}_num_cell"),
                control_type,
                dm.data_args,
                dm.py_logger,
            ),
        )


def build_new_pert_dict(dm):
    """
    Build new pert dict.

    :param dm: Input `dm` value.
    :return: Requested object(s) for downstream use.
    """
    dm.py_logger.info("Calling new_pert_dict ...")

    all_perts = set()

    # process to get all perturbation / cell type categories
    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Processing meta data ...",
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                pass
            elif data_type == "Perturb-seq":
                # perturbation
                pert_arr = f[f"obs/{key_info['pert_col']}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

    pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    return pert_dict


def build_new_all_dict(dm):
    """
    Build new all dict.

    :param dm: Input `dm` value.
    :return: Requested object(s) for downstream use.
    """
    all_perts, all_batches, all_celltypes = set(), set(), set()

    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Processing meta data ...",
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                # cell type
                try:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

            elif data_type == "Perturb-seq":
                # cell line
                try:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # perturbation
                pert_arr = f[f"obs/{key_info['pert_col']}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

    pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.py_logger.info(
        f"#perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, "
        f"#batch: {len(all_batches)}"
    )
    return pert_dict, batch_dict, cell_type_dict


def perturbation_pretraining_setup(dm, stage=None):
    """
    Perturbation pretraining setup.

    :param dm: Input `dm` value.
    :param stage: Input `stage` value.
    :return: Computed output(s) for this function.
    """
    dm.py_logger.info("Calling pretraining data module setup ...")

    all_perts, all_batches, all_celltypes = set(), set(), set()

    # search through data directory for all processed h5 / h5ad files
    (
        dm.dataset_name_list,
        dm.file_list,
        dm.data_type_list,
        dm.key_info_list,
        dm.selected_genes_list,
    ) = dm.get_dataset_names_and_paths(dm.data_args)

    # process to get all perturbation / cell type categories
    for dataset_name, dataset_path, data_type, key_info in tqdm(
        zip(dm.dataset_name_list, dm.file_list, dm.data_type_list, dm.key_info_list),
        "Processing meta data ...",
    ):
        with h5py.File(Path(dataset_path), "r") as f:
            if data_type == "RNA-seq":
                # cell type
                try:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_type_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['rnaseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

            elif data_type == "Perturb-seq":
                # cell line
                try:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{key_info['cell_line_key']}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

                # perturbation
                pert_arr = f[f"obs/{key_info['pert_col']}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                # ds_name + "_" + batch
                try:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{key_info['perturbseq_batch_col']}"][:]
                batches = set(safe_decode_array(batch_arr))
                batches = [dataset_name + "_" + x for x in batches]
                all_batches.update(batches)

    dm.pert_dict = {x: i for i, x in enumerate(sorted(all_perts))}
    dm.batch_dict = {x: i for i, x in enumerate(sorted(all_batches))}
    dm.cell_type_dict = {x: i for i, x in enumerate(sorted(all_celltypes))}

    dm.py_logger.info(
        f"#perturbation: {len(all_perts)}, "
        f"#cell type: {len(all_celltypes)}, "
        f"#batch: {len(all_batches)}"
    )

    # only sample pbmc
    dm.original_dataset_name_list = dm.dataset_name_list
    if dm.data_args.sample_pbmc_only:
        mask = np.array([True if "pbmc" in x.lower() else False for x in dm.dataset_name_list])
        assert mask[0] and (not mask[1:].any())
        dm.dataset_name_list = dm.dataset_name_list[:1]
        dm.file_list = dm.file_list[:1]
        dm.data_type_list = dm.data_type_list[:1]
        dm.key_info_list = dm.key_info_list[:1]

    if hasattr(dm.data_args, "sample_replogle_only") and dm.data_args.sample_replogle_only:
        mask = np.array([True if "replogle" in x.lower() else False for x in dm.dataset_name_list])
        dm.dataset_name_list = np.array(dm.dataset_name_list)[mask].tolist()
        dm.file_list = np.array(dm.file_list)[mask].tolist()
        dm.data_type_list = np.array(dm.data_type_list)[mask].tolist()
        dm.key_info_list = np.array(dm.key_info_list)[mask].tolist()
        new_selected_genes_list = []
        for ms in mask:
            if ms:
                new_selected_genes_list.append(dm.selected_genes_list[ms])
        dm.selected_genes_list = new_selected_genes_list

    if hasattr(dm.data_args, "sample_tahoe100m_only") and dm.data_args.sample_tahoe100m_only:
        mask = np.array([True if "tahoe100m" in x.lower() else False for x in dm.dataset_name_list])
        dm.dataset_name_list = np.array(dm.dataset_name_list)[mask].tolist()
        dm.file_list = np.array(dm.file_list)[mask].tolist()
        dm.data_type_list = np.array(dm.data_type_list)[mask].tolist()
        dm.key_info_list = np.array(dm.key_info_list)[mask].tolist()
        new_selected_genes_list = []
        for ms in mask:
            if ms:
                new_selected_genes_list.append(dm.selected_genes_list[ms])
        dm.selected_genes_list = new_selected_genes_list

    skip_tahoe100m = hasattr(dm.data_args, "skip_tahoe100m") and dm.data_args.skip_tahoe100m
    skip_pbmc = hasattr(dm.data_args, "skip_pbmc") and dm.data_args.skip_pbmc
    skip_replogle = hasattr(dm.data_args, "skip_replogle") and dm.data_args.skip_replogle
    skip_cellxgene = hasattr(dm.data_args, "skip_cellxgene") and dm.data_args.skip_cellxgene

    if skip_tahoe100m or skip_pbmc or skip_replogle or skip_cellxgene:
        dm.py_logger.info("Skipping datasets according to user settings ...")
        mask = []
        for x in dm.dataset_name_list:
            flag = True
            if skip_tahoe100m and ("tahoe100m" in x.lower()):
                flag = False
            if skip_pbmc and ("pbmc" in x.lower()):
                flag = False
            if skip_replogle and ("replogle" in x.lower()):
                flag = False
            if skip_cellxgene and ("cellxgene" in x.lower()):
                flag = False
            mask.append(flag)
        mask = np.array(mask)
        dm.dataset_name_list = np.array(dm.dataset_name_list)[mask].tolist()
        dm.file_list = np.array(dm.file_list)[mask].tolist()
        dm.data_type_list = np.array(dm.data_type_list)[mask].tolist()
        dm.key_info_list = np.array(dm.key_info_list)[mask].tolist()
        new_selected_genes_list = []
        for ms in mask:
            if ms:
                new_selected_genes_list.append(dm.selected_genes_list[ms])
        dm.selected_genes_list = new_selected_genes_list

    if dm.replace_pert_dict:
        dm.pert_dict = build_new_pert_dict(dm)
        dm.py_logger.info(f"Replacing new #perturbation: {len(dm.pert_dict)}")
