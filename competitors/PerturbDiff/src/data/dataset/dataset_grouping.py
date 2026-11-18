"""Module `data/dataset_grouping.py`."""
import hashlib
import os
import pickle
import tempfile

import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from src.common.utils import safe_decode_array


def grouping_cache_key(dataset, ds_name, group_by):
    """Content hash of everything the cell-set grouping depends on.

    The original cache filename was ``{ds_name}_{stage}``, which does not change
    when the split file, the fold, ``keep_control_cell`` or the grouping mode
    change -- so a stale pickle from a previous configuration was silently
    reused, and (worse) ``keep_control_cell`` changes the index space the cached
    positions refer to. Hashing the inputs makes a changed configuration a
    different file, so staleness is structurally impossible.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in (
        str(ds_name),
        str(dataset.stage),
        str(group_by),
        str(bool(getattr(dataset.data_args, "keep_control_cell", False))),
        str(dataset.control_type.get(ds_name)),
    ):
        h.update(part.encode())
        h.update(b"\x00")

    arr = dataset.data_indices[ds_name]
    pieces = [arr["perturb"], arr["control"]] if isinstance(arr, dict) else [arr]
    for piece in pieces:
        a = np.ascontiguousarray(np.asarray(piece, dtype=np.int64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _atomic_pickle_dump(obj, path):
    """Write a pickle atomically.

    Several training jobs launched at once all miss the cache and all write the
    same path. A plain ``open(path, "wb")`` is not atomic, so a concurrent reader
    can load a half-written file. Writing to a temp file in the same directory
    and renaming makes the swap atomic on POSIX; the duplicated compute is
    wasted work but never corruption.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_grouping_", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as fout:
            pickle.dump(obj, fout)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def split_out_control(dataset):
    """
    Remove control cells from virtual indexing
    """

    for ds_name in dataset._names:
        # ignore if it's RNA-seq
        if dataset.control_type[ds_name] is None:
            continue

        arr = dataset.data_indices[ds_name]
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
        target_code = (dataset.control_type[ds_name] == cache.pert_categories).nonzero()[0]
        assert len(target_code) == 1
        if len(arr) > 0:
            control_mask = cache.pert_codes[arr] == target_code
            assert max(arr) < cache.pert_codes.shape[0]
        else:
            control_mask = np.ones_like(arr, dtype=bool)
        control_indices = arr[control_mask]
        if dataset.data_args.keep_control_cell:
            perturb_indices = arr
        else:
            perturb_indices = arr[~control_mask]
        dataset.data_indices[ds_name] = {
            "control": control_indices,
            "perturb": perturb_indices,
        }

        if dataset.data_args.keep_control_cell:
            pass
        else:
            dataset.num_cell[ds_name] -= len(control_indices)
        dataset.py_logger.info(
            f"For dataset {ds_name} (different split), control cells: "
            f"{len(control_indices)}, perturb cells: {len(perturb_indices)}"
        )

def group_by_one_keys(dataset, arr, cache, key1_name):
    """Execute `group_by_one_keys` and return values used by downstream logic."""
    gp_indices, gp_cells = {}, {}

    key1_codes = getattr(cache, f"{key1_name}_codes")[arr]
    num_key1 = len(getattr(cache, f"{key1_name}_categories"))

    for i in range(num_key1):
        mask = key1_codes == i
        gp_indices[i] = mask.nonzero()[0]
        gp_cells[i] = len(gp_indices[i])

    return gp_indices, gp_cells

def group_by_two_keys(dataset, arr, cache, key1_name, key2_name):
    """Execute `group_by_two_keys` and return values used by downstream logic."""
    gp_indices, gp_cells = {}, {}

    key1_codes = getattr(cache, f"{key1_name}_codes")[arr]
    key2_codes = getattr(cache, f"{key2_name}_codes")[arr]
    num_key1 = len(getattr(cache, f"{key1_name}_categories"))
    num_key2 = len(getattr(cache, f"{key2_name}_categories"))

    for i in range(num_key1):
        for j in range(num_key2):
            mask = (key1_codes == i) & (key2_codes == j)
            gp_indices[(i, j)] = mask.nonzero()[0]
            gp_cells[(i, j)] = len(gp_indices[(i, j)])

    return gp_indices, gp_cells

def group_by_three_keys(dataset, arr, cache, key1_name, key2_name, key3_name):
    """
    Group by three keys.

    :param dataset: Input `dataset` value.
    :param arr: Source 1-D array for indexed extraction.
    :param cache: Input `cache` value.
    :param key1_name: Input `key1_name` value.
    :param key2_name: Input `key2_name` value.
    :param key3_name: Input `key3_name` value.
    :return: Computed output(s) for this function.
    """
    gp_indices, gp_cells = {}, {}

    key1_codes = getattr(cache, f"{key1_name}_codes")[arr]
    key2_codes = getattr(cache, f"{key2_name}_codes")[arr]
    key3_codes = getattr(cache, f"{key3_name}_codes")[arr]
    num_key1 = len(getattr(cache, f"{key1_name}_categories"))
    num_key2 = len(getattr(cache, f"{key2_name}_categories"))
    num_key3 = len(getattr(cache, f"{key3_name}_categories"))

    dataset.py_logger.info(
        f"group by three keys: {key1_name}: {num_key1}, {key2_name}: {num_key2}, {key3_name}: {num_key3}"
    )

    for i in tqdm(range(num_key1)):
        for j in range(num_key2):
            for k in range(num_key3):
                mask = (key1_codes == i) & (key2_codes == j) & (key3_codes == k)
                gp_indices[(i, j, k)] = mask.nonzero()[0]
                gp_cells[(i, j, k)] = len(gp_indices[(i, j, k)])

    return gp_indices, gp_cells

def register_mapping_indices(dataset):
    """
    During indexing, it's default for indexing perturb-seq, and we select
        from mapped control cells for it. As for RNA-seq, we also select one
        cell that shares the same cell type and optionally batch.

    In other words, control cells in perturb-seq and the whole RNA-seq would be selected among.
    And we pre-group their indices for convenient mapping and selection
    """
    dataset.grouped_data_indices, dataset.grouped_num_cell = {}, {}
    for ds_name in dataset.data_indices.keys():
        arr = dataset.data_indices[ds_name]
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

        if dataset.control_type[ds_name] is not None:
            arr = arr["control"]
        else:
            # RNA-seq
            arr = np.array([], dtype=np.int64)
        if len(arr) > 0:
            gp_indices, gp_cells = group_by_two_keys(dataset, arr, cache, key1_name="cell_type", key2_name="batch")
        else:
            gp_indices, gp_cells = {}, {}
        dataset.grouped_data_indices[ds_name] = gp_indices
        dataset.grouped_num_cell[ds_name] = gp_cells

    if not hasattr(dataset.data_args, "use_cell_set") or dataset.data_args.use_cell_set is None:
        return

    # Which covariates define a cell set.
    #   "pert_celltype_batch" (default) -- upstream PerturbDiff: sets are
    #       homogeneous in batch as well as perturbation.
    #   "pert_celltype"                 -- STATE's sentence grouping: sets may
    #       span batches; batch stays a per-cell covariate on the CovEncoder.
    # On data with many batches per perturbation the 3-key grouping can leave
    # every group smaller than one cell set, so the sampler fills sets almost
    # entirely with replacement-drawn duplicates. See ZARR_BENCHMARK.md.
    group_by = getattr(dataset.data_args, "cell_set_group_by", "pert_celltype_batch")
    if group_by not in ("pert_celltype_batch", "pert_celltype"):
        raise ValueError(
            f"Unknown data.cell_set_group_by={group_by!r}; expected "
            f"'pert_celltype_batch' or 'pert_celltype'."
        )

    dataset.cell_set_group_by = group_by

    dataset.grouped_pert_data_indices, dataset.grouped_pert_num_cell = {}, {}
    for ds_name in dataset.data_indices.keys():
        # Cache identity = content hash of the split indices + grouping mode +
        # keep_control_cell + control type. See grouping_cache_key().
        cache_tag = f"{ds_name}_{dataset.stage}_{group_by}_{grouping_cache_key(dataset, ds_name, group_by)}"
        indices_cache_file = os.path.join(dataset.data_args.indices_cache_dir, f"grouped_pert_data_indices_{cache_tag}.pkl")
        num_cell_cache_file = os.path.join(dataset.data_args.indices_cache_dir, f"grouped_pert_num_cell_{cache_tag}.pkl")

        if (
            (not getattr(dataset.data_args, "skip_cached_indices", False))
            and os.path.exists(indices_cache_file)
            and os.path.exists(num_cell_cache_file)
        ):
            with open(indices_cache_file, "rb") as fin:
                gp_indices = pickle.load(fin)
            with open(num_cell_cache_file, "rb") as fin:
                gp_cells = pickle.load(fin)

            dataset.grouped_pert_data_indices[ds_name] = gp_indices
            dataset.grouped_pert_num_cell[ds_name] = gp_cells
            continue

        arr = dataset.data_indices[ds_name]
        cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]

        if dataset.control_type[ds_name] is not None:
            arr = arr["perturb"]
            if len(arr) > 0:
                if group_by == "pert_celltype":
                    gp_indices, gp_cells = group_by_two_keys(
                        dataset,
                        arr,
                        cache,
                        key1_name="pert",
                        key2_name="cell_type",
                    )
                else:
                    gp_indices, gp_cells = group_by_three_keys(
                        dataset,
                        arr,
                        cache,
                        key1_name="pert",
                        key2_name="cell_type",
                        key3_name="batch",
                    )
            else:
                gp_indices, gp_cells = {}, {}
        else:
            # RNA-seq
            if "cellxgene" in ds_name.lower():
                gp_indices, gp_cells = group_by_one_keys(dataset, arr, cache, key1_name="cell_type")
            else:
                raise NotImplementedError

        dataset.grouped_pert_data_indices[ds_name] = gp_indices
        dataset.grouped_pert_num_cell[ds_name] = gp_cells

        _atomic_pickle_dump(dataset.grouped_pert_data_indices[ds_name], indices_cache_file)
        _atomic_pickle_dump(dataset.grouped_pert_num_cell[ds_name], num_cell_cache_file)

    if dataset.data_args.keep_control_cell:
        for ds_name in dataset.data_indices.keys():
            arr = dataset.data_indices[ds_name]
            cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
            if dataset.control_type[ds_name] is None:
                continue
            target_code = (dataset.control_type[ds_name] == cache.pert_categories).nonzero()[0]

            assert len(target_code) == 1
            target_code = target_code.item()

            # grouped_data_indices holds positions into data_indices["control"]
            # (that is how mapping_cells consumes it), but grouped_pert_data_indices
            # is consumed by the sampler through data_indices["perturb"]
            # (dataset_core.compute_index). Copying the values across unchanged --
            # as the original code did -- reinterprets control-space positions as
            # perturb-space ones, so the "control" cell sets fill with unrelated
            # cells. Translate through the global row id instead.
            perturb_rows = dataset.data_indices[ds_name]["perturb"]
            control_rows = dataset.data_indices[ds_name]["control"]
            pos_in_perturb = {int(r): i for i, r in enumerate(perturb_rows)}

            def _to_perturb_space(positions):
                if len(positions) == 0:
                    return np.array([], dtype=np.int64)
                rows = control_rows[positions]
                return np.fromiter(
                    (pos_in_perturb[int(r)] for r in rows), dtype=np.int64, count=len(rows)
                )

            if group_by == "pert_celltype":
                # Collapse the control pools over batch, so a control set spans
                # batches exactly like a perturbed one in this mode.
                merged = {}
                for (ct_code, _batch_code), v in dataset.grouped_data_indices[ds_name].items():
                    key = (target_code, ct_code)
                    translated = _to_perturb_space(v)
                    prev = merged.get(key)
                    merged[key] = (
                        translated if prev is None else np.concatenate([prev, translated])
                    )
                for key, v in merged.items():
                    dataset.grouped_pert_data_indices[ds_name][key] = v
                    dataset.grouped_pert_num_cell[ds_name][key] = len(v)
            else:
                for k, v in dataset.grouped_data_indices[ds_name].items():
                    translated = _to_perturb_space(v)
                    key = (target_code, k[0], k[1])
                    dataset.grouped_pert_data_indices[ds_name][key] = translated
                    dataset.grouped_pert_num_cell[ds_name][key] = len(translated)

def get_selected_gene_vars(dataset):
    """
    Get selected gene vars.

    :param dataset: Input `dataset` value.
    :return: Requested object(s) for downstream use.
    """
    if dataset.data_args.selected_gene_file is None:
        dataset.selected_genes = {}
        for ds_name in dataset._names:
            dataset.selected_genes[ds_name] = dataset.selected_genes_list[ds_name]

    else:
        with open(dataset.data_args.selected_gene_file, "rb") as fin:
            dataset.selected_genes = pickle.load(fin)

        if isinstance(dataset.selected_genes, set):
            dataset.selected_genes = sorted(dataset.selected_genes)

    dataset.selected_gene_mask, dataset.gene_vars = {}, {}

    for ds_name in dataset._names:
        h5f = dataset.store.dataset_file(ds_name)
        try:
            all_genes = safe_decode_array((h5f["/var/_index"]))
        except:
            try:
                all_genes = safe_decode_array((h5f["/var/gene_name_index"]))
            except:
                try:
                    all_genes = safe_decode_array((h5f["/var/ensembl_id"]))
                except:
                    try:
                        all_genes = safe_decode_array((h5f["/var/gene_name"]))
                    except:
                        all_genes = safe_decode_array((h5f["/var/gene_short_name"]))

        if "cellxgene" in ds_name:
            gm = pd.read_csv(dataset.data_args.gene_mapping_file)
            gmap = {k: v for k, v in zip(gm["id"], gm["name"])}
            new_all_genes = []
            not_found = 0
            for x in all_genes:
                if x in gmap:
                    new_all_genes.append(gmap[x])
                else:
                    new_all_genes.append(x)
                    not_found += 1
            dataset.py_logger.info(
                f"[Warning]: {not_found} gene names in RNA-seq {ds_name} are not found in gene mapping dict"
            )
            dataset.gene_vars[ds_name] = np.array(new_all_genes)
            all_genes = new_all_genes
        else:
            dataset.gene_vars[ds_name] = all_genes

        if isinstance(dataset.selected_genes, dict):
            mask = np.isin(all_genes, dataset.selected_genes[ds_name])
        else:
            mask = np.isin(all_genes, dataset.selected_genes)
        dataset.selected_gene_mask[ds_name] = mask
        dataset.py_logger.info(
            f"Shared gene count in data {ds_name}: \t\t---- ({mask.sum()} / {len(dataset.selected_genes)}; original: {len(all_genes)})"
        )
