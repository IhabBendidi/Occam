"""Sampling generation core split from rawdata_diffusion_sample.py (logic-preserving)."""

import gc
import time

import anndata
import numpy as np
import torch
from geomloss import SamplesLoss
from pytorch_lightning.utilities import move_data_to_device
from sklearn.metrics import r2_score

from src.apps.sampling.sampling_io import save_adata
from pathlib import Path

from src.apps.sampling.sampling_generation_helpers import (
    build_obs_data,
    build_obs_data_zarr,
    build_zarr_controls,
    build_self_condition,
    build_x_ctrl,
    build_gene_embedding_cache,
    collect_batch_covariates,
    load_ctrl_adata,
    load_selected_genes,
    resolve_sampling_runner,
)
def _split_dataloader(datamodule, split: str):
    """Pick the val_dataloader() entry for a named split.

    Indexing positionally is fragile: the order comes from `datamodule.all_split_names`, so a change
    there would silently start scoring the wrong split -- and validation and test differ by 22k
    cells here, which is the kind of mistake that produces a plausible number.
    """
    loaders = datamodule.val_dataloader()
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    names = list(getattr(datamodule, "all_split_names", ["validation", "test"]))
    if split not in names:
        raise ValueError(f"split {split!r} not in datamodule.all_split_names={names}")
    idx = names.index(split)
    if idx >= len(loaders):
        raise ValueError(
            f"split {split!r} is at index {idx} of {names} but val_dataloader() returned "
            f"{len(loaders)} loader(s); the split is probably empty."
        )
    loader = loaders[idx]
    n = len(getattr(loader, "dataset", []) or [])
    if n == 0:
        raise ValueError(
            f"split {split!r} has an empty dataloader. On this benchmark that means the split file "
            f"has no rows for it -- point the manifest at a split JSON with a real {split} list."
        )
    return loader


def generate_samples(
    model,
    diffusion,
    cfg, 
    device, 
    logger, 
    datamodule, 
    pca_for_decode=None,
    store_noised_truth=False,
    split="test",
):
    """
    Dispatcher that preserves original behavior:
      - If `pca_for_decode` is None -> sample directly in **original gene space**.
      - If `pca_for_decode` is provided -> delegate to `generate_samples_pca` and return its outputs.

    Gene-space mode:
      - pert_emb and samples are [B, S, G] (G ~ 2000)
      - R^2 and MMD are computed in gene space (this is the working space, not a 'decoded' space)
      - Returns (truths, samples, trajectories, decoded=None)
    """
    # `val_dataloader()` returns one loader per entry of `all_split_names`, which is
    # ["validation", "test"] -- so index 0 is validation and index 1 is test. This used to be a
    # hardcoded [1]; it is selected by name now so the same sampling path can score validation
    # during training, which is what checkpoint selection needs.
    dataloader = _split_dataloader(datamodule, split)

    dataloader_iter = iter(dataloader)
    model = model.to(device)
    model.eval()

    # Model type helpers
    model_type = getattr(getattr(model, "model_cfg", None), "model_type", None)
    if model_type is None:
        model_type = getattr(cfg.model, "model_type", None)

    # Set up sampling parameters
    batch_size = cfg.sampling.batch_size
    if cfg.data.use_cell_set is not None:
        batch_size = int(batch_size // cfg.data.use_cell_set)
    # Evaluation must be deterministic and cover the whole test split, so the
    # numbers are comparable across runs and against other models.
    total_batches = len(dataloader)
    requested_batches = cfg.sampling.num_sampled_batches
    if requested_batches is None:
        num_sampled_batches = total_batches
    else:
        # Cap it: asking for more batches than exist used to raise StopIteration.
        num_sampled_batches = min(int(requested_batches), total_batches)
        if num_sampled_batches < total_batches:
            logger.warning(
                "PARTIAL EVAL: sampling.num_sampled_batches=%d of %d batches "
                "(~%.1f%% of the test split). Every metric below -- and the saved "
                "adatas -- cover only this subset. Set sampling.num_sampled_batches=null "
                "for full coverage.",
                num_sampled_batches,
                total_batches,
                100.0 * num_sampled_batches / max(total_batches, 1),
            )
    num_samples = len(dataloader.sampler)

    # The test loader is built with shuffle=False (see data_module.build_val_dataloaders),
    # which makes the cell-set order sequential and reproducible. Guard it, so a
    # future change to the loader cannot silently randomise the evaluation order.
    if getattr(dataloader.sampler, "shuffle", False):
        logger.warning(
            "Test dataloader sampler has shuffle=True; generation order is not "
            "reproducible. Expected shuffle=False for evaluation."
        )

    use_ddim = cfg.sampling.use_ddim
    clip_denoised = cfg.sampling.clip_denoised
    progress = cfg.sampling.progress

    # Resolve sampling dimensionality from config if available.
    input_dim = (
        getattr(cfg.model, "input_dim", None)
        or getattr(cfg.model, "output_size", None)
        or getattr(cfg.model, "gene_dim", None)
    )

    logger.info(f"Generating {num_samples} samples with batch size {batch_size}")
    logger.info(f"Using {'DDIM' if use_ddim else 'DDPM'} sampling")
    
    all_truths = []
    all_samples = []
    all_trajectories = []
    all_covariates = []
    
    # Generate samples in batches

    cell_set_number = cfg.data.use_cell_set if hasattr(cfg.data, 'use_cell_set') and cfg.data.use_cell_set is not None else 1

    data_args = getattr(dataloader.dataset, "data_args", None)
    normalize_counts = getattr(data_args, "normalize_counts", None) if data_args is not None else None

    assert store_noised_truth == False
    
    # load_selected_genes validates the panel against data.pad_length, so the
    # old hard-coded `assert len(_genes) == 2000` is no longer needed.
    _genes = load_selected_genes(cfg)

    with torch.no_grad():
        for batch_idx in range(num_sampled_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            logger.info(f"Generating batch {batch_idx + 1}/{num_sampled_batches} (size: {current_batch_size})")
            
            sample_fn, sampling_kwargs = resolve_sampling_runner(cfg, diffusion, use_ddim)

            batch_data = next(dataloader_iter)
            
            batch_data = move_data_to_device(batch_data, device)

            batch_data["batch_emb"] = model._encode_covariates(batch_data)
            pert_emb = batch_data["pert_emb"]
            # Fallback to runtime batch shape when config does not expose a sampling dim.
            resolved_input_dim = int(input_dim) if input_dim is not None else int(pert_emb.shape[-1])

            if len(pert_emb) != current_batch_size:
                logger.warning("batch size mismatch detected; this may be the last batch")
            current_batch_size = len(pert_emb)
            device = pert_emb.device
            gene_emb = build_gene_embedding_cache(model, batch_data, device)
            self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
            # Generate samples
            start_time = time.time()
            mmd_loss = SamplesLoss(loss="energy", blur=0.05, scaling=0.5).to(device)

            mask = ~batch_data["is_padded_list"].bool()

            sample, traj = sample_fn(
                model.model,
                (current_batch_size, cell_set_number, resolved_input_dim),
                self_condition=self_condition,
                clip_denoised=clip_denoised,
                device=device,
                progress=progress,
                **sampling_kwargs
            )

            # remove duplicated sample that are used for padding
            pert_emb = pert_emb[mask]
            if sample is not None:
                sample = sample[mask]

            logger.debug("sample: %s", sample)
            logger.debug("pert_emb: %s", pert_emb)

            pert_emb_cpu = pert_emb.detach().cpu()
            if sample is not None:
                sample_cpu = sample.detach().cpu()
            
            # Store samples
            np_mask = np.isin(batch_data["col_genes"][0], _genes)
            truth_np = pert_emb_cpu.numpy()[:, np_mask]
            if sample is not None:
                sample_np = sample_cpu.numpy()[:, np_mask]
            
            if sample is not None:

                r2_metric = r2_score(truth_np.mean(0), sample_np.mean(0))
                torch_mask = torch.as_tensor(np_mask, device=device, dtype=torch.bool)
                sample_eval = sample[:, torch_mask]
                truth_eval = pert_emb[:, torch_mask]
                mmd_metric = mmd_loss(sample_eval, truth_eval).item()

                batch_time = time.time() - start_time
                logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s, r2_metric for this batch: {r2_metric}, mmd_metric for this batch: {mmd_metric}")
            
            all_truths.append(truth_np)
            if sample is not None:
                all_samples.append(sample_np)

            # covariates
            all_covariates.extend(collect_batch_covariates(batch_data, dataloader, datamodule, mask))

    # Concatenate all samples
    all_truths = np.concatenate(all_truths, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)

    logger.info(f"Generated {all_samples.shape[0]} samples with shape {all_samples.shape}")

    # Coverage check: with num_sampled_batches=null every real (non-padded) cell
    # of the test split must appear exactly once. A mismatch means the metrics
    # below describe a different cell population than the split does.
    expected_cells = getattr(dataloader.sampler, "original_total", None)
    if expected_cells is not None:
        if num_sampled_batches < total_batches:
            logger.warning(
                "PARTIAL EVAL: generated %d of %d test cells (%d of %d batches).",
                all_samples.shape[0], expected_cells, num_sampled_batches, total_batches,
            )
        elif all_samples.shape[0] != expected_cells:
            logger.warning(
                "Coverage mismatch: generated %d cells but the test split holds %d "
                "real cells. Metrics may not describe the full split.",
                all_samples.shape[0], expected_cells,
            )
        else:
            logger.info("Full test-split coverage: %d cells.", expected_cells)

    r2_metric = r2_score(all_truths.mean(0), all_samples.mean(0))
    logger.info(f"Overall r2_metric: {r2_metric}")


    all_pert = np.concatenate([x[0] for x in all_covariates], axis=0)
    all_celltype = np.concatenate([x[1] for x in all_covariates], axis=0)
    all_batch = np.concatenate([x[2] for x in all_covariates], axis=0)
        
    var_index = _genes
    source_format = getattr(cfg.data, "source_format", "h5ad")

    if source_format == "zarr":
        # Controls live in the test split itself; no separate control h5ad.
        X_ctrl, ctrl_obs = build_zarr_controls(cfg, datamodule, _genes, logger)
        ctrl_adata = None
    else:
        ctrl_adata, _ = load_ctrl_adata(cfg)
        X_ctrl = build_x_ctrl(ctrl_adata, _genes, cfg)

    if normalize_counts is not None and normalize_counts is not False:
        all_samples *= normalize_counts
        all_truths *= normalize_counts

    if source_format == "zarr":
        obs = build_obs_data_zarr(cfg, all_pert, all_celltype, all_batch, ctrl_obs)
    else:
        obs = build_obs_data(cfg, all_pert, all_celltype, all_batch, ctrl_adata)
    
    if source_format != "zarr" and len(np_mask) > 2000:  # e.g. data_name = "Tahoe100mPBMCPretrain"
        logger.info("Restricting to downstream data selected genes only...")
        assert (ctrl_adata.var.index[ctrl_adata.var.highly_variable] == _genes).all()

        # unmerged data setting, no need of using sel_mask
        sel_mask = np.isin(batch_data["col_genes"][0], _genes)
        if all_samples.shape[-1] != len(_genes):
        
            assert sel_mask.sum() == 2000
            all_samples = all_samples[:, sel_mask]
            all_truths = all_truths[:, sel_mask]

        # reorder
        cur_genes = np.array(batch_data["col_genes"][0])[sel_mask].tolist()
        sort_idx = [cur_genes.index(g) for g in _genes]
        assert (np.array(cur_genes)[sort_idx] == _genes).all()
        all_samples = all_samples[:, sort_idx]
        all_truths = all_truths[:, sort_idx]

        var_index = _genes

    pred_adata = anndata.AnnData(X=np.concatenate([all_samples, X_ctrl]), obs=obs)
    true_adata = anndata.AnnData(X=np.concatenate([all_truths, X_ctrl]), obs=obs)
    if var_index is not None:
        pred_adata.var.index = var_index
        true_adata.var.index = var_index

    # Use a shared timestamp so that auxiliary files align with the main outputs.
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    save_adata(pred_adata, true_adata, cfg, logger, timestamp=run_timestamp)

    # cell-eval: the only metrics comparable to what we have for STATE. R^2 and
    # the energy-MMD above are PerturbDiff-native and cannot be compared across
    # models.
    if getattr(cfg.sampling, "run_cell_eval", False):
        from src.apps.sampling.sampling_eval_celleval import run_cell_eval

        if cfg.sampling.output_dir is None:
            eval_dir = Path(cfg.save_dir_path) / "samples" / f"cell_eval_{run_timestamp}"
        else:
            eval_dir = Path(cfg.sampling.output_dir) / f"cell_eval_{run_timestamp}"
        run_cell_eval(
            pred_adata,
            true_adata,
            output_dir=eval_dir,
            pert_col=str(cfg.data.pert_col),
            cell_type_key=str(cfg.data.cell_line_key),
            control_pert=str(cfg.data.control_pert),
            profile=str(getattr(cfg.sampling, "cell_eval_profile", "full")),
            logger=logger,
        )

    gc.collect()

    return all_truths, all_samples, all_trajectories, None
