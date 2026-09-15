"""Stage entry points: :func:`run_pretrain` and :func:`run_finetune`.

Extracted from ``training_pipeline.py:2244-2884``, where these were
the research tree's two stage entry points (see docs/reproducing_chain_a.md
for the symbol map).

The orchestrator always uses the **standalone** finetune path -- load stage 1's
saved checkpoint from disk, then train -- because that is what produced the
released weights. The research tree also had a "tandem" mode
(``--finetune`` alongside pretraining) that handed the live model straight into
finetuning without a save/load round trip. It is deliberately not carried over,
for three reasons:

1. Standalone loads with ``strict=False``, which drops
   ``_recon_loss_fn.log_theta`` and re-initialises the loss module. Tandem keeps
   the live nb_softplus module in memory until ``finetune()`` swaps it. The two
   differ in parameter set at step 0.
2. Standalone resets ``best_val_loss`` and starts a fresh run directory.
3. Tandem had a latent double-transform bug: with
   ``ft_batch_size != batch_size`` it rebuilt the dataset with CPU transforms
   while ``_defer_transforms`` was still set, applying NormalizePerCell+Log1p on
   CPU *and* again on GPU. The released chain avoided it by running standalone.
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional, Tuple, Any
import pickle
from tqdm import tqdm
import warnings


from occam.model import AutoEncoder, create_occam_model
from occam.util import AverageMeter
from occam.data.datasets import (
    apply_noise, OccamAdataDataset, OccamChunkedDataset,
    optim_train_val_chunk_split, compute_global_chunk_info,
)
from occam.transforms import NormalizePerCell, Log1p, Scale, compute_gene_stats
from occam.run_dir import get_output_manager, ExperimentManager
from occam.evaluation.perturbation import tvn_on_controls, aggregate, known_relationship_benchmark, pert_signal_consistency_benchmark, DEFAULT_RECALL_THR_PAIRS

try:
    from occam.evaluation import run_end_of_training_benchmark
    _BENCHMARK_AVAILABLE = True
except ImportError:
    _BENCHMARK_AVAILABLE = False

from occam._wandb_guard import WANDB_AVAILABLE, wandb
from occam.ddp import (
    cleanup_distributed, get_rank, is_distributed, is_main_process,
    is_rank0_env, setup_distributed, _clean_state_dict,
    _inner_model_for_weights,
)
from occam.mae.config import PretrainConfig
from occam.mae.finetune_config import FinetuneConfig
from occam.mae.trainer import OccamTrainer

__all__ = ["run_pretrain", "run_finetune", "create_dataloader", "create_train_val_split"]


def _seed_worker(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def create_dataloader(dataset, batch_size, shuffle=True, drop_last=False, is_distributed=False, num_workers=0, pin_memory=False, seed=42):
    """
    Create a DataLoader with optional DistributedSampler for DDP.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle data (ignored if using DistributedSampler)
        drop_last: Whether to drop last incomplete batch
        is_distributed: Whether to use DistributedSampler
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        seed: Random seed for the DataLoader's generator
        
    Returns:
        DataLoader
    """
    g = torch.Generator()
    g.manual_seed(seed)

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed)
        loader_shuffle = False
    else:
        sampler = None
        loader_shuffle = shuffle
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        generator=g,
        worker_init_fn=_seed_worker,
    )


def create_train_val_split(dataset, val_split: float = 0.1, seed: int = 42):
    """
    Create train/validation split from a single dataset.
    
    Args:
        dataset: Original dataset to split
        val_split: Fraction for validation (default 0.1 = 10%)
        seed: Random seed for reproducible split
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set random seed for reproducible split
    generator = torch.Generator().manual_seed(seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Create split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"📊 Created train/val split: {train_size} train, {val_size} val ({val_split*100:.1f}%)")
    
    return train_dataset, val_dataset


def run_pretrain(
    train_adata_path: str,
    val_adata_path: Optional[str] = None,
    config: Optional[PretrainConfig] = None,
    gene_name_col: Optional[str] = None,
    val_gene_name_col: Optional[str] = None,
    save_dir: str = './occam_pretrain',
    device: str = 'cuda',
    experiment_description: str = "",
    use_output_manager: bool = True,
    skip_wandb_finish: bool = False,
) -> Tuple[OccamTrainer, dict, DataLoader, Optional[DataLoader]]:
    """Train Occam from an h5ad file.

    Builds a :class:`OccamAdataDataset` (backed **or** in-memory), creates
    DataLoaders, trains the model and returns the trainer together with a
    metadata dict describing the gene names and transforms used.

    Args:
        train_adata_path: Path to training ``.h5ad`` file.
        val_adata_path: Optional path to validation ``.h5ad`` file.  When
            *None*, the training set is split using ``config.val_split``.
        config: Training configuration.  Defaults to :class:`PretrainConfig`.
        gene_name_col: Column in ``adata.var`` for gene names (auto-detected
            when *None*).
        val_gene_name_col: Gene name column for the validation file.
        save_dir: Fallback directory for saving results.
        device: Device for training (``'cuda'`` or ``'cpu'``).
        experiment_description: Free-text experiment description.
        use_output_manager: Use the centralised output manager.

    Returns:
        ``(trainer, dataset_info)`` where *dataset_info* is a dict containing
        ``gene_names``, ``transforms``, ``target_sum``, etc.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Initialize distributed process group early ----
    # Must happen before dataset creation so rank/world_size are available
    # for chunk partitioning in optim mode.
    if not dist.is_initialized():
        setup_distributed()

    # ---- Experiment management (rank 0) ----
    experiment: Optional[ExperimentManager] = None
    misc_save_dir = save_dir
    if use_output_manager and is_rank0_env():
        try:
            output_manager = get_output_manager()
            experiment = output_manager.create_experiment(
                experiment_type="training",
                description=experiment_description,
                config=(config.to_dict() if config is not None else {}),
                tags=["Occam", "training"],
            )
            misc_save_dir = str(experiment.experiment_dir)
        except Exception as e:
            print(f"Failed to create experiment via OutputManager: {e}")
            experiment = None

    if config is None:
        config = PretrainConfig()

    # ---- Build transform list ----
    transforms = []
    if not config.skip_normalize:
        transforms.append(NormalizePerCell(config.target_sum))
    if not config.skip_log1p:
        transforms.append(Log1p())
    if config.scale:
        print("Computing per-gene statistics for z-score scaling...")
        stats = compute_gene_stats(train_adata_path, target_sum=config.target_sum)
        transforms.append(Scale(
            torch.tensor(stats["mean"]),
            torch.tensor(stats["std"]),
        ))
        print(f"  Stats computed over {stats['n_cells']:,} cells")

    # ---- Create training dataset ----
    if config.data_mode == "optim":
        _rank = dist.get_rank() if dist.is_initialized() else 0
        _world = dist.get_world_size() if dist.is_initialized() else 1

        # Determine chunk indices for train (and optionally val)
        _optim_train_chunk_ids = None
        _optim_val_chunk_ids = None
        if val_adata_path is None:
            # Chunk-based train/val split (works for both single file and folder)
            _n_obs, _n_chunks = compute_global_chunk_info(
                train_adata_path, config.optim_chunk_size
            )
            _optim_train_chunk_ids, _optim_val_chunk_ids = optim_train_val_chunk_split(
                total_chunks=_n_chunks,
                val_fraction=config.val_split, seed=42,
            )
            print(
                f"Optim chunk-based train/val split: "
                f"{len(_optim_train_chunk_ids)} train chunks, "
                f"{len(_optim_val_chunk_ids)} val chunks "
                f"({config.val_split*100:.1f}%)"
            )

        print(f"Creating training dataset (mode=optim, optimized chunked loader)...")
        train_dataset = OccamChunkedDataset(
            adata_path=train_adata_path,
            batch_size=config.batch_size,
            transforms=[],
            gene_name_col=gene_name_col,
            chunk_size=config.optim_chunk_size,
            buffer_chunks=config.optim_buffer_chunks,
            seed=42,
            shuffle=True,
            drop_last=True,
            rank=_rank,
            world_size=_world,
            chunk_indices=_optim_train_chunk_ids,
            track_presence=config.gene_presence_mask,
            min_gene_overlap_pct=config.min_gene_overlap_pct,
        )
    else:
        print(f"Creating training dataset (mode={config.data_mode})...")
        train_dataset = OccamAdataDataset(
            adata_path=train_adata_path,
            mode=config.data_mode,
            transforms=transforms,
            gene_name_col=gene_name_col,
        )

    # ---- Initialize trainer (sets up DDP / multi-GPU) ----
    exp_dir_name_for_notes = os.path.basename(misc_save_dir) if misc_save_dir else None
    trainer = OccamTrainer(config, device, experiment_dir_name=exp_dir_name_for_notes)

    # Attach gene name and transform info for evaluation & checkpointing
    trainer._gene_names = train_dataset.gene_names
    trainer._dataset_info = train_dataset.get_dataset_info()
    trainer._transforms = transforms
    if config.data_mode == "optim":
        trainer._defer_transforms = True  # Transforms run on GPU in train/val loops
        # In optim mode, dataset has transforms=[] (deferred to GPU), so
        # dataset_info.target_sum is None. Fill it from the real transforms.
        if trainer._dataset_info.get('target_sum') is None:
            for t in transforms:
                if isinstance(t, NormalizePerCell):
                    trainer._dataset_info['target_sum'] = t.target_sum
                    break
            if trainer._dataset_info.get('target_sum') is None:
                trainer._dataset_info['target_sum'] = config.target_sum
        # Store folder summary for wandb logging (per-file samples/genes breakdown)
        if hasattr(train_dataset, 'get_folder_summary'):
            trainer._folder_summary = train_dataset.get_folder_summary()

    # For optim mode, setup model explicitly (avoids consuming an iterable batch)
    if config.data_mode == "optim":
        trainer._setup_model(train_dataset.num_genes)

    # ---- Validation dataset & dataloaders ----
    val_loader = None
    if config.data_mode == "optim":
        # Optim: train loader uses batch_size=None (pre-batched)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

        if val_adata_path is not None:
            print("Creating validation dataset (in_memory mode)...")
            val_dataset = OccamAdataDataset(
                adata_path=val_adata_path,
                mode="in_memory",
                transforms=[],
                gene_name_col=val_gene_name_col,
                gene_vocabulary=train_dataset.gene_names,
            )
            val_loader = create_dataloader(
                val_dataset, batch_size=config.batch_size * 2, shuffle=False,
                is_distributed=trainer.is_distributed, num_workers=4, pin_memory=True,
            )
        else:
            # Val from reserved chunks (same file, no extra memory)
            print(f"Creating validation dataset from reserved chunks...")
            val_dataset = OccamChunkedDataset(
                adata_path=train_adata_path,
                batch_size=config.batch_size * 2,
                transforms=[],
                gene_name_col=gene_name_col,
                chunk_size=config.optim_chunk_size,
                buffer_chunks=min(config.optim_buffer_chunks, len(_optim_val_chunk_ids)),
                seed=42,
                shuffle=False,
                drop_last=False,
                rank=_rank,
                world_size=_world,
                chunk_indices=_optim_val_chunk_ids,
                track_presence=config.gene_presence_mask,
                min_gene_overlap_pct=config.min_gene_overlap_pct,
            )
            val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
    elif val_adata_path is not None:
        print("Creating validation dataset...")
        val_dataset = OccamAdataDataset(
            adata_path=val_adata_path,
            mode=config.data_mode,
            transforms=transforms,
            gene_name_col=val_gene_name_col,
            gene_vocabulary=train_dataset.gene_names,
        )

        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            is_distributed=trainer.is_distributed,
            num_workers=4 if config.data_mode == "in_memory" else 0,
            pin_memory=config.data_mode == "in_memory",
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            is_distributed=trainer.is_distributed,
            num_workers=4 if config.data_mode == "in_memory" else 0,
            pin_memory=config.data_mode == "in_memory",
        )
    else:
        # Split training data
        print(f"No validation data provided. Creating {config.val_split*100:.1f}% split...")
        train_split, val_split = create_train_val_split(
            train_dataset,
            val_split=config.val_split,
            seed=42,
        )

        train_loader = create_dataloader(
            train_split,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            is_distributed=trainer.is_distributed,
            num_workers=4 if config.data_mode == "in_memory" else 0,
            pin_memory=config.data_mode == "in_memory",
        )
        val_loader = create_dataloader(
            val_split,
            batch_size=config.batch_size * 2,
            shuffle=False,
            is_distributed=trainer.is_distributed,
            num_workers=4 if config.data_mode == "in_memory" else 0,
            pin_memory=config.data_mode == "in_memory",
        )

    # ---- Save config ----
    if is_rank0_env():
        with open(os.path.join(misc_save_dir, "training_config.pkl"), "wb") as f:
            pickle.dump(config.to_dict(), f)

    # ---- Train ----
    training_results = trainer.train(
        train_loader,
        val_loader,
        save_dir=save_dir,
        experiment_description=experiment_description,
        use_output_manager=use_output_manager,
        existing_experiment=experiment,
        skip_wandb_finish=skip_wandb_finish,
    )

    # ---- Save final embeddings (rank 0) ----
    # Use is_rank0_env() only — is_main_process() is unreliable after cleanup_distributed()
    if not trainer.is_distributed or is_rank0_env():
        if config.data_mode != "optim":
            print("Extracting final embeddings...")
            embeddings, labels = trainer.extract_embeddings(train_loader)
            np.save(os.path.join(misc_save_dir, "train_embeddings.npy"), embeddings)
            np.save(os.path.join(misc_save_dir, "train_labels.npy"), labels)
        else:
            print("Skipping train embedding extraction for optim mode (iterable dataset).")

        if val_loader is not None:
            val_embeddings, val_labels = trainer.extract_embeddings(val_loader)
            np.save(os.path.join(misc_save_dir, "val_embeddings.npy"), val_embeddings)
            np.save(os.path.join(misc_save_dir, "val_labels.npy"), val_labels)

    dataset_info = train_dataset.get_dataset_info()
    print(f"Training completed! Results saved to {misc_save_dir}")
    return trainer, dataset_info, train_loader, val_loader


def run_finetune(
    checkpoint_path: str,
    train_adata_path: str,
    val_adata_path: Optional[str] = None,
    ft_config: Optional[FinetuneConfig] = None,
    gene_name_col: Optional[str] = None,
    val_gene_name_col: Optional[str] = None,
    save_dir: str = './occam_finetune',
    device: str = 'cuda',
    use_output_manager: bool = True,
) -> Tuple[OccamTrainer, dict]:
    """Standalone finetuning from a pretrained checkpoint.

    Loads model weights, gene vocabulary, and training config from the
    checkpoint, creates datasets aligned with the training gene vocabulary,
    and runs finetuning.

    Args:
        checkpoint_path: Path to pretrained ``.pth`` checkpoint.
        train_adata_path: Path to training ``.h5ad`` file.
        val_adata_path: Optional validation ``.h5ad`` file.
        ft_config: Finetuning configuration.
        gene_name_col: Gene name column in training data.
        val_gene_name_col: Gene name column in validation data.
        save_dir: Directory to save finetuning results.
        use_output_manager: Route output through the centralised output manager,
            which allocates a globally-numbered ``exp_<NNN>_...`` directory and
            IGNORES ``save_dir``. The orchestrator passes ``False`` so that
            artifacts land in the deterministic run directory it can find them
            in. Two of the released checkpoints became unattributable because
            that global numbering collided between concurrent array tasks; see
            docs/reproducing_chain_a.md.
        device: Device for training.

    Returns:
        ``(trainer, dataset_info)``
    """
    if ft_config is None:
        ft_config = FinetuneConfig()

    os.makedirs(save_dir, exist_ok=True)

    # ---- Initialize distributed process group early ----
    if not dist.is_initialized():
        setup_distributed()

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    training_config = PretrainConfig.from_dict(checkpoint['config'])
    ckpt_gene_names = checkpoint.get('gene_names', None)
    ckpt_dataset_info = checkpoint.get('dataset_info', None)

    # Resolve finetuning config to determine data_mode, batch_size, etc.
    resolved = ft_config.resolve(training_config)
    if resolved.use_wandb and resolved.wandb_run_name is None:
        from datetime import datetime as _dt
        _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        resolved.wandb_run_name = f"occam_finetune_{resolved.model_size}_{resolved.data_mode}_{_ts}"

    # ---- Build transforms ----
    target_sum = resolved.target_sum
    transforms = []
    if not resolved.skip_normalize:
        transforms.append(NormalizePerCell(target_sum))
    if not resolved.skip_log1p:
        transforms.append(Log1p())
    if resolved.scale:
        print("Computing per-gene statistics for z-score scaling...")
        stats = compute_gene_stats(train_adata_path, target_sum=target_sum)
        transforms.append(Scale(
            torch.tensor(stats["mean"]),
            torch.tensor(stats["std"]),
        ))

    # ---- Create datasets ----
    _optim_chunk_size = getattr(resolved, 'optim_chunk_size', 8192)
    _optim_buffer_chunks = getattr(resolved, 'optim_buffer_chunks', 4)
    if resolved.data_mode == "optim":
        _rank = dist.get_rank() if dist.is_initialized() else 0
        _world = dist.get_world_size() if dist.is_initialized() else 1

        _optim_train_chunk_ids = None
        _optim_val_chunk_ids = None
        if val_adata_path is None:
            _n_obs, _n_chunks = compute_global_chunk_info(
                train_adata_path, _optim_chunk_size,
                gene_vocabulary=ckpt_gene_names,
                gene_name_col=gene_name_col,
                min_gene_overlap_pct=getattr(resolved, 'min_gene_overlap_pct', None),
            )
            _optim_train_chunk_ids, _optim_val_chunk_ids = optim_train_val_chunk_split(
                total_chunks=_n_chunks,
                val_fraction=resolved.val_split, seed=42,
            )
            print(
                f"Optim chunk-based train/val split: "
                f"{len(_optim_train_chunk_ids)} train chunks, "
                f"{len(_optim_val_chunk_ids)} val chunks "
                f"({resolved.val_split*100:.1f}%)"
            )

        print(f"Creating training dataset (mode=optim, optimized chunked loader)...")
        train_dataset = OccamChunkedDataset(
            adata_path=train_adata_path,
            batch_size=resolved.batch_size,
            transforms=[],
            gene_name_col=gene_name_col,
            gene_vocabulary=ckpt_gene_names,
            chunk_size=_optim_chunk_size,
            buffer_chunks=_optim_buffer_chunks,
            seed=42,
            shuffle=True,
            drop_last=True,
            rank=_rank,
            world_size=_world,
            chunk_indices=_optim_train_chunk_ids,
            track_presence=getattr(resolved, 'gene_presence_mask', False),
            min_gene_overlap_pct=getattr(resolved, 'min_gene_overlap_pct', None),
        )
    else:
        print(f"Creating training dataset (mode={resolved.data_mode})...")
        train_dataset = OccamAdataDataset(
            adata_path=train_adata_path,
            mode=resolved.data_mode,
            transforms=transforms,
            gene_name_col=gene_name_col,
            gene_vocabulary=ckpt_gene_names,
        )

    # ---- Experiment management ----
    experiment = None
    misc_save_dir = save_dir
    if use_output_manager and is_rank0_env():
        try:
            output_manager = get_output_manager()
            experiment = output_manager.create_experiment(
                experiment_type="finetuning",
                description=f"Finetuning from {os.path.basename(checkpoint_path)}",
                config=ft_config.to_dict(),
                tags=["occam", "finetuning"],
            )
            misc_save_dir = str(experiment.experiment_dir)
        except Exception as e:
            print(f"Failed to create experiment via OutputManager: {e}")

    # ---- Initialize trainer ----
    exp_dir_name = os.path.basename(misc_save_dir) if misc_save_dir else None
    trainer = OccamTrainer(resolved, device, experiment_dir_name=exp_dir_name)

    trainer._gene_names = ckpt_gene_names if ckpt_gene_names else train_dataset.gene_names
    trainer._dataset_info = ckpt_dataset_info if ckpt_dataset_info else train_dataset.get_dataset_info()
    trainer._transforms = transforms
    if resolved.data_mode == "optim":
        trainer._defer_transforms = True
        if trainer._dataset_info.get('target_sum') is None:
            for t in transforms:
                if isinstance(t, NormalizePerCell):
                    trainer._dataset_info['target_sum'] = t.target_sum
                    break
            if trainer._dataset_info.get('target_sum') is None:
                trainer._dataset_info['target_sum'] = resolved.target_sum
        if hasattr(train_dataset, 'get_folder_summary'):
            trainer._folder_summary = train_dataset.get_folder_summary()

    # ---- Setup model and load weights ----
    num_genes = train_dataset.num_genes
    trainer._setup_model(num_genes)

    cleaned = _clean_state_dict(checkpoint['model_state_dict'])

    base_model = _inner_model_for_weights(trainer.model)
    base_model.load_state_dict(cleaned, strict=False)
    # Log any skipped keys (e.g. loss fn params when switching loss type)
    model_keys = set(base_model.state_dict().keys())
    ckpt_keys = set(cleaned.keys())
    unexpected = ckpt_keys - model_keys
    missing = model_keys - ckpt_keys
    if unexpected:
        print(f"  Ignored checkpoint keys (not in model): {unexpected}")
    if missing:
        print(f"  Randomly initialized keys (not in checkpoint): {missing}")
    print(f"Model loaded: {base_model.get_parameter_count():,} parameters, {num_genes} genes")

    # ---- Create dataloaders ----
    val_loader = None
    if resolved.data_mode == "optim":
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

        if val_adata_path is not None:
            print("Creating validation dataset (in_memory mode)...")
            val_dataset = OccamAdataDataset(
                adata_path=val_adata_path,
                mode="in_memory",
                transforms=[],
                gene_name_col=val_gene_name_col,
                gene_vocabulary=trainer._gene_names,
            )
            val_loader = create_dataloader(
                val_dataset, batch_size=resolved.batch_size * 2, shuffle=False,
                is_distributed=trainer.is_distributed, num_workers=4, pin_memory=True,
            )
        else:
            print(f"Creating validation dataset from reserved chunks...")
            val_dataset = OccamChunkedDataset(
                adata_path=train_adata_path,
                batch_size=resolved.batch_size * 2,
                transforms=[],
                gene_name_col=gene_name_col,
                gene_vocabulary=ckpt_gene_names,
                chunk_size=_optim_chunk_size,
                buffer_chunks=min(_optim_buffer_chunks, len(_optim_val_chunk_ids)),
                seed=42,
                shuffle=False,
                drop_last=False,
                rank=_rank,
                world_size=_world,
                chunk_indices=_optim_val_chunk_ids,
                track_presence=getattr(resolved, 'gene_presence_mask', False),
                min_gene_overlap_pct=getattr(resolved, 'min_gene_overlap_pct', None),
            )
            val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
    elif val_adata_path is not None:
        print("Creating validation dataset...")
        val_dataset = OccamAdataDataset(
            adata_path=val_adata_path,
            mode=resolved.data_mode,
            transforms=transforms,
            gene_name_col=val_gene_name_col,
            gene_vocabulary=trainer._gene_names,
        )
        train_loader = create_dataloader(
            train_dataset, batch_size=resolved.batch_size, shuffle=True, drop_last=True,
            is_distributed=trainer.is_distributed,
            num_workers=4 if resolved.data_mode == "in_memory" else 0,
            pin_memory=resolved.data_mode == "in_memory",
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=resolved.batch_size * 2, shuffle=False,
            is_distributed=trainer.is_distributed,
            num_workers=4 if resolved.data_mode == "in_memory" else 0,
            pin_memory=resolved.data_mode == "in_memory",
        )
    else:
        print(f"No validation data. Creating {resolved.val_split*100:.1f}% split...")
        train_split, val_split_ds = create_train_val_split(train_dataset, val_split=resolved.val_split, seed=42)
        train_loader = create_dataloader(
            train_split, batch_size=resolved.batch_size, shuffle=True, drop_last=True,
            is_distributed=trainer.is_distributed,
            num_workers=4 if resolved.data_mode == "in_memory" else 0,
            pin_memory=resolved.data_mode == "in_memory",
        )
        val_loader = create_dataloader(
            val_split_ds, batch_size=resolved.batch_size * 2, shuffle=False,
            is_distributed=trainer.is_distributed,
            num_workers=4 if resolved.data_mode == "in_memory" else 0,
            pin_memory=resolved.data_mode == "in_memory",
        )

    # ---- Save finetuning config ----
    if is_rank0_env():
        import json
        ft_cfg_path = os.path.join(misc_save_dir, "finetuning_config.json")
        with open(ft_cfg_path, "w") as f:
            json.dump({"ft_config": ft_config.to_dict(), "resolved_config": resolved.to_dict()}, f, indent=2)

    # ---- Finetune ----
    ft_results = trainer.finetune(
        train_loader,
        val_loader,
        save_dir=misc_save_dir,
        ft_config=ft_config,
        training_config=training_config,
        epoch_offset=0,
        wandb_prefix='finetune_',
        checkpoint_prefix='finetuned_',
        experiment=experiment,
    )

    dataset_info = train_dataset.get_dataset_info()
    print(f"Finetuning completed! Results saved to {misc_save_dir}")
    return trainer, dataset_info


# Backward compatibility aliases
OccamTrainer = OccamTrainer
train_occam = run_pretrain
finetune_occam = run_finetune
