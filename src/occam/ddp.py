"""Distributed-training helpers, and the stage fence.

Extracted from ``training_pipeline.py:107-267``. The one addition is
:func:`stage_fence`, which the three-stage orchestrator needs.

Why the fence exists: each stage's ``train()`` calls ``dist.barrier()`` then
``cleanup_distributed()`` *before* running its end-of-training benchmark, so the
process group is already destroyed while the code keeps going -- which is why
rank checks after that point use :func:`is_rank0_env` (environment-based) rather
than :func:`is_main_process` (requires an initialised group). Chaining three
stages in one process then re-creates a NCCL group per stage. The fence makes
that transition explicit and releases CUDA memory between stages.
"""

from __future__ import annotations

import gc
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = [
    "setup_distributed", "cleanup_distributed", "setup_multi_gpu",
    "is_distributed", "get_rank", "is_main_process", "is_rank0_env",
    "stage_fence",
]


def setup_distributed(timeout_minutes=120):
    """
    Setup distributed training environment.

    Args:
        timeout_minutes: NCCL timeout in minutes (default 120). Must exceed
            the longest rank-0-only operation (periodic eval + scib can take
            30+ min on large eval datasets like replogle 247k cells).

    Returns:
        tuple: (rank, world_size, device_id)
    """
    from datetime import timedelta

    # Already initialized — just return current rank info
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return rank, world_size, local_rank

    # Check if running in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        # Initialize the process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU communication
            init_method='env://',
            timeout=timedelta(minutes=timeout_minutes),
        )

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device_id = local_rank

        print(f"Distributed training initialized successfully on GPU {device_id}")
        return rank, world_size, device_id
    else:
        # Not running in distributed mode
        return None, None, None


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _clean_state_dict(state_dict: dict) -> dict:
    """Strip DDP 'module.' and torch.compile '_orig_mod.' prefixes from state dict keys."""
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('_orig_mod.'):
            key = key[len('_orig_mod.'):]
        cleaned[key] = v
    return cleaned


def _inner_model_for_weights(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/DataParallel and torch.compile to the module that owns checkpoint tensors."""
    if isinstance(model, (DDP, torch.nn.DataParallel)):
        model = model.module
    inner = getattr(model, '_orig_mod', None)
    if inner is not None:
        return inner
    return model


def is_distributed():
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def is_rank0_env() -> bool:
    """Detect if this process is rank 0 using environment variables (works before dist init)."""
    rank = os.environ.get('RANK')
    if rank is not None:
        return rank == '0'
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        return local_rank == '0'
    # Non-DDP or env not set → treat as main
    return True


def setup_multi_gpu(device: str = 'cuda', use_multi_gpu: bool = True, use_ddp: bool = True):
    """
    Setup multi-GPU configuration with automatic detection and fallback.
    
    Args:
        device: Base device type ('cuda' or 'cpu')
        use_multi_gpu: Whether to enable multi-GPU support
        use_ddp: Whether to use DistributedDataParallel (True) or DataParallel (False)
        
    Returns:
        tuple: (primary_device, use_parallel, num_gpus, distributed_info)
        distributed_info is (rank, world_size, device_id) for DDP, None for DataParallel
    """
    if device == 'cpu' or not torch.cuda.is_available():
        return 'cpu', False, 0, None
    
    # Check if we're in distributed mode
    rank, world_size, device_id = setup_distributed()
    
    if rank is not None:  # Distributed mode
        primary_device = f'cuda:{device_id}'
        if is_main_process():
            print(f"🚀 DistributedDataParallel setup!")
            print(f"📱 World size: {world_size}")
            print(f"📱 Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        return primary_device, True, world_size, (rank, world_size, device_id)
    
    # Non-distributed mode
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("⚠️ No CUDA GPUs detected, falling back to CPU")
        return 'cpu', False, 0, None
    elif num_gpus == 1:
        primary_device = 'cuda:0'
        print(f"🚀 Using single GPU: {torch.cuda.get_device_name(0)}")
        return primary_device, False, 1, None
    else:
        primary_device = 'cuda:0'
        if use_multi_gpu:
            if use_ddp:
                print(f"⚠️ Multiple GPUs detected but not in distributed mode!")
                print(f"💡 To use DistributedDataParallel, launch with: torchrun --nproc_per_node={num_gpus} run_pretrain.py ...")
                print(f"🔄 Falling back to DataParallel for now")
            
            print(f"🚀 Multi-GPU setup detected!")
            print(f"📱 Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"📱 Primary device: {primary_device}")
            print(f"📱 {'DataParallel' if not use_ddp else 'DataParallel (fallback)'} will be used for multi-GPU training")
            return primary_device, True, num_gpus, None
        else:
            print(f"🚀 Using single GPU: {torch.cuda.get_device_name(0)} (multi-GPU disabled)")
            return primary_device, False, 1, None


def stage_fence(stage_name: str = "") -> None:
    """Close out a pipeline stage: barrier, tear down the group, free CUDA memory.

    Called by the orchestrator *between* stages, never inside one. Safe to call
    when distributed was never initialised.
    """
    label = f"[{stage_name}] " if stage_name else ""
    if is_distributed():
        dist.barrier()
        cleanup_distributed()
        print(f"{label}process group torn down")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    print(f"{label}stage fence complete")
