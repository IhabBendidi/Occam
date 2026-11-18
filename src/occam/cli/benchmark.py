"""Standalone benchmark evaluation on saved Occam checkpoints.

Usage:
    # Single checkpoint
    python eval_benchmark.py --checkpoint path/to/best_model.pth

    # Multiple checkpoints (glob)
    python eval_benchmark.py --checkpoint "outputs/experiments/exp_79*/checkpoints/best_model.pth"

    # With wandb logging
    python eval_benchmark.py --checkpoint path/to/best_model.pth --wandb_project Occam
"""

import argparse
import glob
import os
import sys

import torch

from occam.model import AutoEncoder, resolve_decoder_hidden
from occam.transforms import NormalizePerCell, Log1p
from occam.evaluation.benchmark_runner import run_end_of_training_benchmark


class _CheckpointWrapper:
    """Minimal object satisfying the benchmark_runner trainer interface."""

    def __init__(self, model, device, gene_names, target_sum, config=None):
        self.model = model
        self.device = device
        self._gene_names = gene_names
        skip_normalize = getattr(config, 'skip_normalize', False) if config else False
        if skip_normalize:
            self._transforms = [Log1p()]
        else:
            self._transforms = [NormalizePerCell(target_sum), Log1p()]
        self.config = config


def load_and_eval(checkpoint_path, device, wandb_module=None,
                  run_scib=True, shuffle_embeddings=False):
    """Load a checkpoint and run the full benchmark suite on it.

    Handles both plain MAE and unified encoder+decoder checkpoints; see
    :mod:`occam.inference.loader` for why that distinction needs care.
    """
    from occam.inference.loader import load_encoder_wrapper

    try:
        wrapper = load_encoder_wrapper(checkpoint_path, device=device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR: {exc}")
        return None

    results = run_end_of_training_benchmark(
        wrapper, wandb_module=wandb_module, shuffle_embeddings=shuffle_embeddings,
    )

    if run_scib:
        try:
            from occam.evaluation.scib_metrics import run_scib_full
            scib_results = run_scib_full(wrapper, wandb_module=wandb_module,
                                         shuffle_embeddings=shuffle_embeddings)
            if results is not None:
                results["scib"] = scib_results
        except Exception as e:  # noqa: BLE001
            print(f"[scib] Full eval failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Occam benchmark on saved checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pth file (supports glob patterns)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_scib_eval", action="store_true",
                        help="Disable scib cell-type benchmarks")
    parser.add_argument("--shuffle_embeddings", action="store_true",
                        help="Randomly permute embeddings to break label correspondence (random baseline)")
    args = parser.parse_args()

    # Resolve glob
    paths = sorted(glob.glob(args.checkpoint))
    if not paths:
        print(f"No checkpoints matched: {args.checkpoint}")
        sys.exit(1)
    print(f"Found {len(paths)} checkpoint(s) to evaluate")

    wandb_mod = None
    if args.wandb_project:
        import wandb
        wandb_mod = wandb

    shuffle_prefix = "shuffle_" if args.shuffle_embeddings else ""

    for path in paths:
        # Init a separate wandb run per checkpoint if requested
        run_name = args.wandb_run_name
        if wandb_mod is not None:
            if run_name is None:
                # Derive name from experiment directory
                parts = path.split(os.sep)
                for p in parts:
                    if p.startswith("exp_"):
                        run_name = f"{shuffle_prefix}benchmark_{p}"
                        break
                else:
                    run_name = f"{shuffle_prefix}benchmark_{os.path.basename(os.path.dirname(os.path.dirname(path)))}"
            wandb_tags = ["shuffle_baseline"] if args.shuffle_embeddings else []
            wandb_mod.init(project=args.wandb_project, name=run_name, tags=wandb_tags, reinit=True)

        results = load_and_eval(path, args.device, wandb_module=wandb_mod,
                               run_scib=not args.no_scib_eval,
                               shuffle_embeddings=args.shuffle_embeddings)

        if wandb_mod is not None and wandb_mod.run is not None:
            wandb_mod.finish()

        if results is not None:
            print(f"\n{'='*60}")
            print(f"Checkpoint: {path}")
            print(f"Overall score: {results['overall_score']:.4f}")
            for ds, score in results["per_dataset_scores"].items():
                print(f"  {ds}: {score:.4f}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
