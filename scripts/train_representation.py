import argparse
import sys
from pathlib import Path

import yaml

from occam.config.loader import ConfigError, OccamConfig
from occam.config.schema import flatten_stage
from occam.pipeline.orchestrator import Orchestrator, StageFailed

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    p = argparse.ArgumentParser(description="Train an Occam encoder and decoder in one command.")
    p.add_argument("--data", required=True, type=Path, help="Training .h5ad file or folder")
    p.add_argument("--output", required=True, type=Path, help="Run directory")
    p.add_argument("--validation", type=Path, help="Optional separate validation .h5ad")
    p.add_argument("--gene-col", default="index", help="Column in .var containing gene IDs; default: .var_names")
    p.add_argument("--epochs", nargs=3, type=positive_int, default=[100, 100, 200],
                   metavar=("DENOISE", "UNMASK", "DECODER"))
    p.add_argument("--batch-size", type=positive_int, default=1024)
    p.add_argument("--data-mode", choices=["backed", "optim", "in_memory"], default="backed")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--resume", choices=["auto", "never", "force"], default="auto")
    p.add_argument("--dry-run", action="store_true", help="Show the plan without training or writing files")
    p.add_argument("--smoke", action="store_true", help="Two epochs per stage, smaller layers, CPU; for testing only")
    return p


def make_config(args):
    if not 0 < args.val_fraction < 1:
        raise ConfigError("--val-fraction must be between 0 and 1")
    if args.data.is_dir() and args.data_mode != "optim":
        raise ConfigError("For a folder of dense .h5ad files, pass --data-mode optim")
    raw = yaml.safe_load((ROOT / "configs/representation.yaml").read_text())
    output = args.output.expanduser().resolve()
    raw.update(name=output.name, seed=args.seed, device=args.device)
    raw["run"].update(root=str(output.parent), resume=args.resume)
    raw["data"].update(
        train=str(args.data.expanduser().resolve()),
        val=str(args.validation.expanduser().resolve()) if args.validation else None,
        train_gene_col=args.gene_col, val_gene_col=args.gene_col,
    )
    stages = raw["stages"]
    for stage, epochs in zip(("pretrain", "finetune", "decoder"), args.epochs):
        stages[stage]["optim"].update(epochs=epochs, batch_size=args.batch_size)
    stages["pretrain"]["data"].update(data_mode=args.data_mode, val_split=args.val_fraction)
    stages["finetune"]["runtime"]["data_mode"] = args.data_mode
    stages["decoder"]["data"].update(
        data_mode=args.data_mode, val_split=args.val_fraction, train_gene_col=args.gene_col,
    )
    stages["decoder"]["postdates_release"] = {"no_amp": args.device == "cpu" or args.smoke}
    if args.smoke:
        raw["device"] = "cpu"
        for stage in stages.values():
            stage["optim"].update(epochs=2, batch_size=min(args.batch_size, 32))
        stages["pretrain"]["model"].update(hidden_size=32, mae_decoder_hidden=64)
        stages["pretrain"]["optim"]["warmup_epochs"] = 0
        stages["decoder"]["architecture"].update(decoder_hidden_dim=64, decoder_num_blocks=2)
        stages["decoder"]["augmentation"]["no_augmentation"] = True
    for stage, block in stages.items():
        flatten_stage(stage, block)
    return OccamConfig(raw, str(output / "training.yaml"), str(ROOT))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cfg = make_config(args)
        orch = Orchestrator(cfg, dry_run=args.dry_run)
        print(orch.describe_plan())
        print("\nOutput checkpoint: " + str(Path(cfg.run_dir) / "decoder/checkpoints/best_decoder.pth"))
        if args.dry_run:
            return 2 if any(action == "blocked" for _, action in orch.plan()) else 0
        for name in ("train", "val"):
            path = cfg.data.get(name)
            if path and not Path(path).exists():
                raise ConfigError(f"{name} data not found: {path}")
        if any(action == "blocked" for _, action in orch.plan()):
            return orch.run()
        import random

        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.path).write_text(yaml.safe_dump(cfg.raw, sort_keys=False))
        result = orch.run()
        if result == 0:
            print("\nReady: " + str(Path(cfg.run_dir) / "decoder/checkpoints/best_decoder.pth"))
        return result
    except (ConfigError, StageFailed, ValueError) as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
