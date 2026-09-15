"""``python -m occam.train`` -- the three-stage pipeline, driven by one config.

    python -m occam.train --config configs/occam_small.yaml
    python -m occam.train --config configs/occam_small.yaml --dry-run
    python -m occam.train --config configs/occam_small.yaml --stages decoder
    python -m occam.train --config configs/occam_small.yaml --resume force
    python -m occam.train --config configs/occam_small.yaml \
        --stages decoder \
        --set stages.decoder.init_from=checkpoints/reference/occam_small_stage2_finetune.pth

The config is the record of the run. ``--set`` may only change keys the config
already declares, so a run can always be reproduced from a config file rather
than from a shell-history reconstruction.
"""

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="occam-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", "-c", required=True,
                   help="path to a YAML config (see configs/occam_small.yaml)")
    p.add_argument("--stages", default=None,
                   help="comma-separated subset to run, e.g. 'finetune,decoder'. "
                        "Default: run.stages from the config.")
    p.add_argument("--resume", default=None, choices=["auto", "never", "force"],
                   help="override run.resume")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override an existing config key; repeatable")
    p.add_argument("--run-suffix", default=None,
                   help="suffix the run directory, for sweeps")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and the stage plan, run nothing")
    p.add_argument("--print-config", action="store_true",
                   help="print the fully resolved config and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from occam.config.loader import ConfigError, has_unresolved_refs, load_config
    from occam.pipeline.ledger import Ledger
    from occam.pipeline.orchestrator import Orchestrator, StageFailed

    try:
        # Load once without a ledger to learn the run directory, then reload
        # with the ledger so any already-recorded artifacts resolve eagerly.
        bootstrap = load_config(args.config, overrides=args.overrides)
        if args.run_suffix:
            bootstrap.run["run_suffix"] = args.run_suffix
        ledger = Ledger.load(bootstrap.run_dir)

        cfg = load_config(args.config, overrides=args.overrides, ledger=ledger.data)
        if args.stages:
            wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
            cfg.run["stages"] = wanted
        if args.resume:
            cfg.run["resume"] = args.resume
        if args.run_suffix:
            cfg.run["run_suffix"] = args.run_suffix
    except ConfigError as exc:
        print(f"\nconfig error: {exc}\n", file=sys.stderr)
        return 2

    if args.print_config:
        import yaml
        yaml.safe_dump(cfg.raw, sys.stdout, sort_keys=False, default_flow_style=False)
        return 0

    orch = Orchestrator(cfg, dry_run=args.dry_run)

    print(orch.describe_plan())
    deferred = has_unresolved_refs(cfg.stages)
    if deferred:
        print("\ndeferred references (resolved from the ledger as each stage starts):")
        for d in deferred:
            print(f"  {d}")

    if args.dry_run:
        print("\n[occam] --dry-run: nothing was run")
        blocked = [s for s, a in orch.plan() if a == "blocked"]
        return 2 if blocked else 0

    try:
        return orch.run()
    except StageFailed as exc:
        print(f"\n[occam] pipeline stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[occam] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
