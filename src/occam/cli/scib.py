"""Batch-integration evaluation (scIB) for a released checkpoint.

    occam-scib --checkpoint checkpoints/release/occam_small_encoder_decoder.pth

Scores the encoder's embeddings on the pancreas and PBMC panels: biological
conservation (cell-type separation) against batch mixing, combined as
``0.6 * bio + 0.4 * batch``.

Dataset locations come from the path registry (``configs/paths.yaml``), so this
needs ``scib_dir`` set. Install the extra deps with ``pip install -e ".[eval]"``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="occam-scib", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="encoder or unified checkpoint")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset, e.g. 'pbmc_12k'. Default: all three.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--output", default=None, help="write results as JSON here")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")

    from occam import paths

    try:
        wanted = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
        datasets = paths.scib_datasets(wanted)
    except paths.PathRegistryError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    missing = [(n, p) for n, p, _, _ in datasets if not os.path.exists(p)]
    if missing:
        print("scIB datasets missing from disk:", file=sys.stderr)
        for name, path in missing:
            print(f"  {name}: {path}", file=sys.stderr)
        print("\nSet 'scib_dir' in configs/paths.yaml.", file=sys.stderr)
        return 2

    from occam.evaluation.scib_metrics import run_scib_full
    from occam.inference.loader import load_encoder_wrapper

    try:
        wrapper = load_encoder_wrapper(args.checkpoint, device=args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    if args.batch_size:
        wrapper.config.eval_batch_size = args.batch_size

    results = run_scib_full(wrapper)

    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
