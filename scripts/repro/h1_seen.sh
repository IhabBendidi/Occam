#!/usr/bin/env bash
# h1_seen: score the rhj_fewshot checkpoints on H1 (80 perts). See scripts/repro/ood_seen.sh.
BENCH=h1_seen exec bash "$(dirname "${BASH_SOURCE[0]}")/ood_seen.sh" "$@"
