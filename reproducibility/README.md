# `reproducibility/` — frozen results, one folder per benchmark

## What this is for

**Every number that appears in a paper, a doc or a comparison table gets frozen here, with the
provenance needed to re-derive and to interpret it.**

The problem it solves: `runs/` is a working directory. Sweeps overwrite it, `best-*.ckpt` is
re-selected at every validation, and arm directories get pruned. A number whose weights and
predictions have been overwritten is not reproducible — it is folklore.

## Layout

```
reproducibility/
├── README.md                 ← you are here
└── <benchmark>/              ← one per benchmark; rhj_fewshot is the first
    ├── README.md             ← the question it asks, the eval space, THE ANCHORS
    └── <arm>_s<seed>/
        ├── MANIFEST.json     ← sha256s, selection rule, scores, deliberate_deviations
        ├── checkpoints/      ← ONLY the checkpoint that was evaluated
        ├── artifacts/        ← config + the run's own condition-space objects
        ├── predictions/      ← per-split prediction h5ad
        └── celleval/         ← per-split scores as cell-eval wrote them
```

| benchmark | what it asks | entries |
|---|---|---|
| [`rhj_fewshot`](rhj_fewshot/) | predict a perturbation's effect in a context where it was never measured, from a single control cell | `state_s42` ✅ · `txpert_s42` ⛔ withdrawn |

## Three rules

1. **One evaluated checkpoint per entry.** Not every checkpoint a run wrote. Keeping weights
   that were never scored invites someone to score them later and report the result as that
   arm's.
2. **`deliberate_deviations` is mandatory and load-bearing.** Every competitor here was
   retrained on our data, our gene axis and our splits, so **none of these numbers reproduces
   the method's published figure.** The manifest has to say so, per entry, or the number will
   eventually be compared to one that it cannot be compared to.
3. **A withdrawn result stays, marked.** It is struck through in the tables and flagged in its
   README, never deleted — the history of what was believed when is itself evidence, and twice
   in this campaign a "known" result turned out to rest on a measurement nobody had re-read.

## Adding an entry

```bash
python scripts/freeze_for_reproducibility.py \
    --arm <arm> --seed <n> --benchmark <benchmark> \
    --run-dir <the training run dir> \
    --selected <the ONE checkpoint that was evaluated>
```
Then fill in `deliberate_deviations`. The tool refuses to run without `--selected`.

## HuggingFace

Weights and h5ads are **gitignored**; they go to HuggingFace. Git tracks the manifests, the
configs and the cell-eval CSVs — everything needed to read a result without downloading
gigabytes. After any download, verify against the manifest's `sha256`: a silently truncated
multi-GB file still loads.
