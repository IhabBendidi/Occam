# `reproducibility/` — frozen checkpoints for the RHJ few-shot benchmark

## What this folder is for

**Every checkpoint whose number appears in a paper, a doc or a comparison table gets frozen
here, with the exact provenance needed to re-derive that number.** The weights are ~100 MB
each, so they are **gitignored** and destined for HuggingFace; what git tracks is the
`MANIFEST.json` and the training config next to it.

The problem this solves: `runs/` is a working directory. Sweeps overwrite it, `best-*.ckpt`
is re-selected every validation, and arm directories get pruned. A number in a doc whose
weights have been overwritten is not reproducible — it is folklore. Freezing costs 100 MB.

## What each entry must contain

| file | why |
|---|---|
| `MANIFEST.json` | sha256 of the weights, the selection rule, the seed, the deliberate deviations from the method's published setup, the validation curve, the per-split test scores, and the git commit |
| `<config>.yaml` | the exact config that produced it, copied verbatim |
| `best-*.ckpt` / `*.pth` | the weights (gitignored; upload target) |

⚠️ **`deliberate_deviations` is the load-bearing field.** Every competitor here was retrained
on *our* data, *our* 6,983-gene axis and *our* splits, so none of these checkpoints reproduce
the method's published number, and comparing them to one is wrong. The manifest says so
explicitly per entry.

## How to read the scores in a manifest

`test_scores_5_splits_uncalibrated` is `model_nocal` under the split-half paired protocol,
scored by cell-eval 0.6.0 / pdex 0.1.24. Anchors, on the same panel and the same protocol:

| anchor | `pearson_delta` | what it is |
|---|---|---|
| `ctrlcell` | −0.0018 | predict a control cell — zero-signal floor |
| `ctrlmean` | 0.0016 | predict the control mean |
| `meanpert` | 0.1324 | predict the mean perturbation effect, ignoring identity |
| `transfer` | 0.2323 | the same perturbation measured in a different cell line |
| `experimental` | 0.3722 | half-A vs half-B of the SAME experiment — the replicate ceiling |

A score is only meaningful against these. Read `transfer` as the bar to beat and
`experimental` as the ceiling, never 1.0.

## Entries

| entry | method | test `pearson_delta` (5 splits) | notes |
|---|---|---|---|
| `txpert_rhj_fewshot_s42/` | TxPert, ep49, seed 42 | **0.3077 ± 0.0032** | GO + STRING only (2 KGs, not their published 4) |

## Uploading to HuggingFace

Not done yet. When it happens, the manifest's `sha256` is the integrity check — verify it
after download, because a silently truncated 100 MB file still loads.
