# RPE1 / HepG2 / Jurkat few-shot benchmark (`rhj_fewshot`)

A frozen definition for few-shot perturbation transfer: train on every perturbation of two
cell lines plus most of a third, and hold out named perturbations in that third context only.

Built by `scripts/build_rhj_fewshot.py` (v1.0.0). Everything here is derived and reproducible
from the recorded inputs; **do not hand-edit** — `tests/test_rhj_fewshot_artifact.py` re-hashes
every file against `benchmark.json`.

## What is frozen

| | |
|---|---|
| gene axis | **6,983** Ensembl IDs, `sha256 ef64f2ba7ff7816e…` |
| validation | **103** HepG2 perturbations |
| test | **621** HepG2 perturbations |
| held out of HepG2 training | **724** (103 + 621), **42,731 cells** |
| HepG2 training perturbations | **1,657** |
| RPE1, Jurkat | **all 2,381 eligible perturbations train** |
| `benchmark_sha256` | `d934211e5eb1e37e…` |

## The gene axis, and why Ensembl

The 3-way intersection of `var['ensembl_gene_id']` is 6,985, **all** of which are in the
encoder's 44,728-gene vocabulary. Two are dropped — both Ensembl IDs that map to the symbol
`HSPA14` — leaving **6,983**.

Both rejected alternatives are traps, not preferences:

- `var['gene_name']` matches essentially nothing in the encoder vocabulary (2 of 6,984), and
  carries the `HSPA14` duplicate in all three files.
- `var.index` holds **gene symbols** in the HepG2 file and **positional integer strings**
  (`'0'`, `'1'`, …) in the other two. Any intersection of those is a coincidence of
  numbering, not a gene match.

`HSPA14` is dropped rather than deduplicated because its Ensembl mapping on this axis is
ambiguous, the scored files must carry **symbols** in `var.index` for the scorer's
self-target-gene exclusion to work, and `HSPA14` is itself a perturbation in this dataset.

Restricting to the axis costs each file part of its own measurement — HepG2 drops 2,641 genes,
Jurkat 1,899, RPE1 1,766 — and that is the point: afterwards all three present the encoder
with the *same* 6,983 non-zero columns and the same 37,745 zeros, so **cell line is the only
thing that differs between contexts**.

⚠️ The axis also moves the encoder's input away from its pretraining distribution, and the
encoder is frozen, so nothing adapts. `benchmark.gene_axis.sha256` is recorded in every
checkpoint for this reason: scoring a model on a different axis than it trained on is silent
and severe.

## The split

RPE1 and Jurkat contribute **every** eligible perturbation to training. HepG2 is split, and
its validation and test perturbations are withheld **from HepG2 only**.

⚠️ **`splits.json` has `train ∩ val != {}` on purpose.** `train` is the *global perturbation
name* set the streaming datasets filter on, and all 724 held-out symbols legitimately train
from Jurkat and RPE1 — they are present in both. `exclusions.json` is what makes the held-out
**cells** held out, per (file, perturbation). If you read `splits.json` alone it looks like
leakage. It is not; the two files must be read together.

Eligibility = present as a non-control perturbation in **all three** files **and** resolvable
to a perturbation-graph node:

```
2,391  non-control labels present in all three files   (2 are missing from at least one)
2,381  of those resolve to a graph node                 <- eligible
  724  eligible labels held out of HepG2
1,657  eligible labels remain as HepG2 training
```

### The four dropped perturbations

Ten labels have no node in the 18,216-gene perturbation graph, so no embedding exists and the
model cannot represent them at all. On the streaming path they are dropped *silently* (a bare
`continue` in the v1 loader, `pert_ok &= pid >= 0` in v2), which is exactly how a held-out set
shrinks without anyone noticing. Four of them were in the requested held-out lists:

| symbol | requested split | HepG2 cells |
|---|---|---|
| `CCDC144NL` | val | 80 |
| `ALG1L` | test | 70 |
| `C14orf178` | test | 125 |
| `NEDD8-MDP1` | test | 13 |

They are **dropped**: 104 → 103 val, 624 → 621 test. `REQUESTED.json` keeps the original lists
so the difference between what was asked for and what is scorable stays visible.

The other six (`GOLGA6L1`, `KRTAP4-7`, `MTRNR2L1`, `OR4F4`, `RBM14-RBM4`, `RPS10-NUDT3`) sit
in HepG2 training and are simply never trained on.

The builder **raises** if any of these counts changes, so a graph-vocabulary update cannot
redefine the benchmark without someone editing `EXPECT_*` deliberately.

## What this does NOT measure

- **Not unseen-context transfer.** Every scored perturbation is in a cell line the model
  trained on; only the *perturbation* is unseen in that context. The axis this benchmark
  varies is few-shot perturbation transfer, not context generalisation.
- **Not a clean "unseen perturbation" test either.** All 724 held-out symbols are seen in
  Jurkat and RPE1 — deliberately, that is what makes it *few-shot*. A model can transfer a
  perturbation it learned elsewhere. If you want genuinely unseen perturbations, hold them out
  of all three files.
- **Nothing about the pretraining corpus.** 7 of 9 DiverseRNA blocks are unattributed, so it
  cannot be asserted that any of these three cell lines is absent from pretraining.
- **The graph set is a deviation.** The research-tree recipe these hyperparameters come from
  used `go string pxmap-top1 txmap-top1`; this benchmark ships `go` + `string` only, for
  portability. That is an uncontrolled difference from every prior measured number.

## Files

| file | what |
|---|---|
| `benchmark.json` | provenance root: builder version, argv, git sha, source and encoder sha256s, axis info, all counts, per-artifact sha256 |
| `gene_axis.csv` | the 6,983 Ensembl IDs and their symbols, in scored order |
| `splits.json` | `PerturbationSplitter`-compatible; loaded verbatim by the trainer |
| `exclusions.json` | per-(file, perturbation) training exclusions |
| `perturbations.csv` | per-perturbation eligibility, split, graph id, cell counts per context |
| `files/*.json` | per-source manifests including `per_pert_cells` |
| `REQUESTED.json` | the originally specified lists, before eligibility filtering |
