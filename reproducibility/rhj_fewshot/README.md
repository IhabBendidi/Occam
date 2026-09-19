# `rhj_fewshot` — frozen results

## What this benchmark asks

**Predict a perturbation's effect in a cellular context where that perturbation was never
measured, from a single control cell.** HepG2 is the held-out context: its 611-perturbation
test cohort is excluded from HepG2 training rows, but the *same* perturbations still train from
jurkat and RPE1. That is the positive half of the few-shot design — the task is
few-shot-in-a-new-context, not never-seen-at-all.

- **Axis**: 6,983 genes, frozen, symbols in `var.index`.
- **Eval space**: `log1p(counts / rowsum_FULL_PANEL * 2000)`. The full-panel denominator is
  load-bearing: it is what makes the arms' 10,000 → 2,000 remap exact.
- **Protocol**: split-half paired. Half-A perturbed cells + half-A controls are the target;
  the prediction is paired with **half-B** controls. `rng = default_rng(42 + 1000*split)`,
  5 splits.
- **Scorer**: cell-eval 0.6.0 / pdex 0.1.24, `--profile full --allow-discrete`, uncalibrated
  (`model_nocal`). **No calibration and no NB emission for any competitor.**

## Anchors — read every number against these, never against 0

| anchor | `pearson_delta` | what it is |
|---|---|---|
| `ctrlcell` | −0.0018 | predict a control cell — the zero-signal floor |
| `ctrlmean` | 0.0016 | predict the control mean |
| `meanpert` | **0.1324** | predict the mean perturbation effect, **ignoring identity**. Discriminates at chance by construction, which is what makes "it learned something perturbation-specific" testable |
| `transfer` | **0.2323** | the same perturbation measured in a different cell line — **the bar** |
| `experimental` | 0.3722 | half-A vs half-B of the SAME experiment. *Not* a ceiling: a model can beat it from the two-sided noise asymmetry alone |
| Occam (ours) | 0.3415 | for comparison |

## Entries

**One checkpoint per model per benchmark.** Every entry is the single checkpoint whose
predictions produced the number in its row — nothing else is frozen.

| entry | `pearson_delta` (5 splits) | vs `transfer` 0.2323 | status |
|---|---|---|---|
| [`txpert_s42`](txpert_s42/) | **0.3463 ± 0.0035** | +0.1140 | ✅ current. **Best competitor, and above Occam's 0.3415.** |
| [`perturbdiff_s42`](perturbdiff_s42/) | **0.2449 ± 0.0038** | +0.0126 | ✅ current (genept arm) |
| [`state_s42`](state_s42/) | **0.2362 ± 0.0019** | +0.0039 | ✅ current |
| `cellflow_s42` | — | — | ⏳ trained (val `pearson_delta` 0.2449 gene-space); test export not yet run |
| [`_WITHDRAWN_txpert_s42_mean_control_export`](_WITHDRAWN_txpert_s42_mean_control_export/) | ~~0.3077 ± 0.0032~~ | — | ⛔ **WITHDRAWN** — see below |

Full metric breakdown, the diagnostic A/B arms and the baselines:
[`docs/rhj_competitors_results.md`](../../docs/rhj_competitors_results.md).

### ⛔ the original `txpert` export is withdrawn

Kept, not deleted: the record of what was believed when is itself evidence. Three faults
(`docs/rhj_competitors.md` C28): exported rows omitted the `-1` perturbation token the model
was trained with; the basal input was a single control cell while the checkpoint was trained
*and selected* on the `(cell_type, batch)` **mean** control; and the frozen manifest recorded
`kg_list: []`. Replaced by `txpert_s42`, a retrain on single-cell controls with
`omit_cntr: true`, which scores **0.3463**.

⚠️ The replacement is **not** a clean re-measurement of the same model: the fairness decision
changed the training regime, so the +0.0386 over the withdrawn number mixes the export fix with
a different basal-control recipe. The A/B that separates them is `txpert_mean` (**0.3343**, the
upstream mean-control regime carried through training *and* inference), in
`docs/rhj_competitors_results.md`.

## What each entry contains, and what it deliberately omits

| | |
|---|---|
| `checkpoints/` | **only the checkpoint the number came from.** The others a run writes ("best" under val loss, `last`, `final`) are excluded on purpose — keeping a never-evaluated checkpoint invites someone to score it later and report it as this arm's |
| `artifacts/` | the run's own condition-space objects: one-hot maps, `var_dims`, resolved `config.yaml`, the validation `metrics.csv`. Not decoration — our exports used to **re-derive** the perturbation and batch vocabularies rather than read them, and a rebuild in a different order would point every covariate at the wrong column with no error |
| `predictions/split_N/` | the per-split prediction h5ad — the one artifact recoverable from nothing else here |
| `celleval/split_N/` | the scores as cell-eval wrote them: `agg_results.csv`, `results.csv`, `metrics.json` |
| `MANIFEST.json` | sha256 of every file, the selection rule, per-split and aggregate scores, the anchors, and **`deliberate_deviations`** |
| *omitted:* target h5ad | a deterministic function of the frozen panel and the split seed, so reproducible; it would double the footprint |
| *omitted:* `pred_de.csv` | ~95 MB per split, fully derivable from the prediction h5ad |

## ⚠️ Read `deliberate_deviations` before quoting any number here

Every arm was **retrained on our data, our gene axis and our splits**, so **none of these
numbers reproduces the method's published figure** — comparing them to one is wrong. For three
arms the recipe file our own launchers cite as authoritative
(`reproducibility/{state,txpert,perturbdiff}__replogle_nadig_fewshot/...`) **does not exist in
this checkout**; those hyperparameters were confirmed against run logs instead, and the
manifests say so.

## Adding an arm

```bash
python scripts/freeze_for_reproducibility.py \
    --arm <arm> --seed 42 --benchmark rhj_fewshot \
    --run-dir runs/competitors/<arm>/<run> \
    --scored-dir runs/rhj_splithalf/compet_<arm>_s42 \
    --selected <the ONE checkpoint that was evaluated> \
    --deviation "..." --deviation "..."     # repeatable; the freeze REFUSES without at least one
```

Weights and h5ads are **gitignored** — they go to HuggingFace. What git tracks is the manifest,
the config and the cell-eval CSVs. The manifest's `sha256` is the integrity check after
download: a silently truncated multi-GB file still loads.
