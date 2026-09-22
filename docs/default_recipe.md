# The Occam vcell DEFAULT RECIPE (adopted 2026-09-22)

## What this is and why it exists

**Standing objective.** Predict a perturbation's effect in a context where that perturbation
was never measured, from a single control cell, and — on combinatorial corpora — predict a
PAIR from the singles.

**What this file answers.** Which configuration a NEW vcell run should start from, so that no
one re-derives it and no one compares a new idea against a stale baseline. It was adopted by
user decision after the schedule axis closed on `rhj_fewshot` (F-sched3/F-sched4,
`docs/rhj_splithalf.md`) and the 72-cell composition matrix closed on `norman_combo`
(F-nor5…F-nor9, `docs/norman_results.md`).

**What is fixed and what is not.** The SCHEDULE and the SHAPE of the loss base are fixed and
dataset-independent. The three loss WEIGHTS are **not** — they are scale-matched per corpus
and copying them across datasets is a bug (see "New dataset" below).

## The recipe

| axis | value | why |
|---|---|---|
| loss base | **Q02** = `smooth_l1` + `delta_cos` + `delta_mag` + latent MSE on `delta_z` | best on `rhj_fewshot` (0.4043); best single cell on `norman_combo` (0.7774) |
| epochs | **150** | +0.026 pΔ on RHJ over 30 ep — more than any loss term measured; 180 ep is flat (+0.0003) |
| `eta_min` | **3.0e-04** | floors above 3e-4 cost at 150 ep (F-sched3) |
| `batch_size` | **2048** | W23's winning setting |
| `lr` / `decoder_lr` | **1.0e-03 / 1.0e-03** | W23 raised `decoder_lr` together with the batch size |
| `warmup_epochs` | **5** | unchanged from the original recipe |
| composition mode | **`decode_seq_mean`** | best mean rank (2.08/8), top-two on 5 of 6 arms, best single cell, best `mae` marginal |

Where it lives, so it cannot drift:

- `configs/rhj_fewshot.yaml` and `configs/norman_combo.yaml` — `stages.vcell.optim` and `.loss`
- `VirtualCellModel.forward_composed(mode="decode_seq_mean")` — `src/occam/vcell/model.py`
- `scripts/eval_splithalf.py --compose` default
- `tests/test_default_recipe.py` — 5 tests pin all of the above

## The evidence, in one table per axis

**Schedule (RHJ, `pearson_delta`, `nocal`, 5 paired split-halves).** Anchors: `ctrlcell`
−0.0018 · `ctrlmean` 0.0016 · `meanpert` 0.1324 · `transfer` 0.2323 · `experimental` 0.3722.

| arm | base | epochs | eta_min | pearson_delta |
|---|---|---|---|---|
| **Y06** | Q02 | **150** | **3e-4** | **0.4043** ⭐ campaign best |
| AA01 | Q02 | 180 | 3e-4 | 0.4046 (+0.0003 — flat) |
| W23 | Q02 | 90 | 1e-5 | 0.3783 |

**Schedule (Norman).** Paired 150 ep/3e-4 minus 30 ep/1e-7, same base and mode:
**+0.0840 pΔ mean, positive in 43/48 cells; 35/36 on the recon-emission modes** (F-nor6).

**Loss base (Norman, mean over the 6 recon modes × both panels, pooling both schedules).**

| base | mean pΔ |
|---|---|
| M03 (dcos only, balanced) | **0.6758** |
| Q02 (M06 + latent MSE) | 0.6415 |
| M06 (direction-dominated) | 0.5806 |

At the top cell the order reverses: `NQ02__decode_seq_mean` 0.7774 (mean of panels) vs
`NM03__delta_sum` 0.7596, and on the 150-epoch recon-mode marginal the two are tied
(Q02 0.6953 vs M03 0.6976). **Q02 is the default because it wins RHJ outright and wins
Norman's best cell; M03's marginal lead comes from arms and modes the default does not use.**
M06 is ruled out on both benchmarks.

**Composition mode (Norman, mean over the 6 arms).** Full 12-mode × 6-arm × 2-panel matrix in
`docs/norman_results.md`; the 8 distinct modes (the three `latent_seq_*` are identical to
1.07e-08) rank:

| mode | mean rank (1–8) | pΔ add | pΔ non | mae add ↓ | mae non ↓ |
|---|---|---|---|---|---|
| **`decode_seq_mean`** | **2.08** | 0.7351 | **0.6518** | **0.0145** | **0.0169** |
| `sum` | 2.25 | **0.7402** | 0.6320 | 0.0150 | 0.0181 |
| `delta_sum` | 3.08 | 0.6890 | 0.6316 | 0.0188 | 0.0212 |
| `decode_seq_ba` | 4.17 | 0.6863 | 0.5746 | 0.0155 | 0.0187 |
| `decode_seq_ab` | 5.50 | 0.6279 | 0.5408 | 0.0160 | 0.0185 |
| `delta_sum_realctrl` | 5.92 | 0.4235 | 0.3956 | 0.0288 | 0.0293 |
| `latent_seq_mean` | 6.25 | 0.5840 | 0.4985 | 0.0178 | 0.0199 |
| `resid_seq_mean` | 6.75 | 0.4043 | 0.3801 | 0.0278 | 0.0282 |

⚠️ `decode_seq_mean` is a **no-op on any single-perturbation corpus** (`rhj_fewshot`
included): with one perturbation the second round is never selected and the row keeps its
first-round prediction bit-identically — pinned by
`tests/test_composition_modes.py::test_every_RECON_mode_is_identical_on_a_SINGLE_perturbation`.
Setting it as the global default therefore changes no RHJ number.

⚠️ **`sum` remains the mode to beat, not a discarded option.** It has the higher additive
marginal and the most per-arm wins (6 of 12 arm×panel cells), and it costs one decoder pass
against `decode_seq_mean`'s four. Use it when compute is the constraint; it degrades faster
into non-additivity (ratio 0.854 vs the bar's own 0.879).

## Running it

Four commands per benchmark, no overrides — see `scripts/repro/README.md`:

```bash
sbatch scripts/repro/rhj_train.sh          # or norman_train.sh   (~4.5 h / ~3 h, 1 H100)
sbatch scripts/repro/rhj_eval.sh           # or norman_eval.sh    (5 splits, ~40 min)
python scripts/export_split_metrics.py     # all 35 metrics, per arm, per split, per panel
python scripts/repro/occam_predict.py ...  # inference on your own control cells
```

The published checkpoints and the full per-split metric tables are on HuggingFace
(`docs/hf_release.md`).

## Starting a NEW dataset

1. Copy `configs/norman_combo.yaml` (combinatorial) or `configs/rhj_fewshot.yaml` (single).
2. **Re-measure the three loss weights on the new corpus** and overwrite them.
   `scripts/norman/norman_loss_scale.py` is the model to copy: it measures each term's own
   magnitude on the corpus and sets the weight so the term contributes the intended fraction.
   Never copy the literals — RHJ uses dcos 0.5 / mag 0.01363 / lmse 0.0044, Norman uses
   0.32977 / 0.0400866 / 0.00198826 for the SAME base. `tests/test_default_recipe.py`
   asserts the two are different, so a copy-paste fails the suite.
3. Keep `epochs: 150`, `eta_min: 3e-4`, `batch_size: 2048`, `lr`/`decoder_lr` 1e-3.
4. `latent_loss_type` must be `delta_z`. At `"none"` the latent weight is silently ignored
   (the F-loss1 failure mode).
5. Leave `--compose` at its default unless the panel has no multi-gene labels, where it is a
   no-op anyway.

## What changing these defaults changed on disk

Every run launched before 2026-09-22 passed its schedule through `--set` / grid columns, so
its own `resolved_config.yaml` still reproduces it exactly. ⚠️ But a grid row that RELIED on
the config default (30 epochs, eta_min 1e-7, batch 256, decoder_lr 3e-4, all loss weights 0)
now resolves to the new recipe and gets a **different stage hash** — it will train a new run
rather than reuse the old directory. Re-running an old arm requires its explicit overrides.
