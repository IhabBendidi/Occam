# Stage-3 retrain: does softplus + NB beat the released decoder on jurkat?

## What we are trying to achieve overall

A decoder that turns an Occam embedding back into gene expression faithfully
enough to support perturbation prediction. The encoder is fixed and is not
under test here — this is stage 3 only, on the frozen small/512 encoder.

## The question this run answers

**Does replacing the released decoder's `rectified_sigmoid` head and smooth-L1
loss with `softplus` + a negative-binomial likelihood improve calibrated
cell-eval on the small/512 encoder, as it did on the base/768 one?**

Both outcomes are informative. A win means the released decoder should be
superseded. A null means the gain measured on the base/768 encoder does not
transfer, and the released decoder stays — which is itself worth knowing,
because the whole prepared-retrain decision rests on that transfer.

## What is varied, and what is held fixed

| | released decoder (arm A) | retrain (arm B) |
|---|---|---|
| output head | `rectified_sigmoid` (ceiling log1p(2000)) | **`softplus`** |
| loss | `smooth_l1`, β 2.0 | **`nb`**, learned per-gene `log_theta` |
| learning rate | 3e-6 | **5e-5** |
| geometry | 2048 × **5** | 2048 × **4** |
| encoder | frozen small/512, stage-2 finetuned | same |
| target_sum, augmentation, batch size, epochs, seed | | same |

The head and the loss move together on purpose: `softplus` is unbounded above,
and the NB likelihood is what keeps it in range. Paired with smooth-L1 instead,
it showed ~38% validation-loss spikes at the two lower learning rates.

Geometry is **not** held fixed — 2048×5 is what the release happens to be, and
2048×4 is what every H1 arm used. It is a confound, stated rather than hidden.
It is small next to the head change (137.4 M vs 129.0 M parameters), and the
alternative — rebuilding at 5 blocks — would answer a question nobody asked.

⚠️ Do **not** "upgrade" this to 3072×10. That geometry wins decoder-only
cell-eval and **loses** virtual-cell transfer by 1.6–2.9 sd, negative in 6/6
seed-paired arms, with one arm diverging 3/3 at `loss=nan` from epoch 1.

## How both arms are measured

One eval file, **`data/eval/pert/nadig_2024_jurkat.h5ad`**: 262,956 cells,
2,394 perturbations, 12,013 `non-targeting` control cells, 8,882 of the
encoder's 44,728 genes present.

ARC cell-eval v1, `--profile full`, `injctrl`, per-gene quantile calibration fit
on control cells only, present genes only, 32 threads. Arm A goes through
`occam-celleval`; arm B through the trainer's own end-of-training cell-eval.
Both call `DecoderTrainer.run_cell_eval_on_file` with the same settings, so the
difference between the arms is the decoder and nothing else.

⚠️ **Calibration is the dominant term, not the loss.** It moves `pearson_delta`
0.23 → 0.60 on nadig and 0.21 → 0.81 on TREK; the head/loss choice is worth
about 0.02. It is fit on the eval file's *control* cells, so it cannot
manufacture perturbation signal — it removes the decoder's per-gene shift — but
it is fit on real data and every number below states its calibration setting.

## Anchors

From the base/768 encoder on a different panel, so these set the expected
**direction, not the expected number** (`injctrl`, `pearson_delta`, higher is
better):

| arm | nadig pΔ | nadig disc_l2 | nadig predicted sig. genes | TREK pΔ |
|---|---|---|---|---|
| rectified_sigmoid + smooth_l1, calibrated | 0.2285 | 0.9798 | 3,361 | — |
| softplus + smooth_l1, calibrated | 0.5807 | 0.9885 | 1,588 | 0.8045 |
| **softplus + nb 5e-5, calibrated** | **0.6009** | **0.9907** | 1,572 | **0.8056** |
| softplus + nb 5e-5, **no** calibration | 0.2336 | 0.9202 | 7,330 | 0.2060 |

Ground truth is 262.2 significant genes per perturbation on nadig, so
3,361 → 1,572 predicted significant genes is also a large cut in DE inflation.

**The comparison that decides anything is arm B against arm A measured here**,
on jurkat, with the same encoder. The table above cannot settle it: different
encoder, different panel.

## What each outcome means

| result | reading |
|---|---|
| B's calibrated `pearson_delta` clearly above A's | the head + loss change survives the move to small/512; the retrained decoder supersedes the released one |
| B at or below A | the gain does not survive the encoder change; the released decoder stays, and the prepared-retrain recommendation is withdrawn |
| B up on `pearson_delta` but down on `discrimination_score_l2` | the decoder got better at the average perturbation and worse at telling perturbations apart — report both, promote neither |

Read `pearson_delta` and `discrimination_score_l2`. **Do not rank on
`overlap_at_N`**: a zero-signal predictor beat the real model on it
(TREK 0.3360 vs 0.2030) at chance-level discrimination.

## Checkpoint selection, and why it is not cell-eval

`best_decoder.pth` is selected on the mean reconstruction combined score
(Pearson + Spearman + 0.1·mse_inv) with **`eval_present_genes_only: true`**.
That flag is load-bearing: the combined score otherwise includes Spearman over
zero-filled padding genes, which collapses as a monotone function of the
zero-fill fraction — TREK at 1.4% padding loses 0.69 Spearman, nadig at 80.2%
loses 90.46, while Pearson and MAE barely move. The released checkpoint's own
epoch-185 selection was made without it and is untrustworthy for that reason.

Selection is **not** on cell-eval: one cell-eval run takes hours, so a
per-epoch selection on it was never implementable.

## Runs

| arm | what | job | status |
|---|---|---|---|
| A | released decoder, jurkat, calibrated | `` | launched 2026-09-15 |
| B | softplus + nb retrain, 200 epochs | `` | launched 2026-09-15 |

Commands, for the record:

```bash
sbatch scripts/slurm/celleval_decoder.sbatch     # arm A, defaults to the release
sbatch scripts/slurm/retrain_decoder.sbatch      # arm B
```

Arm A writes `runs/celleval/occam_small_encoder_decoder__nadig_2024_jurkat__quantile/metrics.json`.
Arm B writes `runs/occam_small_decoder_softplus_nb/decoder/`.

⚠️ **Memory: cell-eval on this file needs ~700 GB, not the 200 GB the launchers
originally asked for.** The first arm A (job 18006458) was killed
OUT_OF_MEMORY at a 188 GB peak. `run_cell_eval_on_file` holds two dense
`(262,956 x 44,728)` float32 arrays — predictions and targets, ~47 GB each —
plus the 43.8 GB in-memory dataset, and `np.concatenate` transiently doubles
one of them. Restriction to the 8,873 present genes happens *after* the forward
pass, so it does not bound the peak. Arm B was cancelled 6 minutes in and
resubmitted for the same reason: its end-of-training cell-eval would have hit
the identical wall after ~12 h of training. Both launchers now request 700 GB
(the nodes have 2 TB).

The obvious fix — slicing to present genes inside the forward loop, which is
numerically identical and would cut the peak ~5x — was deliberately NOT applied
while this comparison is in flight, so both arms run byte-identical code.

## Run log — two infrastructure failures, both fixed

| job | outcome | cause | fix |
|---|---|---|---|
| A `18006458` | OUT_OF_MEMORY at 4:46, 188 GB peak | two dense `262,956 x 44,728` float32 arrays (~47 GB each) plus the 43.8 GB in-memory dataset; `np.concatenate` doubles one transiently. Present-gene restriction happens *after* the forward pass, so it does not bound the peak | request 700 GB (nodes have 2 TB) |
| A `18006462` | COMPLETED in 2:20:05, but **cell-eval exited 1** | `FileNotFoundError: /psm_...` inside `pdex`, cell-eval's DE backend. 2,394 perturbations x 8,873 genes at cell-eval's default `--batch-size 100` is **~212,000 pool tasks**, each re-attaching to the single POSIX shared-memory segment holding the expression matrix. CPython's shared-memory resource tracker unlinks segments early — fixed in 3.13, and cell-eval runs under 3.10 | pass `--batch-size 2000` (a ~20x cut in task count) and drop to 16 threads, the value every successful cell-eval run on this cluster has used |

Everything upstream of cell-eval worked on `18006462`, which is what makes the
diagnosis safe: 8,873/44,728 genes present, quantile calibration fit on 12,013
control cells and applied to 8,205 of 8,873 genes (668 degenerate, skipped),
predicted zero-fraction 0.4661 against the target's 0.6060.

⚠️ Arm B (`18006463`) is still training and its process holds the pre-fix code
in memory, so its own end-of-training cell-eval will hit the same
shared-memory failure. That is non-fatal — it prints and returns no metrics —
and `best_decoder.pth` is unaffected, so arm B's number will come from
`occam-celleval` on its checkpoint. Arm B was **not** restarted again: the fix
is unproven until arm A completes, and arm B has ~15 h of training left, which
is more than enough time to decide.

## Arm B training, in progress (epoch 51/200 at 3 h)

Recorded because it answers a question the final cell-eval cannot: does
`softplus` x `nb` train stably on this corpus at lr 5e-5? It does.

| | |
|---|---|
| train loss, epochs 1 -> 51 | 0.070599 -> 0.055293, **monotone decreasing at every logged epoch** |
| val loss | min 0.053982, max 0.065348, last 0.053982 (the current minimum) |
| worst val excursion above the running best | **10.1%** |
| `nan` anywhere in the log | **no** |
| throughput | ~3.5 min/epoch -> ~11.7 h for 200 epochs |

The 10.1% figure is the one that matters: `softplus` x `smooth_l1` showed ~38%
validation-loss spikes at the two lower learning rates, which is the stated
reason the NB likelihood was paired with the unbounded head rather than
shipping softplus alone. On this corpus and this encoder that pairing behaves
as claimed.

Reconstruction eval on jurkat (present genes only), the checkpoint-selection
metric:

| epoch | Pearson | Spearman | MAE | MSE | combined |
|---|---|---|---|---|---|
| 25 | 85.44 | 40.14 | 0.0950 | 0.0218 | 64.46 |
| 50 | 85.83 | 40.80 | 0.0958 | 0.0211 | **64.96** |

`best_decoder.pth` was rewritten at both evals, so selection is still moving.

⚠️ These are **reconstruction** numbers, not perturbation numbers, and the two
have already been shown to rank decoders differently on this project (3072x10
wins decoder-only cell-eval and loses virtual-cell transfer 6/6). They say the
optimisation is healthy. They say nothing about whether arm B beats arm A.

## Results

Not yet measured. Arm A lands in ~5 h, arm B in ~18 h (≈12 h training + ≈5 h
cell-eval).

| metric | A: released (rect_sigmoid + smooth_l1) | B: retrain (softplus + nb) | B − A |
|---|---|---|---|
| `pearson_delta` | — | — | — |
| `discrimination_score_l2` | — | — | — |
| `discrimination_score_l1` | — | — | — |
| `overlap_at_N` *(do not rank on this)* | — | — | — |
| `de_direction_match` | — | — | — |
| `mse` *(error: lower is better)* | — | — | — |
| `mae` *(error: lower is better)* | — | — | — |

Both arms: `injctrl`, quantile calibration, present genes only, jurkat,
cell-eval 0.6.0 `--profile full`.

⚠️ **n = 1 per arm, no seed replicates.** Arm B is a single training run at seed
42. A difference smaller than the run-to-run spread — which has not been
measured on this panel — is not a result.
