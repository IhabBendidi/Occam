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

⚠️ Arm B's in-run **BMDB numbers are meaningless** and should be ignored: it
runs the pre-fix code, in which the Ensembl-to-symbol translation was missing,
so it reports `valid_pairs: 1`. Fixed in `occam.gene_ids` (1 pair -> 142,856 on
the released decoder; see [evaluation.md](evaluation.md)). Checkpoint selection
and cell-eval are unaffected — selection reads the reconstruction combined
score, not BMDB.

⚠️ These are **reconstruction** numbers, not perturbation numbers, and the two
have already been shown to rank decoders differently on this project (3072x10
wins decoder-only cell-eval and loses virtual-cell transfer 6/6). They say the
optimisation is healthy. They say nothing about whether arm B beats arm A.

## The run that produced arm B's checkpoint died at epoch ~195 of 200

Job `18006463` FAILED (exit 120) at 10:48:54 elapsed, killed by a GPFS outage
that made `/rxrx/scratch` return `ESTALE` cluster-side. It was at epoch 192/200
ten minutes earlier, so the filesystem killed it, not the training.

**It cost almost nothing.** `best_decoder.pth` is **epoch 124**, written hours
before the outage, verified intact afterwards: every decoder tensor finite,
`best_combined_score` 65.36, `best_val_loss` 0.053064, NB `theta` mean 20.52
(range 0.78-237.1). The evals after it were *worse* — 64.69 at epoch 149 and
65.21 at 174 — so selection had already peaked and the lost epoch-199 eval was
unlikely to beat it. What was lost is the epoch-199 eval and the in-run
cell-eval, which would have failed anyway on pre-fix code.

Arm B is therefore reported as **epoch 124 of a run that reached ~195**, not as
a completed 200-epoch run.

## Emission arms on the epoch-124 checkpoint (in flight)

Three parallel jobs on one checkpoint, one emission each: `18014497` raw,
`18014498` int_libnorm, `18014499` int_nolibnorm. Verified from the written
prediction files rather than from the logs, on a 200-cell slice of perturbed
rows:

| arm | zero fraction | mean count depth | values integral in count space |
|---|---|---|---|
| **target (real data)** | **0.6060** | 2000 | no — counts / real depth, rescaled |
| `raw` | 0.5943 | 2006.2 | 0.5945 (i.e. only the zeros) |
| `int_libnorm` | **0.9069** | **2000.0** | 0.9069 (rescaling makes non-zeros fractional again) |
| `int_nolibnorm` | **0.9069** | **1511.1** | **1.0000** |

Two things are already visible before any metric returns:

1. **`raw` is nearly calibrated to the target's sparsity already** — 0.5943
   against 0.6060 — because the quantile calibration matches per-gene
   distributions, and that includes the zero mass. There is little sparsity
   error left for integerisation to fix.
2. **Rounding over-sparsifies by ~30 percentage points**, 0.6060 -> 0.9069. At
   `target_sum` 2,000 over 8,873 present genes the mean count per gene is ~0.23,
   so most genes round to zero. `int_nolibnorm` additionally loses **24.6% of
   the count mass** (1511 vs 2000), a uniform downward shift on every surviving
   gene.

So the prediction registered before the numbers land: **both integer arms
should lose `pearson_delta` relative to `raw`**, because they move sparsity
away from the target rather than towards it. The interesting counter-case is
`de_nsig_counts_pred` — far fewer non-zero genes means far fewer testable ones,
so the over-call factor could fall even as the delta geometry degrades. If that
is what happens, it is a warning about reading DE-inflation improvements as
quality improvements, not a win.

## Results

**Arm B at epoch 49 of 200 already beats arm A on every promoted metric**, and
the pre-registered mechanistic prediction held. Arm A: job `18006756`,
COMPLETED 2:57:37. Arm B early read: job `18009517`, COMPLETED 2:49:17, the
epoch-49 `best_decoder.pth` snapshot.

| metric | A: released (rect_sigmoid + smooth_l1) | B: retrain @ ep49 (softplus + nb) | B − A |
|---|---|---|---|
| `pearson_delta` | 0.1985 | **0.5636** | **+0.3651** |
| `discrimination_score_l2` | 0.9652 | **0.9849** | +0.0197 |
| `discrimination_score_cosine` | 0.9365 | **0.9986** | +0.0621 |
| `de_direction_match` | 0.8212 | **0.8664** | +0.0452 |
| `overlap_at_N` *(do not rank on this)* | 0.0155 | 0.0501 | +0.0346 |
| `mse` *(error: lower better)* | 0.0169 | **0.0015** | −0.0154 |
| `mae` *(error: lower better)* | 0.0678 | **0.0249** | −0.0429 |

Both arms: `injctrl`, quantile calibration, present genes only (8,873/44,728),
jurkat, cell-eval 0.6.0 `--profile full`, 16 threads, `--batch-size 2000`,
identical target file (sha256 `cea6dd49b29b0566`).

### The pre-registered prediction, and it held

Before arm B was scored, this document predicted that if the stated mechanism
transfers, `de_nsig_counts_pred` should **roughly halve**. It did:

| | A: released | B @ ep49 | base/768 anchor |
|---|---|---|---|
| ground-truth significant genes | 262.2 | 262.2 | 262.2 |
| predicted significant genes | 3,871.0 | **1,972.7** | 3,361 → 1,572 |
| over-call factor | 14.8x | **7.5x** | 12.8x → 6.0x |
| reduction | — | **−49.0%** | −53.2% |

−49.0% here against −53.2% on the base/768 encoder. **The gain arrives through
the mechanism it was attributed to**, not merely alongside it: removing the
exact-zero floor removes the near-zero within-group variance that a Wilcoxon
test reads as strong evidence. A `pearson_delta` gain with a flat significant-
gene count would have been unexplained; this is not that.

### The trade-off, stated because it is real

| | A: released | B @ ep49 |
|---|---|---|
| `de_sig_genes_recall` | **0.6341** | 0.5166 |
| `precision_at_N` | 0.0388 | **0.0596** |

Calling 1,899 fewer genes significant means recovering fewer of the true ones:
recall drops 0.117 while precision rises. That is the expected shape of a
reduction in over-calling and it is the right trade here — the decoder still
over-calls by 7.5x — but a use case that needs recall over precision should
know the released decoder scores higher on it, for the uninteresting reason
that it calls nearly everything significant.

### Against the anchors

| arm | encoder | panel | calibrated `pearson_delta` |
|---|---|---|---|
| rect_sigmoid + smooth_l1 | base/768 | nadig | 0.2285 |
| rect_sigmoid + smooth_l1 | **small/512** | **jurkat** | **0.1985** |
| softplus + nb | base/768 | nadig | 0.6009 |
| **softplus + nb @ ep49** | **small/512** | **jurkat** | **0.5636** |

Both rows of the port land just below their base/768 counterparts, by 0.030 and
0.037. The head/loss effect reproduces at nearly its full measured size on a
different encoder and a different panel.

### Verdict against the pre-registered criterion

The criterion was: *B clearly above A's measured value -> the change survives
the move to small/512 and the retrained decoder supersedes the released one.*
**0.5636 against 0.1985 meets it, at epoch 49 of 200.** The final checkpoint can
only confirm or refine this; a reversal would itself be a finding and is not
expected, since reconstruction has been flat since epoch 50.

⚠️ **n = 1 per arm, no seed replicates**, one panel, and the geometry differs
(2048x5 released, 2048x4 retrained). The geometry confound is small next to a
+0.365 effect but it is not zero, and nothing here speaks to unseen-context
transfer: the decoder never sees perturbation labels, so this measures whether
a reconstruction PRESERVES perturbation deltas, not whether a model predicts
them.

### Cost, measured — and a corrected claim

The DE-reuse saving was overstated when it was introduced. Measured artifact
timings:

| step | A (no cache) | B (cache hit) |
|---|---|---|
| forward pass + two ~9.3 GB h5ad | ~35 min | ~35 min |
| **real-side DE** | **56 min** | **0 — reused** |
| prediction-side DE | 58 min | 76 min |
| metric computation | 56 min | 81 min |
| **total** | **2:57:37** | **2:49:17** |

The cache removed a measured 56-minute step, but net wall clock fell only 8
minutes: the softplus arm's own prediction-side DE and metric computation ran
18 and 25 minutes longer, because its predictions have no exact-zero floor and
are therefore denser to test. The earlier "~1.5 h of a ~2.5 h run" figure was
wrong; **the cache is worth ~56 min of compute, not ~1.5 h of wall clock**, and
on this panel a denser-prediction arm gives part of it back.
