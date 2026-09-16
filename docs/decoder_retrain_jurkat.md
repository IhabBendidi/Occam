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

**Arm B beats the released decoder by +0.3747 `pearson_delta`. Integer emission
loses on every per-perturbation metric that matters.** All four arms:
`injctrl`, quantile calibration, present genes only (8,873/44,728), jurkat,
cell-eval 0.6.0 `--profile full`, 16 threads, `--batch-size 2000`, one shared
ground-truth DE (target sha256 `cea6dd49b29b0566`, served from cache in all
three emission arms).

| metric | A released | **B raw** | B int_libnorm | B int_nolibnorm |
|---|---|---|---|---|
| `pearson_delta` | 0.1985 | **0.5732** | 0.5299 | 0.5396 |
| `discrimination_score_l2` | 0.9652 | 0.9867 | **0.9891** | 0.9819 |
| `discrimination_score_cosine` | 0.9365 | **0.9991** | 0.9987 | 0.9989 |
| `de_direction_match` | 0.8212 | **0.8705** | 0.7590 | 0.7605 |
| `de_spearman_sig` | 0.9379 | **0.9589** | 0.8568 | 0.8598 |
| `overlap_at_N` *(do not rank on this)* | 0.0155 | 0.0497 | 0.0382 | 0.0389 |
| `precision_at_N` | 0.0388 | 0.0632 | 0.0864 | **0.0887** |
| `de_sig_genes_recall` | **0.6341** | 0.5130 | 0.3646 | 0.3744 |
| `de_nsig_counts_pred` *(truth 262.2)* | 3871.0 | 1822.8 | 1034.0 | **1029.6** |
| `mse` *(error: lower better)* | 0.0169 | **0.0014** | 0.0051 | 0.0047 |
| `mae` *(error: lower better)* | 0.0678 | **0.0239** | 0.0557 | 0.0554 |
| `clustering_agreement` | 0.6311 | 0.6385 | **0.6497** | 0.6489 |
| `pearson_edistance` | 0.9282 | 0.9043 | **0.9558** | 0.9488 |

### Headline: the retrain wins, and through its stated mechanism

| | A released | B raw (ep124) | change |
|---|---|---|---|
| `pearson_delta` | 0.1985 | **0.5732** | **+0.3747** |
| `de_nsig_counts_pred` | 3,871.0 | 1,822.8 | **−52.9%** |
| over-call vs truth (262.2) | 14.8x | 6.9x | |

−52.9% against the base/768 anchor's −53.2%. The mechanism transferred at
essentially its full measured size across a change of encoder *and* panel.

Epoch 124 vs the epoch-49 snapshot: `pearson_delta` 0.5636 -> 0.5732 (+0.0096),
consistent with the small reconstruction gain (combined 64.96 -> 65.36). The
conclusion never depended on the extra 75 epochs.

### Emission: the registered prediction held, including the trap

Predicted before the runs returned: both integer arms lose `pearson_delta`
because rounding moves sparsity away from the target (0.6060) rather than
towards it, and `de_nsig_counts_pred` could fall anyway simply because fewer
genes remain testable — which would be a warning, not a win. Both happened.

Decomposing the 2x2 (the fourth cell was not run, so these are main effects
under additivity):

| effect | contrast | `pearson_delta` |
|---|---|---|
| integerisation, given no rescaling | `raw` -> `int_nolibnorm` | **−0.0336** |
| library-size rescaling, given integerisation | `int_nolibnorm` -> `int_libnorm` | **−0.0097** |

**Integerisation costs ~0.034. Putting the library size back adds nothing** —
it is slightly negative, i.e. the 24.6% of count mass that rounding destroys is
not worth restoring by rescaling. So neither factor is a lever here.

The trap, stated plainly: `de_nsig_counts_pred` almost halves again under
integer emission, 1,823 -> 1,034, and a reader who tracked only DE inflation
would call that the best arm. It is not. `de_sig_genes_recall` collapses
0.5130 -> 0.3646 and `de_direction_match` falls 0.8705 -> 0.7590. The
significant-gene count fell because **9.1% of gene-cell entries survive
rounding instead of 40.6%** — there is simply less left to test. Precision
rises (0.0632 -> 0.0887) for the same reason. **A DE-inflation improvement is
only evidence of quality when recall and direction hold; here they do not.**

The two panel-level metrics do prefer the integer arms
(`clustering_agreement` 0.6385 -> 0.6497, `pearson_edistance` 0.9043 ->
0.9558), and `discrimination_score_l2` is flat across all three. Those measure
between-perturbation geometry, which sparsification can sharpen even as
per-perturbation fidelity degrades. They are not the headline and they do not
overturn it.

### Verdict against the pre-registered criterion

The criterion was: *B clearly above A's measured value -> the head/loss change
survives the move to small/512, and the retrained decoder supersedes the
released one.* **0.5732 against 0.1985 meets it**, on the metric the criterion
named, with the mechanism confirmed independently.

`raw` is the emission to use. Integer emission is recorded as measured and
rejected, at this `target_sum`; it might behave differently at realistic
sequencing depth, where rounding is not a ~0.23-count-per-gene operation.

⚠️ **n = 1 per arm, no seed replicates, one panel, one cell line.** The
geometry differs between A and B (2048x5 vs 2048x4) and is a real if small
confound. And nothing here measures unseen-context transfer: the decoder never
sees perturbation labels, so this is whether a reconstruction PRESERVES
perturbation deltas, not whether a model predicts them. The 7/9 unattributed
DiverseRNA blocks mean jurkat cannot even be asserted to be an unseen context.
