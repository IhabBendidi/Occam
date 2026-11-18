# How each competitor handles DOUBLE perturbations, in its own published method

## Objective and question

**Standing objective.** Predict a perturbation's effect in a context where it was never
measured. `norman_combo` tests the composition axis: train on single-gene perturbations,
predict a PAIR where both genes were seen individually and the pair never was.

**What this document answers.** Before launching the four ported competitors on
`norman_combo`, which of them actually evaluate combinatorial perturbations in their own
paper, and by what mechanism? Each must be run **its own way**, not ours — otherwise the
number measures our choice of composition rule rather than their method.

**Sources.** Read 2026-09-22 from the papers and, where the port is vendored here, from
the method's own code. Two full texts are paywalled (`Nature Biotechnology` for TxPert,
`Cell` for State); for those the mechanism was confirmed from the arXiv version and from
the vendored implementation, which is the stronger evidence anyway.

## Summary

| method | evaluates doubles? | its own mechanism | our mode that matches |
|---|---|---|---|
| **TxPert** | **YES** — Norman, GEARS splits, doubles where both singles were seen | **additive**: sum the perturbation embeddings, decode once | **`sum`** |
| **CellFlow** | **YES** — Norman, including "novel combinations of genes individually observed" | **set aggregation**: multi-head attention *or* DeepSets over the perturbation set, then an FFN encoder | no exact match — a *learned* pooling, closest in spirit to `sum` |
| **State** | combinatorial **drug** perturbations, by **consecutive forward passes**. Not Norman genetic doubles | sequential application | **`decode_seq_*`** |
| **PerturbDiff** | **NO** | none published | — |

## TxPert — the closest match to our protocol, and it validates it

arXiv [2505.14919](https://arxiv.org/abs/2505.14919); published in *Nature Biotechnology*
2026 ([s41587-026-03113-4](https://www.nature.com/articles/s41587-026-03113-4), paywalled).

| | |
|---|---|
| dataset | **Norman**, "pre-defined splits were loaded from the GEARS setup" |
| split | doubles where "both individual perturbations had previously been seen in isolation during training" — **exactly our split** |
| encoding | additive by design: the latent shift "adds embeddings for all perturbed genes before decoding" |
| baselines | **additive baseline** `ŷ(i,j) = x̄ + δᵢ + δⱼ` with batch-matched control estimates; a general mean baseline (per-perturbation mean for seen, global perturbation mean for unseen); **GEARS**; **scLAMBDA** |
| metric | **Pearson Δ** primary, plus retrieval metrics |

**Confirmed in their own code**, `competitors/TxPert/gspp/models/gears_derivative.py:256-275`:

```python
self.pert_z = torch.zeros(batch_size, z_p.size(-1), device=self.device)
for i, perts in enumerate(pert_idxs):
    for p in perts:
        ...
        self.pert_z[i] = self.pert_z[i] + z_p[p]
```

A GEARS-format condition is `"A+B"`, so `perts` is `[id_A, id_B]` and the loop **sums the
two perturbation embeddings** into one latent shift, decoded once. Their singles are
`"A+ctrl"`, i.e. **the second slot IS the combinatorial slot** — the port already knew
this (the `-1` control token in that slot is what C28 was about).

**F-comp1. Our `norman_combo` design independently matches TxPert's published protocol
on all three axes that matter**: the same split semantics (both singles seen, pair
unseen), the same additive bar (`sum_of_singles` = their `x̄ + δᵢ + δⱼ`), and the same
primary metric (`pearson_delta`). That was not coordinated — it is convergence on the
obvious protocol, and it means our numbers and theirs are about the same quantity.

⚠️ It also means TxPert's native mechanism is **our `sum` mode**, which on our matrix is
the 2nd-best of 8 modes (additive marginal 0.7402, best on 6 of 12 arm×panel cells) but
degrades faster into non-additivity than the bar itself. TxPert should be run with its own
additive composition, not with `decode_seq_mean`.

## CellFlow — a learned set aggregation, and the port is already wired for it

bioRxiv [2025.04.11.648220](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1).

Evaluates genetic combinations on Norman (their Figure 4), across "combinations where all
genes were unseen during training, combinations containing one unseen gene, and **novel
combinations of genes that had been individually observed**" — the last is our split.

> "CellFlow implements two set aggregation schemes, multi-head attention and deep sets,
> followed by an encoder consisting of a feed-forward network."

Permutation-invariant by construction, handling "an arbitrary combination of perturbation
variables in a single representation". Baselines: **GEARS, biolord, scGPT, Identity**.
Metrics: **R²** and **energy distance** — neither is `pearson_delta`, so their published
numbers are not comparable to ours; only a re-run under our scorer is.

**The port is already built for two slots.** `competitors/CellFlow/occam_bridge/schema.py`
carries `max_combination_length`, emits `obs[gene_1] … obs[gene_K]` padded with a null
token, sorts the ids "so gene_1/gene_2 are deterministic", and its docstring says:

> "Norman uses `max_combination_length=2`; single-KO Replogle uses `1`."

So CellFlow needs **no new composition code** — it needs `max_combination_length=2` and a
Norman export. Its aggregation is learned, so unlike TxPert it is not equivalent to any of
our 12 fixed modes.

## State — sequential forward passes, on drugs not genes

*Cell* 2026, [S0092-8674(26)00921-9](https://www.cell.com/cell/fulltext/S0092-8674(26)00921-9)
(paywalled; description below is from the accessible summary, **not** verified against the
full text).

State's combinatorial experiment uses **drug**-perturbed cell states as input contexts and
predicts responses to combinatorial **drug** perturbations, "simulating these combinatorial
effects by running consecutive forward passes with the State model". We found no evaluation
on Norman genetic doubles.

⚠️ Consequence: **State has no published genetic-double protocol to copy.** Its mechanism —
consecutive forward passes — is exactly our `decode_seq_*` family (apply A, decode,
re-encode, apply B), so running it that way is faithful to its own combinatorial design
even though the drug/gene setting differs. That must be labelled as our extension of their
method, not as their published protocol. The port has no combinatorial support today
(`grep` finds nothing in `competitors/STATE`).

## PerturbDiff — no combinatorial evaluation at all

arXiv [2602.19685](https://arxiv.org/abs/2602.19685v1). Benchmarks are **PBMC** (cytokine
signalling), **Tahoe100M** (drugs) and **Replogle** (single-gene CRISPR knockdown). Norman
is cited once, about Replogle's characteristics, not as an evaluation. Baselines: CPA,
State, CellFlow, Squidiff, plus Mean and Linear.

> No combinatorial hold-out split is defined or evaluated, and the paper does not describe
> how multi-gene perturbations would be encoded.

Its MM-DiT takes parallel control/perturbed token streams with no multi-perturbation
mechanism. `grep` finds no combo handling in `competitors/PerturbDiff`.

⚠️ **Any `norman_combo` number for PerturbDiff is our invention, not their method.**
If it is run, the composition rule has to be chosen by us and recorded as a deviation as
large as any in `docs/rhj_competitors_results.md`.

## What this implies for launching them on `norman_combo`

Stated before any run, so the conclusion is not written after seeing the numbers:

1. **TxPert** — run with its native additive composition (`"A+B"` → summed embeddings).
   Nothing new to implement; the export needs pair labels resolved to two ids. Its number
   is directly comparable to its own paper's claim, and to our `sum` mode.
2. **CellFlow** — set `max_combination_length=2` and export Norman. Its aggregation is
   learned; a win over TxPert here would be evidence that learned pooling beats a fixed
   sum, which is the one genuinely new thing this panel could show.
3. **State** — no published genetic protocol. Run it as consecutive forward passes, its
   own drug-combination mechanism, and label it an extension.
4. **PerturbDiff** — has no mechanism. Either skip it on this benchmark, or run one
   explicitly-chosen rule and report it as ours. **Recommend skipping**: a number whose
   composition rule we invented tells us about our rule, not about their model.

⚠️ Their published metrics differ from ours (CellFlow reports R² and energy distance,
TxPert Pearson Δ, State neither on genetics), so **only a re-run under our scorer on our
frozen panel is comparable**. Nothing in their papers can be quoted into our table.

---

## Wiring them onto `norman_combo` — what was blocking, and what is left

**Status 2026-09-22: the shared data layer now works on `norman_combo`. No competitor is
training on it yet.** Written down so the remaining work is a list rather than a
rediscovery.

### Two blockers found by trying, both in the ONE reader all four competitors use

`competitors/shared/occam_data.py::load_occam_benchmark` refused `norman_combo` outright.

1. **No per-source manifest.** The reader needs `<bench>/files/<source>.json` — the path,
   the frozen sha256, the perturbation and batch columns, the per-perturbation cell
   counts. `rhj_fewshot`'s builder writes those; Norman's never did, so the reader failed
   with *"no per-file manifest for cell line(s) … found []"*.
   → `scripts/norman/norman_write_file_manifest.py` (new) derives it from
   `benchmark.json`'s own `sources` block and the source `.h5ad`, and **re-reads the
   sha256 and asserts it** against the frozen one rather than copying it. Run by
   `scripts/repro/build_benchmark_norman.sh`.
2. **The required cell lines were a module constant.** `OCCAM_CELL_LINES` is
   `("HEPG2", "jurkat", "RPE1")`, and the check was `for c in OCCAM_CELL_LINES` — so even
   with the manifest in place Norman died asking for HepG2. → the required lines now come
   from the benchmark's own `sources`.
3. **The Ensembl column name is not universal.** `rhj_fewshot`'s three files use
   `ensembl_gene_id`; `norman_2019_k562` uses `gene_id`. The reader hardcoded the first
   and refused with *"var has no ensembl_gene_id column"*. → the per-source manifest may
   name it (`gene_col`), otherwise known aliases are tried in order and the error lists
   all of them.

Verified after the fixes — RHJ unregressed, Norman reads:

| benchmark | axis | sources | exclusions |
|---|---|---|---|
| `rhj_fewshot` | 6,983 | HEPG2, RPE1, jurkat | 724 |
| `norman_combo` | 8,000 | K562 | 131 pairs |

and `load_occam_adata(..., cell_lines=["K562"])` returns a frame with **130 pairs
visible**, controls labelled, both `ensembl_gene_id` and `gene_symbol` in `.var`.

### What is left, per arm

The training legs still hardcode `data/benchmarks/rhj_fewshot`
(`scripts/slurm/competitor_train_*.sbatch`, `competitor_export.sbatch`,
`competitor_score.sbatch`, `scripts/eval_competitor.py:PANEL_DIR`).

| step | work | cost |
|---|---|---|
| thread a `BENCH` variable through the 4 train legs + export + score + `eval_competitor.py` | mechanical, one variable, but it touches every leg | ~1 h |
| **TxPert** — native additive sum; its GEARS condition is already `"A+B"` → two ids | config + export only, probably no model change | ~4 h/seed train |
| **CellFlow** — `max_combination_length=2` and a Norman config | the bridge is already written for two slots | ~10 h train |
| **State** — `decode_seq_mean`: predict A, feed the prediction back as the control, predict B, average both orders | new code in its adapter, at the EXPORT layer — no model change, because its own combinatorial design is consecutive forward passes | ~6 h train |
| **PerturbDiff** — same sequential `decode_seq_mean` | same, and it has **no published combinatorial protocol**, so this is our rule and must be labelled as one | ~8 h train |

⚠️ For State and PerturbDiff the sequential composition is implementable **entirely in the
export**, because export controls what is fed in: `pred_A = f(ctrl, A)`, then
`pred_AB = f(pred_A, B)`, averaged over both orders. Nothing inside either model changes,
which is what keeps the arm faithful to the method rather than to our port of it.

⚠️ Order matters for that family: on our own matrix `decode_seq_ba` beats `decode_seq_ab`
on all 6 arms and both panels (max |ab − ba| = 8.96e-02), which is why the mean of the two
orders is the default rather than either one.
---

## 2026-09-22 launch attempt — what I got wrong, and the real remaining work

**Three launches, twelve jobs each, all four arms failed every time. No competitor is
training on `norman_combo`.** Written up in full because the failures were informative and
because the estimate that preceded them was wrong.

### The assumption that was wrong

I estimated "~1 h of plumbing: thread a `BENCH` variable". The threading was necessary and
is done — but it was **not** the binding constraint. Each of the four arms **vendored its
own copy** of two things:

1. **the cell-line list.** `("HEPG2", "jurkat", "RPE1")` appears in
   `competitors/shared/occam_data.py`, `CellFlow/occam_bridge/occam.py`,
   `PerturbDiff/src/data/occam_bridge.py`, `STATE/.../occam_source.py` (as an alias
   table) and `TxPert/gspp/data/occam_loader.py`. **Five copies.** I fixed four and the
   fifth failed the next launch.
2. **a perturbation filter that drops `"A+B"`.** This is the real work, and it is the
   *same class of bug* that hit our own code earlier in this campaign — the bare
   `label in pert2id` lookup that emptied our validation split and that
   `resolve_pert_indices` / `_can_resolve` fixed. Every port has its own copy, in a
   different place, and a combinatorial label survives none of them.

### The three launches, and what each taught

| attempt | how long the arms survived | what it revealed |
|---|---|---|
| 18325396–407 | 12–52 s | the cell-line list was still the RHJ constant at the *adapter* call sites, not just in the manifest check |
| 18326450–461 | 31–101 s | past the cell-line gate; now each arm's **perturbation/split filter** drops all 131 pairs |
| 18326465 (cellflow only) | 31 s | `list(None)` — my own half-landed fix |

The arms refuse **early and loudly** rather than training on a silently shrunken set,
which is the port working as designed. Each message names the exact quantity lost:

```
TxPert       131 held-out perturbation(s) from the split file are absent from the data
             after vocabulary filtering (e.g. ['AHR+FEV', ...]). The scored set would
             not match the other competitor arms.
CellFlow     validation covers 0 perturbations, expected exactly the 20 in splits.json
STATE        validation split is empty ... Refusing to fall back to the test loader,
             which would select checkpoints on test. Current [splits]: {}
PerturbDiff  unknown cell line(s) ['HEPG2']; this benchmark has ['K562']
```

### What is actually left, per arm

| arm | where the pair is lost | what it needs |
|---|---|---|
| **TxPert** | `datamodule.py:_load_zarr_splits` — its `pert2id` extension ALREADY splits `"A+B"` into singles (`:585`), so the vocabulary is fine; the `obs[CONDITION]` column does not carry the pair through `_process_treatment_data` | make the condition column keep `"A+B"`; its model already sums the two ids |
| **CellFlow** | the validation frame has 0 of the 20 val pairs — the `gene_1`/`gene_2` slots are built by `occam_bridge/schema.py`, which is written for this, but the *row filter* that selects validation rows drops them first | thread the pair through the row filter, then `max_combination_length: 2` does the rest |
| **STATE** | `[splits]` resolves to `{}` on this benchmark — its split plumbing reads a TOML-shaped fold that the Norman `splits.json` does not present | map `splits.json`'s `val`/`test` pair lists into its fold structure |
| **PerturbDiff** | a **fifth** copy of the cell-line constants in `src/data/occam_bridge.py:66-77`, plus the same pair-filter problem behind it | fix the constants, then the filter |

### Honest estimate now

**Half a day to a day of per-arm surgery in four vendored codebases**, not one hour. Each
fix is small but each is in unfamiliar code with its own conventions, and the two arms with
NATIVE combinatorial mechanisms (TxPert, CellFlow) are exactly the two where the fix must
not disturb that mechanism — which is the entire reason for running them.

⚠️ **Do not "fix" this by filtering the pairs out of the split.** That would make each arm
score a different set from Occam and from each other, which is the one thing the frozen
cohort exists to prevent. The arms are right to refuse.

⚠️ The BENCH threading itself is done, tested and worth keeping regardless: the axis width
and all three digests are now read from the artifact by every leg, which is what makes
"same genes, same splits" checkable — and it caught three model dimensions hardcoded to
6983 that would have trained a 6,983-wide model against 8,000-wide targets without
crashing.

---

## ALL FOUR ARMS NOW RUN ON `norman_combo` (2026-09-22)

| arm | training | composition |
|---|---|---|
| **txpert** | **COMPLETED**, 15m37s | native additive sum |
| **cellflow** | running | native set aggregation, `max_combination_length: 2` |
| **state** | running | `decode_seq_mean` at the export boundary |
| **perturbdiff** | running | `decode_seq_mean`, + composed GenePT pair embeddings |

It took **six launches**. Every failure was a hardcoded `rhj_fewshot` assumption in a
different place, and every one of them refused loudly rather than training on wrong data —
which is the only reason this converged at all.

### The eight fixes, in the order they were found

| # | what was hardcoded | where | what it did on Norman |
|---|---|---|---|
| 1 | the required cell lines | `occam_data.load_occam_benchmark` | refused the benchmark outright |
| 2 | `cell_lines=OCCAM_CELL_LINES` as a DEFAULT ARG | `occam_data.load_occam_adata` | every caller asked for HepG2 on a K562 benchmark |
| 3 | the Ensembl column name | `occam_data._read_one` | "var has no ensembl_gene_id column" |
| 4 | `line == "HEPG2"` | `occam_data.occam_split_indices`, STATE ×2 | val + test came out EMPTY, everything became train |
| 5 | `EVALUATED_LINE = "HEPG2"` | PerturbDiff's split branch | same, plus "validation split is empty" |
| 6 | `occam_line_from_path`'s alias table | STATE ×1, PerturbDiff ×2 **(three copies)** | "cannot tell which Occam cell line 'K562' is" |
| 7 | the dataset list | PerturbDiff's TOML manifest | named the three RHJ h5ads |
| 8 | the gene count `6983` | PerturbDiff `input_dim`, `hidden_num`, **`pad_length`**; TxPert `hvg_number`; TxPert `test_cell_type: hepg2` | a 6,983-wide model against 8,000-wide targets — **does not crash** |

Plus two that are not hardcodes but genuine per-benchmark work:

**TxPert's `+ctrl` suffix was applied GLOBALLY.** `data_utils.py:267` read
`if not all("+ctrl" in c for c in <all perturbed conditions>)` and then rewrote **every**
perturbed row. That is correct only when every perturbation is single-gene: on a
combinatorial corpus the singles fail the test, so the whole column is rewritten and a pair
`"AHR+FEV"` becomes `"AHR+FEV+ctrl"` — **three slots in a two-slot format**. All 131
held-out pairs then matched nothing. Now applied **per row**, only to rows with no `+`,
which is also the original intent: the suffix exists to fill the *unused* combinatorial slot
of a single-gene perturbation.

**PerturbDiff's GenePT table covered 8 of 236 conditions.**
`genept_occam_rhj_fewshot.pkl` is keyed by *that* benchmark's symbols. Rebuilt with
`scripts/build_genept_for_occam.py --bench-dir data/benchmarks/norman_combo`: all **105
single genes** resolve from the full GenePT source.

⚠️ The remaining 131 keys were the PAIRS, and a GenePT table is per GENE. `--compose-pairs
mean` (new, default) gives `A+B` the **mean** of its components' embeddings. **This is our
choice, not GenePT's and not PerturbDiff's**, recorded in the table's `.json` manifest as
`composed_pairs`:

* **MEAN not SUM** so the norm stays in the range the conditioning MLP saw in training,
  which only ever sees singles (Norman trains on 105 single conditions);
* it is the same assumption the arm's emission already makes — `decode_seq_mean` averages
  the two orders — so conditioning and emission agree rather than pulling apart;
* and it is **symmetric**, which a pair must be: `A+B` and `B+A` are one condition.

### What this cost, and what it bought

The "~1 h of plumbing" estimate was wrong by roughly a factor of six. What made it
converge was that **every one of these failures was loud**: a refusal naming the exact
quantity lost, at 12–101 seconds, instead of a run that trains for 8 hours on a silently
shrunken cohort and reports a plausible number. Four of the eight would have been
invisible without a guard.

⚠️ `rhj_fewshot` is unregressed throughout: `tests/test_occam_data_reader.py` still
reproduces its splits exactly (finetune 612,122 / val 6,731 / test 37,490), and the
bench tag is EMPTY on it so all 296 existing run directories keep their names.

---

## The EXPORT leg had its own five hardcodes (2026-09-22, later)

Training completing is not a score. TxPert trained to completion (15m37s) and CellFlow too
(36m45s), and **both then lost all 5 splits in the export** — because the Norman-derived
settings had only been threaded into the TRAIN leg.

| # | what | where | what it did |
|---|---|---|---|
| 9 | the datamodule overrides | TxPert `export_occam.py` composed hydra with only `bench_dir` + `seed`, so `test_cell_type: hepg2` came from the config | "Validation split is empty." **after** training completed — 5/5 splits lost |
| 10 | the run directory | CellFlow's export defaulted to `occam_s42`, the **rhj** run | loaded a 6,983-component PCA basis for an 8,000-gene panel |
| 11 | the checkpoint directory | TxPert and STATE resolved `occam_s{seed}` with no bench tag | would have scored the **wrong checkpoint** on the right panel |
| 12 | `--cell-type default="HEPG2"` | all three of CellFlow / STATE / PerturbDiff | next failure in line |
| 13 | the sampler's control pool | PerturbDiff `sampling_generation_helpers.py:188`, `cell_lines=["HEPG2"]` | validation cell-eval raised, the callback **swallowed it**, and the run died 43 min later inside `ModelCheckpoint` looking for a score that was never published |

**#10 is the one worth remembering.** An export that silently loads the wrong run's
artifacts produces a number, not an error. It was caught only because the emission-space
guard compares the axis (`"the gene axis is not ours: 6983 genes vs the frozen 8000"`).
An arm whose two benchmarks happened to share a gene count would have reported a
plausible, wrong result.

**#13 is the one that cost the most wall clock.** The validation callback catches its own
exceptions and logs `"validation cell-eval failed at epoch 43; not publishing a score"`,
so the actual cause was two steps removed from the crash. The fix is one line; finding it
meant reading past a `MisconfigurationException` about a monitored key.

### The design lesson, applied

Every one of #9–#13 is the same mistake: **a benchmark-dependent value passed in by a
launcher instead of derived where it is used.** The fixes derive them inside the adapter,
from `--bench-dir`, which the adapter already receives:

* TxPert's export computes `hvg_number` and `test_cell_type` from `benchmark.json` itself,
  so the two legs cannot disagree and no launcher can forget one.
* All three exports resolve the evaluated cell line from the artifact when `--cell-type`
  is empty (now the default).
* CellFlow's and STATE's and TxPert's run/checkpoint directories carry the bench tag,
  computed from `--bench-dir`.

### CellFlow needed one genuine piece of combinatorial work

`KeyError: "['gene_2'] not in index"`. Its export built a **one-slot** covariate frame
(`gene_1` only) against a model trained at `max_combination_length: 2`. Now it builds every
slot exactly as `occam_bridge/schema.py` does at training time — components **sorted** (so
`A+B` and `B+A` are one condition) and padded with the same `NULL` token — and it reads the
slot count off **the trained object's own obs columns** rather than a constant, so the
adapter follows whatever the run was trained at.
