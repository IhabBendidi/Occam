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

