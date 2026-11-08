# GitHub and Hugging Face release audit

Audit date: 2026-11-08. This audit preserves the frozen benchmark definitions,
source score tables, local checkpoints, and unrelated working-tree changes.

## GitHub

The GitHub `main` branch and local `main` both pointed at
`b0ac4649ae37758b82fd2444032d98f7effe5118` before this update. The unpublished
working tree contained the official Norman NF32ALLA recipe, five-seed evaluation
supplement, checkpoint card, reproduction changes, and official comparison builder.
This update publishes those records with a current results entrypoint.

The existing final score table remains frozen. The Norman supplement is separate;
`python scripts/official_score_tables.py` combines the two sources for display.
The output covers 11 benchmark/panel combinations and 110 comparison rows, each
validated against its coverage ledger for seeds 42–46 and splits 0–4. Headline
uncertainty is sample SD of seed means, after averaging splits within each seed.
Both mandatory control anchors are included. Only uncalibrated scores are used.
Norman additive/nonadditive are the headline strata; other panels remain separate.

## Hugging Face: verified remote state

The public [competitor checkpoint repository](https://huggingface.co/bendidiihab/Occam-competitor-checkpoints)
responded successfully at revision `d9b17d23a2e56450ad83e6116292c6f94238ab5f`
(last modified 2026-10-06). Its 219 files include 15 Norman Occam ablation
checkpoints: NF32ALL, NF32DC, and NF32DCA, each with seeds 42–46 under
`ablations/occam/norman_combo/<variant>/seed_<seed>/best_model.pth`.
There are no `razor/` paths and no NF32ALLA checkpoints in this public listing.
Competitor families remain under `cellflow/`, `perturbdiff/`, `state/`, and `txpert/`.
This verifies file presence and layout, not checkpoint contents or inference.

The private [Occam repository](https://huggingface.co/bendidiihab/Occam)
returned HTTP 401, and this execution session has no Hub authentication token.
Its current revision, checkpoint hashes, score contents, file count, and total
size could not be independently verified. The local October release records
identify NF32ALLA seeds 42–46 under `checkpoints/vcell/norman_combo_NF32ALLA/`,
with the root checkpoint an alias for seed 42; recorded digests are in the
[checkpoint card](../checkpoints/cards/norman_combo_NF32ALLA_CARD.md).
These local records are not a fresh private-Hub verification.
No Hub files or visibility settings were changed during this audit.

## Reproduction and validation

The official RHJ/H1/K562 arm is ABL_N_ALL, from
`scripts/grids/ablation/ABL_N_ALL.tsv` with `configs/rhj_fewshot_pertemb.yaml`.
The older `scripts/repro/rhj_train.sh` and default `rhj_eval.sh` reproduce Y06;
they remain historical reproduction paths. Pass the intended checkpoint explicitly
when evaluating ABL_N_ALL. Norman reproduction now uses
`configs/norman_combo_official.yaml`, NF32ALLA, and `decode_seq_mean`.

Validation used an isolated Python 3.12 environment for lightweight table and
release checks; the pinned reference training runtime is Python 3.8 with
PyTorch 2.4.1. The reference runtime and cell-eval are unavailable here, so this
audit does not claim a fresh training, inference, or cell-eval reproduction.
The checkout also contains pre-existing executable-bit changes; release staging
preserves the executable modes recorded in Git rather than publishing those changes.

The isolated release snapshot passed 25 focused release/recipe/documentation checks. All 1,400
headline metric values in the 350-row Norman supplement matched its uncalibrated
long-form source. The full repository runner on the original checkout reported
170 passed, 47 failed, and 84 skipped; missing torch/anndata/h5py, unavailable Git
on PATH, and pre-existing checkout permission/manifest issues prevent a clean
full-suite result. A legacy README assertion was updated to the official five-seed
results when the historical overview was replaced.
