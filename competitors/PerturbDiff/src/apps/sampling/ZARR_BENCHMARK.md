# `src/apps/sampling/` — generation and evaluation

This is where the benchmark comparison against STATE is won or lost.
Companion to [../../../ZARR_BENCHMARK.md](../../../ZARR_BENCHMARK.md).

| File | Role |
|---|---|
| `sampling_setup.py` | datamodule dispatch + `PlModel.load_from_checkpoint` |
| `sampling_generation.py` | the generation loop, metrics, adata assembly |
| `sampling_generation_helpers.py` | sampler resolution, gene list, covariate decoding, control adata, obs building |
| `sampling_io.py` | writes the two `.h5ad` files |
| `sampling_utils.py` | `setup_device` |

---

## 1. `sampling_setup.py`

`build_sampling_datamodule` (9-57) mirrors the training dispatch; ✅ `IhabxcellZarrFinetune` is
registered there too. Note the inconsistent batch size: `PBMCFinetune` uses `cfg.sampling.batch_size`
(line 21) while `Tahoe100mFinetune`/`ReplogleFinetune` use `cfg.optimization.micro_batch_size`
(line 28). Pick one deliberately for the zarr branch.

`load_sampling_model` (72-103) loads the checkpoint's **own** `hyper_parameters` for
`cov_encoding_cfg` / `model_cfg` / `optimizer_cfg` (lines 89-96), overriding only
`celltype_encoding` from the CLI (line 90). So the model is rebuilt with the *training-time*
covariate cardinalities — but the datamodule is rebuilt from the *current* config. **If the two
disagree, the pert/celltype/batch integer codes silently shift and every prediction is
conditioned on the wrong covariate.** STATE avoids this by pickling `data_module.torch`
at train time and reloading it (`_predict.py:297-328`).

**Recommended change:** dump `pert_dict`, `cell_type_dict`, `batch_dict` and the selected-gene list
alongside the checkpoint at train time, and assert equality in `load_sampling_model`.

---

## 2. `sampling_generation.generate_samples` (24-232)

```
dataloader = datamodule.val_dataloader()[1]        # line 45 — the TEST split
for batch_idx in range(num_sampled_batches):       # line 97
    batch = next(iter)                              # NOTE: iterator, not a for-loop over the loader
    batch["batch_emb"] = model._encode_covariates(batch)
    sample, traj = sample_fn(model.model, (B, S, G), self_condition=…, …)
    mask = ~batch["is_padded_list"].bool()          # drop replacement-padding
    ... r2_score + energy-MMD per batch
adatas = AnnData(concat([samples, X_ctrl])), AnnData(concat([truths, X_ctrl]))
save_adata(...)
```

### Benchmark-parity issues (all addressed)

| Line(s) | Issue | Fix |
|---|---|---|
| ~~60-63~~ | ~~`num_sampled_batches` defaults to 4~~ | **DONE** — default is now `null` (full coverage), with `PARTIAL EVAL` warnings, batch capping and a coverage check; see §2.1 |
| 45 | index `[1]` into `val_dataloader()` hard-codes the split order | keep `all_split_names = ["validation","test"]` or select by name |
| ~~93-94~~ | ~~`load_selected_genes` + hard-coded `assert len == 2000`~~ | **DONE** — accepts `.txt`, validates against `data.pad_length` |
| 148-151 | `np_mask = np.isin(batch["col_genes"][0], _genes)` re-derives the gene mask per batch | with zarr the columns are already in selected order — this becomes the identity |
| ~~185-196~~ | ~~`load_ctrl_adata` reads a separate `replogle.h5ad`~~ | **DONE** — `build_zarr_controls` reads them from the test split |
| 192-194 | multiplies back by `normalize_counts` | keep `normalize_counts: false` for zarr (X is already `log1p(1e4)`) |
| 198-217 | the `len(np_mask) > 2000` re-ordering branch | dead for a 2000-gene zarr panel; leave but verify it does not trigger |
| 155-159 | R² on the batch **mean** + energy MMD | keep as a sanity signal, but the benchmark number must come from cell-eval |

### 2.1 Coverage and determinism (done)

Three changes landed here, because a truncated or unordered eval silently produces a number that
looks like a benchmark result but is not one:

1. **Full coverage by default.** `configs/rawdata_diffusion_sampling.yaml` now has
   `num_sampled_batches: null` (was `4`). The old default evaluated ~4 batches of the test split
   and reported R²/MMD as if complete — 24 of 96 cells on the smoke dataset, and it would have
   been 24 of ~26878 on the real hepg2 split.
2. **Truncation is loud.** An integer value logs a `PARTIAL EVAL` warning up front (`n of N
   batches, ~x% of the split`) and again after generation (`generated n of N test cells`).
   Over-requesting is capped at `len(dataloader)` instead of raising `StopIteration`.
3. **Order is deterministic and verified.** The test loader is built with `shuffle=False`
   (`data_module.build_val_dataloaders`), so cell-set blocks are emitted in construction order
   and the replacement-padding comes from a seeded RNG. `generate_samples` now warns if it ever
   sees `shuffle=True`, and cross-checks the generated cell count against
   `dataloader.sampler.original_total` (the real, non-padded split size), logging either
   `Full test-split coverage: N cells` or a mismatch warning.

The old sampler warning — *"still shuffling within batches … do not use this sampler for
generation"* — was misleading: it fired in the `shuffle=False` branch, the deterministic one. It is
now an accurate `debug` message.

Verified: two sampling runs of the same checkpoint produced **bit-identical** `X` and identical
obs ordering (`np.array_equal` → `True`, max abs diff `0.0`), and full coverage reported 96/96
cells against the smoke split.

Residual run-to-run variation is governed entirely by `cfg.optimization.seed`
(`pl.seed_everything` in the entrypoint), which fixes both the control-cell pairing in
`mapping_cells` and the diffusion noise. Change the seed and predictions change — that is
expected for a generative model; average over seeds if you want an error bar.

---

## 3. `sampling_generation_helpers.py`

* `resolve_sampling_runner` (14-45): DDIM (`ddim_sample_loop`, `eta`, `guidance_strength`) vs
  DDPM (`p_sample_loop`, `nw`, `start_guide_steps`). The README's `COMMON_SAMPLING` sets
  `sampling.use_ddim=true`. `start_time` is clipped to `cfg.model.steps` (23-24).
* `load_selected_genes` (48-59): now accepts the benchmark's `.txt` panel as well as upstream's
  pickles, and validates the length against `data.pad_length` instead of a hard-coded 2000; still
  rewrites `cfg.data.selected_gene_file` from the `sample_*_only` flags.
* `build_gene_embedding_cache` (62-75) / `build_self_condition` (78-86): conditioning tensors.
* `collect_batch_covariates` (89-115): **inverts** the global dicts back to per-dataset category
  names — including `datamodule.batch_dict[ds_name + "_" + str(v)]` (line 111). This mirrors
  `dataset_core.get_covarates:186`; both must use the same namespacing or the `np.vectorize`
  lookup returns `None` and the obs column fills with nulls.
* `load_ctrl_adata` (118-135): dataset-specific `read_h5ad` with
  `assert len(ctrl_adata) == 4976` for replogle/hepg2 non-targeting (line 125).
  **The zarr hepg2 dataset has exactly 4976 `negative_control` rows** — same cells, so the
  numbers will line up if you source controls from zarr.
* `build_x_ctrl` (138-169): masks + reorders control columns into the selected-gene order.
* `build_obs_data` (172-215): hard-coded obs schemas per dataset —
  replogle → `{gene, cell_line, gem_group}`, pbmc → `{cytokine, cell_type, donor}`,
  tahoe → `{drugname_drugconc, cell_line, plate}`.

### Zarr branch (implemented)

`build_zarr_controls` / `build_obs_data_zarr` read controls directly from the test-split rows and
emit a **STATE-compatible obs**, selected by `data.source_format == "zarr"` in `generate_samples`
(`load_ctrl_adata` and its replogle-specific `assert len == 4976` stay on the h5ad path):

```python
obs = pd.DataFrame({
    "gene":      np.concatenate([all_pert, ["empty"] * n_ctrl]),   # Ensembl ids / "empty"
    "cell_type": np.concatenate([all_celltype, ctrl_cell_type]),
    "batch_center": np.concatenate([all_batch, ctrl_batch]),
})
```

Using the same `pert_col` / `cell_type_key` names STATE passes to `MetricsEvaluator`
(`pert_col=data_module.pert_col`, `celltype_col=data_module.cell_type_key`) means the two models'
adatas can be scored by the identical call.

---

## 4. `sampling_io.save_adata` (10-40)

Writes `diffusion_predict_<ts>.h5ad` and `diffusion_true_<ts>.h5ad` into
`cfg.sampling.output_dir` (or `<save_dir_path>/samples`). `cfg.sampling.output_format` is read
(line 33) but unused — the format is always h5ad.

To match STATE's layout, write `adata_pred.h5ad` / `adata_real.h5ad` into
`<output_dir>/eval_<checkpoint_stem>/` so the two result trees are directly comparable.

---

## 5. `sampling_eval_celleval.py` (implemented)

A port of `../../../../STATE/state/src/state/_cli/_tx/_predict.py:896-946`, enabled with
`sampling.run_cell_eval=true`:

```python
clip_anndata_values(adata_pred, max_value=14.0)      # and adata_real   (_predict.py:174-183)
ct_real = split_anndata_on_celltype(adata_real, celltype_col=cell_type_key)
ct_pred = split_anndata_on_celltype(adata_pred, celltype_col=cell_type_key)
for ct in ct_real:
    MetricsEvaluator(
        adata_pred=ct_pred[ct], adata_real=ct_real[ct],
        control_pert="empty", pert_col="gene",
        outdir=results_dir, prefix=ct,
        pdex_kwargs=dict(exp_post_agg=True, is_log1p=True),
        num_threads=len(os.sched_getaffinity(0)),     # cgroup-aware; see _predict.py:5-16
    ).compute(profile="full",
              skip_metrics=["pearson_edistance", "clustering_agreement"])
```

Every one of those arguments must match STATE's exactly for the comparison to mean anything —
especially `control_pert`, `pdex_kwargs`, the `[0, 14]` clip, and the skipped metrics.

cell-eval is pinned to **`==0.7.2`**, the version STATE has installed, so metric implementations
cannot drift between the two competitors. (0.7.2 and 0.8.2 were both checked and emit the same
file set with the same 27 metric columns; the pin removes the question.)

Verified against STATE's own outputs: identical obs columns (`gene` / `cell_type` /
`batch_center`), Ensembl perturbation ids, cell type `HepG2`, **4976 control rows on both sides**,
and `HepG2_results.csv` with the same 27 metric columns. At `num_sampled_batches=null` the adata is
26,878 + 4,976 = 31,854 rows, exactly STATE's.

Expected outputs, matching
`../../../../STATE/state/test_replogle_zarr/hepg2_holdout_ram/eval_last.ckpt/`:

```
HepG2_results.csv
HepG2_pred_de.csv
HepG2_real_de.csv
HepG2_inject_ctrl_agg_results.csv
HepG2_rcstrct_ctrl_agg_results.csv
```

### Control convention

PerturbDiff already concatenates the **real** control matrix into *both* the pred and true adatas
(`sampling_generation.py:219-220`), which is exactly what STATE's `--inject-real-controls` does
(`_predict.py:185-244`). So compare against STATE runs that used that flag, or re-run STATE
without it — but do not mix the two.
