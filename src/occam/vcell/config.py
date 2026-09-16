"""Configuration dataclass for Virtual Cell Mode."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VirtualCellConfig:
    """Configuration for virtual cell perturbation effect prediction."""

    # --- Checkpoint ---
    encoder_checkpoint: str = ""  # unified encoder+decoder checkpoint (phase 3)
    pretrained_encoder: str = ""  # encoder-only checkpoint (train decoder from scratch)
    freeze_encoder: bool = True
    encoder_lr: float = 1e-5  # only used when freeze_encoder=False
    # Hold the encoder frozen for this fraction of training, then release it at
    # encoder_lr. 0.0 = released from step 0 (previous behaviour).
    encoder_lr_delay_frac: float = 0.0
    freeze_decoder: bool = False  # freeze pretrained decoder weights

    # --- GNN ---
    graph_type: str = "go"  # go, string, reactome, pxmap, txmap, depmap, dense, etc.
    graph_cache_dir: str = "data/graphs"
    master_gene_set_path: str = (
        "data/graphs/master_gene_set_sorted.csv"
    )
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 4
    gnn_num_heads: int = 2
    gnn_dropout: float = 0.0
    gnn_skip: str = "none"  # none, skip_cat, skip_cat_learned, skip_sum_learned
    gnn_layer_type: str = "gat_v2"  # gat_v2, gcn, gin, gat
    pert_latent_dim: int = 64
    use_edge_weight: bool = False
    no_gnn: bool = False
    gnn_activation: str = "gelu"
    graph_reduce2perts: bool = True
    graph_edge_filter_mode: Optional[str] = None  # e.g. "top_20", "percentile_90"

    # --- Perturbation model selection ---
    pert_model_type: str = "gnn"  # "gnn" or "adjacency_map"
    graph_types: Optional[List[str]] = None  # multi-KG list; None = use graph_type

    # --- Adjacency map (pert_model_type="adjacency_map") ---
    adj_hidden_dim: int = 128
    adj_dropout: float = 0.2
    adj_deg: int = 1
    adj_normalize: str = "none"
    adj_self_loops: bool = False
    adj_only_reverse: bool = True
    adj_make_undirected: bool = True
    adj_use_ffn: bool = True
    adj_activation: str = "relu"

    # --- Per-KG settings (for multi-KG loading) ---
    kg_norm_weights: Optional[List[str]] = None  # KGs that get norm_weights=True (default: ["go"])

    # --- Data-aware pert vocab ---
    data_perts_only: bool = False

    # --- Decoder ---
    decoder_hidden_dim: int = 3072
    decoder_num_blocks: int = 4
    decoder_dropout: float = 0.3
    decoder_se_reduction: int = 16
    # Only used when building a decoder FROM SCRATCH. When a unified checkpoint is loaded the
    # activation comes from its decoder_config, which is mandatory: activation modules hold no
    # parameters, so a softplus decoder loads strict=True under this default and silently runs
    # rectified_sigmoid (the dead-unit trap, NEURIPS_SCALING_AND_SEED_RESULTS.md §12.7).
    decoder_output_activation: str = "rectified_sigmoid"

    # --- Training ---
    lr: float = 1e-4
    decoder_lr: Optional[float] = None  # decoder learning rate (defaults to lr if None)
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    gradient_clip: float = 1.0
    eta_min: float = 1e-6
    loss_type: str = "smooth_l1"  # smooth_l1, mse, log_cosh, poisson, poisson_smooth_l1, nb, zinb
    smooth_l1_beta: float = 2.0
    # zinb/nb dispersion (Stage C reuse). Only active for loss_type in {nb,zinb}; ignored for smooth_l1 etc.
    vc_load_dispersion: bool = True      # warm-start log_theta/logit_pi from the unified ckpt's saved dispersion
    vc_dispersion_lr: Optional[float] = None  # separate LR for dispersion params (None -> main lr); a sweep axis
    vc_freeze_dispersion: bool = False   # freeze dispersion (ablation). NOTE: freeze_decoder forces this True.
    cosine_loss_weight: float = 0.5  # weight of cosine loss (0 = recon only, 1 = cosine only)
    recon_loss_scale: float = 1.0  # divide recon_loss by this before combining (>1 shrinks Poisson to match smooth_l1 scale)
    poisson_weight: float = 0.5  # blend weight for poisson_smooth_l1: pw*poisson + (1-pw)*smooth_l1
    warmup_epochs: int = 0

    # --- Hurdle-Gamma loss (SC samples in mixed mode) ---
    hurdle_loss_weight: float = 0.0        # 0 = disabled, >0 = enable hurdle for SC
    hurdle_q_min: float = 0.05             # clamp floor for (1-π) in mean preservation
    hurdle_warmup_epochs: int = 5          # epochs with stop_grad on π head
    hurdle_deterministic_pred: bool = True  # deterministic sparsification at prediction time

    # --- Data ---
    data_paths: List[str] = field(default_factory=list)
    target_sum: float = 1e4
    # Skip transforms when data is already library-normalized / log1p-transformed
    # (auto-inherited from the encoder checkpoint config; CLI flags force-enable via OR).
    skip_normalize: bool = False
    skip_log1p: bool = False
    n_controls_per_sample: int = 5
    min_controls_per_cell_line: int = 0  # exclude cell lines with fewer controls (0 = keep all)
    include_controls_as_training: bool = True
    gene_name_col: str = "ensembl_gene_id"  # var column for expression genes
    pert_gene_col: str = "gene_name"  # obs column for perturbation gene symbols
    pert_ensembl_col: str = "ensembl_gene_id"  # obs column for perturbation ensembl IDs
    optim_chunk_size: int = 8192
    optim_buffer_chunks: int = 4
    # ---- Streaming single-cell mode (large corpora that cannot be cached) ----
    # The cached path (build_mixed_dataset) holds every pair in RAM — 909 GB of cache for
    # four datasets — which does not survive adding the ~20 M-cell CD4 T-cell screen.
    # stream_single_cell routes to VirtualCellStreamingDataset instead: controls in
    # memory, perturbed cells streamed from disk. Single-cell only, no averaging.
    stream_single_cell: bool = False
    # --- streaming v2 (PIPELINE_AUDIT_2026-09-09 / EXECUTION_PLAN Phase 2) ---
    stream_v2: bool = False            # sparse multi-worker loader, GPU densify, embedded control pools
    stream_interleave: str = "roundrobin"  # or "proportional": source-mixed to the end of the epoch (F162)
    control_cache_dir: Optional[str] = None  # on-disk cache for control embeddings (keyed on mtime+encoder)
    control_scan_threads: int = 6      # HDF5 reader threads for the one-off control scan
    amp_dtype: str = "fp16"            # "bf16" drops the GradScaler (no zero-scale failure mode)
    bench_batches: int = 0             # >0: stop the first epoch after N batches and print samples/s
    min_pert_snr: Optional[float] = None   # stream_v2: drop (line, pert) groups with effect snr below this
    max_pert_snr: Optional[float] = None   # stream_v2: keep only groups with snr BELOW this (weak-only control arm)
    pert_effect_dir: str = "cache/pert_effects"  # table from scripts/pert_effect_snr.py
    effect_keep_perts: Optional[str] = None      # CSV of perts exempt from the filter (the H1 panel)
    # --- F170 levers: denoised targets / inputs, and effect size in the loss ---
    target_pb_k: int = 1        # 1 = single-cell target (default); k>1 = k-cell mean; <=0 = exact group mean
    ctrl_avg_k: int = 1         # >1 = average k control EMBEDDINGS as the model input (train + eval must match)
    delta_mag_weight: float = 0.0   # weight of (‖pred-ctrl_mean‖/‖group_mean-ctrl_mean‖ - 1)^2
    pert_mean_dir: str = "cache/pert_means"   # table from scripts/pert_group_means.py
    loss_norm: str = "pooled"       # "per_cell": mean over each cell's own masked genes, then over cells (F174)
    source_weight: str = "none"     # "balanced"|"sqrt": per-sample weight equalising source share (F174)
    # F181/E3 — condition the perturbation code on (source, assay modality). all9 mixes
    # incompatible estimands: joung_2023 is OVEREXPRESSION (+0.85 self-delta) inside an
    # all-knockdown corpus, and sources span 5.9x in effect norm. "add" makes the estimand
    # an input instead of noise. H1 inference uses the neutral source + knockdown modality.
    source_cond: str = "none"       # "none"|"add"
    source_cond_slots: int = 64     # embedding rows; index n = NEUTRAL (inference)
    source_dropout: float = 0.25    # P(route a training sample through NEUTRAL) so it is trained
    # F181/E4 — divide each source's TARGET delta by its own median ||Δ|| so all sources
    # share the essential scale. Fixes the 5.9x penetrance spread; does NOT fix sign.
    source_target_scale: str = "none"   # "none"|"median"
    source_scale_json: str = ""         # default cache/source_delta_scale.json
    panel_ctrl_avg_k: int = 1      # eval-side control averaging; MUST equal ctrl_avg_k for a matched read-out
    # obs columns joined with "_" to synthesize a missing `cell_line` (T-cell ships
    # `donor` + `condition` and no cell_line; each donor x condition is its own context)
    cell_line_cols: Optional[List[str]] = None
    # memory guard on control pools: 74,678 controls x 12 contexts at 44,728 genes is
    # ~160 GB, against a 700 GB job. 0 = uncapped.
    max_controls_per_cell_line: int = 0
    # obs column defining a COARSE group whose pooled controls a cell line may borrow
    # when it has none of its own (e.g. `cell_type`: all Feng is `iPSC`, and its
    # genome-wide screens ship 12 and 36 NTCs across 34 and 24 donor lines). None = the
    # group key is the cell line itself, i.e. cell-line matching only, no borrowing.
    control_fallback_col: Optional[str] = None
    # files that lend their CONTROLS to the pools but contribute no perturbations
    control_only_data_paths: Optional[List[str]] = None
    # a cell line's control pool must span at least this many distinct batches. 0 = off.
    # ⚠️ NOT a proxy for control quality: CD4 T-cell has ONE control batch per
    # donor x condition and is perfectly matched (controls and perturbations share it),
    # while Feng genome-wide has more batches and zero overlap. Use with the measured
    # per-line table, never as a default.
    min_control_batches_per_cell_line: int = 0
    # a cell line's controls must share >=1 batch with its own perturbations. This is the
    # criterion that actually means "same experiment"; it drops every borrowed-control
    # pair (all of Feng genome-wide, and with it 197/300 VCC targets).
    require_control_batch_overlap: bool = False
    # ABLATION: ignore per-line pools entirely and always draw from the group pool.
    # Brackets the hierarchical setting from the permissive side. Never a default.
    control_force_group: bool = False
    # per-epoch cap on perturbed cells per (cell_line, perturbation); bounds epoch COST,
    # and the retained subset rotates with the epoch so the corpus is still covered
    max_cells_per_pert_stream: int = 0
    # depth FLOOR: drop (cell_line, perturbation) groups with fewer than this many cells.
    # Feng genome-wide has a MEDIAN of 1-2 cells per group (74%/58% at <=2) against 45-121
    # everywhere else, so its deltas are one cell minus a borrowed control.
    min_cells_per_pert_stream: int = 0
    # STREAMING PAIRING. Default False = a perturbed cell is paired with a control drawn
    # fresh every epoch (`rng.choice`, seeded on seed+epoch), so its target-given-input
    # relation re-randomises each pass. True = the control index is a pure hash of
    # (file, row), and the per-epoch cap keeps the SAME cells, so every input has exactly
    # one output for the whole run — the property the CACHED path has by construction and
    # the streaming path never had. F93 measured the cached path beating a volume-matched
    # streaming path by 0.011 `all` / 0.041 `seen`; this flag tests whether fixed pairing
    # is why. See also the batch-matching difference recorded in F97.
    stream_fixed_pairs: bool = False
    # STREAMING CONTROL MATCHING. Default False = a control is drawn from the cell LINE
    # pool. True = from the same (cell_line, batch) pool, falling back to the line pool
    # when that batch has no controls. F97: the CACHED path is 100% batch-matched
    # (123,711/123,711 pairs in the best checkpoint's own log) while streaming has never
    # been — a Feng-driven decision (its controls and perturbations have disjoint batch
    # ids) inherited unexamined by essential-only runs where batch matching IS available.
    # A batch-matched control removes the batch effect from the delta, which is the signal.
    stream_batch_match: bool = False
    # interleave chunks across files so each batch mixes data sources rather than being
    # dominated by whichever file owns most chunks (T-cell is ~95% of them)
    balance_sources: bool = True

    pseudobulk: bool = False  # aggregate to pseudobulk profiles per (batch, perturbation)
    pseudobulk_by_cellline: bool = False  # aggregate per (cell_line, perturbation) ignoring batch
    mixed_pb_sc: bool = False  # train on both pseudobulk + deterministic single-cell paired samples
    sc_only: bool = False  # keep only SC samples from mixed dataset (drop PB)
    # Cap SC pairs per (cell_line, perturbation), summed over batches. 0 = uncapped.
    # MEMORY control: pairs are dense (n, 44728) float32 for BOTH control and target plus a
    # tiled gene mask, so an uncapped 4-file whole-genome build measures ~1,023 GB;
    # cap 32 -> ~213 GB, cap 64 -> ~392 GB. See build_mixed_dataset().
    max_sc_pairs_per_pert: int = 0
    # Where the mixed PB+SC dataset cache is written. Default (None) is the FIRST data file's
    # own directory, which for the shared biomodal eval folder means dropping a 100-400 GB
    # cache into read-mostly reference data. Point it at scratch for whole-genome builds.
    sc_cache_dir: Optional[str] = None
    # With freeze_encoder, cache the control embedding for each fixed pair instead of
    # re-encoding it every epoch. Exact (encoder is dropout=0 and deterministic), and it
    # removes one of two per-step frozen encoder passes plus ~171 GB of host RAM on the
    # whole-genome build. Escape hatch only.
    precompute_paired_embeddings: bool = True
    # Control identity samples on the SINGLE-CELL path: control in -> control out, carrying
    # the model's null perturbation token (pert_indices=[-1]). Expressed as a fraction of the
    # perturbation pairs in the same file. 0 = off.
    #
    # `include_controls_as_training` only ever applied to the pseudobulk path, so --sc_only
    # discards it and the model otherwise never learns what "no perturbation" looks like.
    # That matters because the H1 evaluation scores deltas against injected REAL controls:
    # drift in the model's control state is charged to every perturbation.
    control_identity_ratio: float = 0.0
    # 'exact'  : the input cell IS the target cell (true autoencoding of a control)
    # 'paired' : a different control cell from the same batch, matching how perturbation
    #            pairs are built (input and target are never the same cell)
    control_identity_mode: str = "exact"
    batch_col: str = "dataset_batch_num"  # obs column identifying experimental batches

    # --- Splits ---
    train_fraction: float = 0.75
    val_fraction: float = 0.05
    test_fraction: float = 0.20
    holdout_cell_line: Optional[str] = None
    # Put EVERY perturbation of every training cell line into train. Correct when the test
    # set is an external held-out cell line, since an in-domain test split would only
    # remove training signal without measuring the axis being studied. Validation then has
    # to come from outside the corpus (see panel_val_frac).
    all_perts_train: bool = False
    split_seed: int = 42
    split_json: Optional[str] = None  # path to load/save splits JSON

    # --- Combinatorial perturbations ---
    # Decompose a double perturbation "A+B" into its members and apply them
    # SEQUENTIALLY in latent space: control + A -> cell_A, then cell_A + B ->
    # cell_AB. Off by default; only meaningful on a corpus that actually
    # contains doubles whose singles are also present.
    combo_decomposition: bool = False
    # Enforce order-invariance: run A-then-B and B-then-A and penalise
    # disagreement, so the composition converges to the same cell_AB either way.
    combo_symmetric: bool = True
    combo_symmetry_weight: float = 0.1
    # Supervise the intermediate cell_A against the observed single-perturbation
    # profile when that single is available as a training target (pseudobulk only).
    combo_intermediate_weight: float = 0.0

    # Write the final-epoch test predictions to disk. The research tree had a
    # separate prediction writer per benchmark dataset, each gated on its own
    # mutually exclusive mode flag. Those datasets and their writers are gone;
    # panel evaluation (occam.vcell.panel) is the supported path.
    save_test_predictions: bool = False

    # --- H1 held-out CELL LINE mode (VCC 2026 proxy) ---
    # The only generalisation axis VCC 2026 scores, and the one no run in this repo has
    # ever been trained or selected for (`holdout_cell_line: None` in all 1,971 runs;
    # VCCv2/CLAUDE.md §10.-1 measured Chain A at pearson_delta 0.0015 on H1 vs 0.332 on
    # K562, a line it was trained on). H1 is held out by CONSTRUCTION here: it lives in a
    # different corpus (Arc's 2025 hESC set) and is simply never listed in data_paths, so
    # no split logic can leak it.
    panel_eval_enabled: bool = False
    panel_dir: str = "data/panels/h1_hesc"


    panel_real_path: str = ""            # default {panel_dir}/panel_real.h5ad  (SCORING side)
    panel_model_controls_path: str = ""  # default {panel_dir}/panel_controls_model.h5ad
    panel_symbol_map: str = "data/panels/gene_symbol_to_ensembl.csv"
    panel_pert_col: str = "target_gene"
    panel_control_label: str = "non-targeting"
    # Control fed to the model per predicted cell. 'single' = one real model-visible
    # control cell (NO averaging); 'batch' = same-batch control cell.
    panel_eval_control: str = "single"
    # Which panels to score, comma-separated. Every H1 perturbation is predicted; these
    # decide how the results are REPORTED.
    #   all    - all 255 H1 perturbations. THE HEADLINE NUMBER. Also the only panel that is
    #            comparable across arms and against the 68 scored arms in VCCv2/RUNS.md,
    #            because it is the same panel those used.
    #   seen   - unseen cell line, SEEN perturbation   (trained on, in other cell lines)
    #   unseen - unseen cell line, UNSEEN perturbation (double generalisation)
    #
    # The sub-panels are re-aggregations of the `all` run's per-perturbation
    # results, not separate scoring runs. See occam.vcell.panel.run.
    # pds_cosine ranks each effect against the others IN ITS OWN PANEL and
    # expr_mse_unbiased_capped_norm is a single ratio over the panel, so a subset cannot be
    # read off the full-panel result. Each panel therefore also needs its own baseline and
    # replicate anchor (see ensure_panel_refs).
    panel_panels: str = "all,seen,unseen"
    # Sub-panels below this many perturbations are skipped outright. Kept low deliberately:
    # pds is a rank within its panel, so a small panel gives an unstable scale rather than a
    # noisier version of the same number -- but dropping a panel silently is worse than
    # reporting it with a warning, so only degenerate panels are skipped.
    panel_min_perts: int = 10
    # Sub-panels smaller than this are scored but flagged as low-confidence in the log.
    panel_small_warn: int = 30
    # Pull every H1-panel perturbation into the train split. Off by default: it maximises
    # the 'seen' panel but empties the 'unseen' one (whole-genome would leave just the 17
    # perturbations absent from every training file), so the seen/unseen decomposition
    # stops working. Not leakage either way — H1 cells are never trained on.
    panel_force_perts_in_train: bool = False
    # Add H1's MODEL-VISIBLE controls to training as identity samples (control in -> same
    # control out) under the null perturbation token. This is the ONLY way H1 enters
    # training; no H1 perturbation is ever a target. Mirrors the real task, where the unseen
    # cell line's non-targeting controls are exactly what you are handed.
    panel_train_controls: bool = True
    panel_train_control_n: int = 0        # 0 = the whole model-visible pool (18,400)
    # Validation drawn from H1 perturbations (there is no in-domain val split under
    # all_perts_train). 0 disables H1 validation.
    # 0.0 by default. The research tree defaulted this to 0.15 AND left the
    # validation perturbations inside the scored panel, which biases
    # early stopping on ~15% of the headline. If set > 0 here, the
    # complement is scored instead.
    panel_val_frac: float = 0.0
    panel_val_cells_per_pert: int = 32
    # Defaults to control_identity_mode so the exact/paired ablation covers H1 too.
    panel_control_identity_mode: Optional[str] = None
    # Control-fit per-gene quantile calibration, the decoder-stage default (§12.19/12.22).
    # 'both' runs calibrated AND uncalibrated so the contribution is measured, not assumed.
    panel_calibrate: str = "both"        # none | quantile | both
    panel_calib_max_cells: int = 20000
    # cell-eval (v1, log1p space, injctrl convention) -- the metric family used all through §12.
    panel_celleval: bool = True
    panel_celleval_timeout: int = 43200
    panel_celleval_threads: int = 16
    # Comma-separated delta shrink factors: pred = ctrl + alpha*(pred - ctrl), per cell.
    # "1.0" = no shrink (previous behaviour). Sweeping this is the direct remedy for
    # magnitude inflation: pds_cosine is scale-invariant so it survives a uniform shrink,
    # while nmae and mse -- both error metrics measuring effect SIZE -- collapse with it.
    panel_shrink: str = "1.0"
    # Gamma-corrected variants, comma-separated: gammacal | residctrl | hybrid.
    # Keep calibration's per-gene SHAPE reweighting while removing its uniform SCALE
    # inflation, using a controls-only estimate of the map's noise amplification. Unlike a
    # swept shrink (fitted to the scored panel), gamma is re-derivable from whatever
    # controls a new context provides. See estimate_gamma().
    panel_gamma_variants: str = ""
    panel_gamma_pairs: int = 4000
    # Which weights the end-of-training H1 evaluation scores. "best" reloads
    # best_model.pth (the selected checkpoint); "final" keeps the last-epoch weights,
    # which is what every run before 2026-08-24 did — and it scored a model the run had
    # already rejected in 29 of 30 LR-sweep configs (mean `overall` 0.1244 vs 0.1389).
    panel_eval_checkpoint: str = "best"
    # Zero-signal control arm: emit the H1 control mean for every perturbation. Establishes
    # what each metric reads at literally zero biology, on this exact panel and pipeline.
    # Zero-signal anchors. MUST be non-empty: on this evaluation the
    # ctrlmean arm outscores every model arm on pearson_delta, so a table
    # without it cannot be read. See occam.vcell.panel.arms.
    panel_anchors: str = "ctrlmean,ctrlcell"

    # --- Evaluation ---
    eval_freq: int = 5  # epochs between evaluations
    run_baselines: bool = False

    # --- Wandb ---
    use_wandb: bool = True
    wandb_project: str = "Occam"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # --- v2: Interaction function ---
    # ---- Frozen benchmark definition (data/benchmarks/<id>/) -----------------------
    # When `bench_dir` is set the trainer loads gene_axis.csv, splits.json and
    # exclusions.json from it, pins the gene axis, applies the per-(file, perturbation)
    # training exclusions, and records the artifact hashes in every checkpoint. Everything
    # about the benchmark is then a property of the artifact, not of this config.
    bench_dir: Optional[str] = None
    bench_id: str = ""
    # Asserted against benchmark.json. Also placed in the stage YAML so that editing an
    # artifact changes the stage hash and the orchestrator refuses to reuse the run.
    bench_sha256: str = ""
    # input_and_loss = restrict what the encoder is SHOWN and what the loss counts (the
    # default, and the only setting under which the three contexts are truly comparable).
    # loss_only = keep the full input, mask the loss.
    bench_axis_scope: str = "input_and_loss"
    bench_axis_sha256: str = ""          # filled from the artifact; part of the cache key
    # Cell lines the val/test datasets may draw from. For rhj_fewshot this is ["HEPG2"]:
    # the held-out perturbations are TRAINING data in Jurkat and RPE1, so a val set that
    # also drew from them would be two-thirds a training metric.
    bench_eval_cell_lines: Optional[List[str]] = None
    # The test-perturbation eval dataset is expensive to build (all cells, dense, full
    # width) and is never evaluated during training. Off by default; the test set is scored
    # separately, on purpose.
    bench_build_test_datasets: bool = False

    # ---- NaN guards -----------------------------------------------------------------
    # Both were previously read only via getattr() defaults in pipeline.py, so they worked
    # but were invisible to anyone reading the config. Declared so they can be set and
    # recorded like every other knob.
    # Fraction of an epoch's batches allowed to produce a non-finite loss before the epoch is
    # declared failed. One non-finite forward permanently NaNs BatchNorm running statistics,
    # so this is a real abort criterion, not a warning threshold.
    nan_guard_max_frac: float = 0.01
    # Check every parameter AND buffer for non-finite values this often (in steps).
    weight_check_every: int = 500

    interaction_type: str = "additive"  # additive, film, gated_mlp, cell_aware_film

    # ---- latent-space health diagnostics (OCCAM_STATE_HYBRID.md §8) --------------------
    # Run REGARDLESS of the latent loss, because the arms without one are exactly the ones
    # that can collapse to the identity (`delta_z_ratio` -> 0, which scores at the identity
    # floor and so reads as "weak", not "broken") or drift off the decoder's manifold
    # (`offmanifold_ratio` >> 1, i.e. the score is decoder extrapolation).
    latent_diag_every: int = 50      # batches between checks; 0 disables entirely
    latent_diag_rows: int = 256      # rows subsampled for the manifold probe
    latent_diag_manifold: bool = True  # False = norms only, skip the extra dec+enc pass
    film_scale: float = 2.0  # FiLM / CellAwareFiLM scale parameter
    film_hidden_dim: int = 0  # Deep FiLM hidden dim (0 = linear FiLM, >0 = 2-layer MLP)
    film_activation: str = "none"  # Activation for Deep FiLM hidden layer
    film_mode: str = "full"  # full, additive_only, multiplicative_only
    # Subtract the interaction evaluated at the MEAN graph embedding, so a component shared
    # across perturbations is unrepresentable by construction rather than penalised after the
    # fact. Perturbed rows only, so control rows and the control-identity loss are unchanged.
    # `ebar` is the mean over the whole graph node table, never a batch statistic, so
    # inference is batch-independent. See VirtualCellModel._interact.
    # ⚠️ Requires `data_perts_only: false` — restricting the node table changes the reference.
    film_mean_free: bool = False
    gated_mlp_hidden_dim: Optional[int] = None  # GatedMLP hidden dim (defaults to hidden_size)
    gated_mlp_num_layers: int = 2  # GatedMLP hidden layers

    # --- v2: Latent loss ---
    latent_loss_type: str = "none"  # none, delta_z, z_hat, both, projected
    latent_loss_weight: float = 1.0  # weight of latent loss (ignored when type=none)
    latent_loss_alpha: float = 0.5  # balance for "both" mode (alpha*delta_z + (1-alpha)*z_hat)
    latent_proj_dim: int = 256  # projection head output dim for "projected" mode

    # --- v2: Delta regularization ---
    delta_reg_type: str = "none"  # none, l2, norm_ratio
    delta_reg_lambda: float = 0.01  # regularization strength
    delta_reg_tau: float = 0.3  # threshold for norm_ratio regularization

    # --- v2: Expression loss ---
    expression_loss_mode: str = "full"  # full, importance_weighted
    importance_scale: float = 1.0  # controls strength of importance weighting (1.0=linear, >1=amplified, <1=flattened)

    # --- v2: GPL ranking loss ---
    gpl_weight: float = 0.0  # weight for GPL ranking loss (0 = disabled)

    # --- v2: Pairwise ranking loss ---
    ranking_loss_weight: float = 0.0  # weight for pairwise margin ranking loss (0 = disabled)
    ranking_n_pairs: int = 200  # number of gene pairs sampled per sample
    ranking_margin: float = 0.0  # margin for margin_ranking_loss

    # --- v2: Decoder LR warmup ---
    decoder_lr_warmup_epochs: int = 0  # 0 = disabled, >0 = linear ramp from 0

    # --- Random initialization ---
    random_encoder: bool = False
    random_decoder: bool = False

    # --- System ---
    seed: int = 42
    num_workers: int = 0
    output_dir: str = "./virtual_cell_results"
