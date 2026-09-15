"""External-decoder training configuration (stage 3).

Extracted from ``decoder_training_pipeline.py:276-434``, where it was
the research tree's decoder config class. Its encoder-checkpoint field was
renamed to ``encoder_checkpoint``; ``occam._compat`` reads the old name off
disk so existing checkpoints and run configs still load.

Two defaults are traps worth knowing about:

* ``decoder_num_blocks`` defaults to 4, but the released decoder is **5**.
  Loading the release without setting it fails on a state-dict mismatch.
* ``decoder_output_activation`` defaults to ``rectified_sigmoid``, which is
  correct for the release but wrong for anything trained with softplus -- and
  the wrong choice loads cleanly while emitting ~99% exact zeros. Occam records
  the activation in every checkpoint it writes; for older ones,
  ``scripts/verify_checkpoint.py`` determines it empirically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# The augmentation defaults, verbatim from the research tree. These are the
# values the released decoder trained with, so they are reproduced here rather
# than re-derived; configs/occam_small.yaml pins them explicitly too.
_DEFAULT_AUG_HPARAMS = {
    'initial_gene_jitter_strength': 0.2,
    'final_gene_jitter_strength': 0.05,
    'gene_jitter_prob': 0.6,
    'initial_gene_mask_rate': 0.05,
    'final_gene_mask_rate': 0.0,
    'gene_mask_prob': 0.5,
    'max_effective_mask_rate': 0.25,
    'initial_mask_activation_threshold': 0.5,
    'final_mask_activation_threshold': 0.0,
    'initial_sensitivity_p': 2.0,
    'final_sensitivity_p': 0.25,
    'sensitivity_p_phase1_epochs_ratio': 0.25,
    'initial_sparsity_modulator_strength': 0.75,
    'final_sparsity_modulator_strength': 0.1,
    'initial_cov_influence_weight': 1.5,
    'final_cov_influence_weight': 0.5,
    'gvf_lower_p': 0.1,
    'gvf_upper_p': 0.95,
}

__all__ = ["DecoderTrainConfig", "_DEFAULT_AUG_HPARAMS"]



@dataclass
class DecoderTrainConfig:
    """Configuration for Occam Decoder training."""

    # Encoder
    encoder_checkpoint: str = ""
    freeze_encoder: bool = True
    encoder_lr: float = 1e-5
    # Skip transforms when data is already library-normalized / log1p-transformed.
    # Auto-inherited from the encoder checkpoint; these force-enable (OR).
    skip_normalize: bool = False
    skip_log1p: bool = False
    # End-of-training ARC cell-eval on the decoder's per-cell reconstruction of eval files.
    cell_eval_enabled: bool = True
    celleval_pert_col: str = "gene"
    celleval_control_label: str = "non-targeting"
    # Restrict cell-eval to encoder-vocabulary genes actually present in the eval file.
    # Zero-filled genes are structurally 0 in the target but non-zero in the prediction,
    # so keeping them scores hallucination on genes the file never measured.
    celleval_present_genes_only: bool = True
    # cell-eval subprocess budget. Wilcoxon DE cost scales as perts x genes x cells, so
    # files with thousands of perturbations need far more than the old hardcoded 1 h:
    # measured at 16 threads, TREK (90 perts) took 27 min/mode while nadig (2,393 perts)
    # projects to ~14 h/mode. Hopper nodes have 224 CPUs — raise threads before timeout.
    # Sample from the fitted NB/ZINB predictive instead of emitting its mean (nb/zinb
    # only; ignored otherwise). The decoder's deterministic mean has far less per-cell
    # variance than real cells, which is what inflates predicted DE counts ~25x on nadig.
    # Zero-handling experiment. Predictions carry small non-zero floats where the target
    # is exactly 0; Wilcoxon reads that as signal, inflating predicted DE counts.
    #   none      leave predictions as-is
    #   oracle    force preds to 0 wherever the TARGET is 0 (upper bound; uses ground truth)
    #   threshold zero |pred| <= t, t auto-chosen to match the target's zero fraction
    # Per-gene quantile calibration of predictions onto the real marginal, fit on
    # CONTROL cells only. The diagnosed fix for the injctrl DE inflation.
    # Final activation of the decoder output head. rectified_sigmoid (legacy) is exactly
    # 0 for pre-activation <= 0 WITH zero gradient there, which traps rarely-detected
    # genes at a hard constant (80% of TREK's 44k genes). softplus is strictly positive
    # with a live gradient everywhere.
    decoder_output_activation: str = "rectified_sigmoid"
    # Score the reconstruction eval only on genes present in each eval file, excluding
    # structural zero padding. Off by default so historical numbers stay comparable.
    eval_present_genes_only: bool = False
    # Zero out predictions below a quantisation floor, expressed in COUNT space
    # (applied as expm1(pred) < thr). 0 = off. 'percell' uses each cell's own smallest
    # nonzero TARGET value, which is that cell's one-raw-count floor.
    eval_zero_threshold: float = 0.0
    eval_zero_percell: bool = False
    # Per-gene quantile calibration inside the reconstruction eval. Data-driven, so it
    # needs no per-dataset threshold. Fit on a HELD-OUT cell split and scored on the
    # complement, otherwise transforming predictions with target-derived quantiles and
    # then scoring against that same target is circular.
    eval_calibrate: str = "none"        # none | quantile
    eval_calib_fit_frac: float = 0.5
    # Which cells fit the eval calibration. 'controls' is the deployable choice: at
    # inference you always have non-targeting controls and never the perturbed targets.
    # Scoring is then on PERTURBED cells only, disjoint from the fit set.
    eval_calib_on: str = "controls"     # controls | holdout
    eval_calib_min_controls: int = 200
    # Library-normalise PREDICTIONS around the calibration step, to probe the ordering
    # question. none | pre (before calib) | post (after calib) | both.
    # Note per-cell Spearman is mathematically invariant to any per-cell rescale, so
    # this axis should move Pearson/MAE only -- that is a built-in sanity check.
    eval_libnorm: str = "none"
    # Fit the calibration reference on log1p-only (un-normalised) controls instead of
    # the normalised+log1p targets. Deliberately violates the source/reference
    # space-consistency rule; included as a negative control.
    eval_calib_ref_raw: bool = False
    eval_calib_max_fit_cells: int = 20000
    celleval_calibrate: str = "none"   # none | quantile
    celleval_zero_mode: str = "none"
    celleval_zero_thresh: float = -1.0   # <0 = auto-match target zero fraction
    celleval_modes: str = "injctrl,reconctrl"
    celleval_sample_predictive: bool = False
    celleval_sample_seed: int = 42
    celleval_num_threads: int = 16
    # cell-eval's --batch-size: (perturbation, gene) pairs per DE task. None =
    # cell-eval's own default of 100, which on a panel with thousands of
    # perturbations becomes hundreds of thousands of pool tasks and trips a
    # shared-memory race in its DE backend. See metrics.run_cell_eval.
    celleval_de_batch_size: Optional[int] = None
    # Directory holding reusable real-side differential expression, keyed by the
    # target file's sha256. The ground-truth DE depends only on the target, so
    # every arm scored against the same target can share one -- about 1.5 h of a
    # 2.5 h run on nadig_2024_jurkat. None disables reuse.
    celleval_de_real_cache: Optional[str] = None
    celleval_timeout: int = 3600
    # Disable AMP (mixed precision) — train/eval in fp32.
    no_amp: bool = False

    # Decoder architecture
    # decoder_size is a preset that fills hidden_dim/num_blocks (see
    # external_decoder.DECODER_SIZE_CONFIGS). It is independent of the encoder's
    # model_size. None = use whatever hidden_dim/num_blocks are set below.
    decoder_size: Optional[str] = None
    decoder_hidden_dim: int = 2048
    decoder_num_blocks: int = 4  # default 4: blocks sweep showed n=4 == n=5 (0.436), 5th redundant
    decoder_weight_tie: bool = False  # recurrent: 1 shared block applied num_blocks times (FAILS n>=2)
    decoder_bottleneck_k: int = 1  # inverted bottleneck in block MLP: H->H/k->H (k=1 = full, unchanged)
    decoder_monarch_blocks: int = 0  # Stage B3 Monarch/BTT: square H->H Linear -> MonarchLinear(b factors); 0/1=off
    decoder_monarch_mup: bool = False  # B3 muP/spectral init+multipliers for MonarchLinear
    decoder_monarch_warmstart: str = ""  # B3/B6: path to dense unified ckpt -> analytic dense->structured warm-start
    decoder_kron_rank: int = 0  # Stage B6 Kronecker: square H->H Linear -> sum of kron_rank A⊗B terms; 0=off
    decoder_glu: bool = False  # Stage B5: SwiGLU-gated residual blocks (mutually exclusive with monarch)
    decoder_glu_ratio: float = 0.667  # GLU inner width = ratio*H; 2/3 = param-neutral vs dense block
    decoder_sparsity_24: bool = False  # Stage B4: 2:4 SR-STE sparsity on block/in/out linears (composes with B3)
    decoder_sparsity_24_decay: float = 2e-4  # SR-STE decay on pruned weights
    decoder_dropout: float = 0.3
    decoder_se_reduction: int = 16

    # Training
    lr: float = 1e-6
    weight_decay: float = 0.01
    batch_size: int = 1024
    epochs: int = 100
    smooth_l1_beta: float = 2.0
    loss_type: str = "smooth_l1"  # smooth_l1, mse, log_cosh, poisson
    # nb/zinb dispersion-param controls (Stage C). dispersion_lr=None -> use global lr.
    dispersion_lr: float = None   # LR for nb/zinb log_theta/logit_pi
    disp_theta_init: float = 0.0  # init log_theta (theta = exp(.); 0.0 -> theta=1) for nb+zinb
    zinb_pi_init: float = -2.0    # init logit_pi (pi = sigmoid(.); -2.0 -> 0.12) for zinb
    eta_min: float = 1e-4
    gradient_clip: float = 0.5
    weight_init: str = "default"
    log_interval: int = 3

    # Data
    data_mode: str = "optim"
    optim_chunk_size: int = 8192
    optim_buffer_chunks: int = 4
    val_split: float = 0.1
    gene_presence_mask: bool = False
    presence_shift: float = 1.0
    min_gene_overlap_pct: Optional[float] = None

    # Augmentation
    no_augmentation: bool = False
    aug_hparams: dict = field(default_factory=lambda: dict(_DEFAULT_AUG_HPARAMS))
    gvf_cache_dir: Optional[str] = None  # shared GVF cache dir (defaults to output_dir/../gvf_cache)

    # Adversarial / Perceptual
    adv_loss: bool = False
    perc_loss: bool = False
    feature_extractor_checkpoint: Optional[str] = None
    adv_mode: str = "frozen"
    perc_mode: str = "frozen"
    lambda_adv: float = 0.1
    lambda_perc: float = 0.1
    disc_lr: float = 2e-4

    # Evaluation
    decoder_eval_freq: int = 10
    bmdb_eval_freq: int = 3
    run_encoder_benchmark: bool = False

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "Occam"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # System
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---- Weight Initialization ----


    # -- construction from persisted configs -----------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecoderTrainConfig":
        """Rebuild from a config dict read off a checkpoint or run_config.json.

        Renames legacy keys and fills in fields the released config predates.
        See ``occam._compat``.
        """
        from occam._compat import apply_legacy_defaults, migrate_config_dict

        merged = apply_legacy_defaults(migrate_config_dict(data))
        obj = cls(encoder_checkpoint=merged.get("encoder_checkpoint", ""))
        for key, value in merged.items():
            setattr(obj, key, value)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}
