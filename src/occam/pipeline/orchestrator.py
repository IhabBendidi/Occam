"""Run the three stages end to end from one config.

Chaining happens through **checkpoint files on disk**, never in memory. That is
a correctness requirement, not a style choice: the research tree's "tandem"
mode handed the live model from pretraining straight into finetuning, and the
two paths differ at step 0 (standalone loads non-strictly, dropping the NB
dispersion parameter and re-initialising the loss). The released weights came
from the standalone path, so that is the only path here.

Stage skipping is conservative. ``--resume auto`` reuses a stage only when it
completed, its per-stage config hash is unchanged, and its artifacts are still
on disk. A changed hash is an error with a field-level diff, not a silent
reuse.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from occam.config.loader import (
    ARCH_AFFECTING_FIELDS, ConfigError, OccamConfig, resolve_artifact_ref,
)
from occam.pipeline.ledger import Ledger, StageStatus

__all__ = ["Orchestrator", "StageSkipped", "StageFailed"]


class StageSkipped(Exception):
    pass


class StageFailed(RuntimeError):
    pass


# Artifact filenames each stage produces, in preference order for `final`.
_PRETRAIN_ARTIFACTS = (
    ("final_model", "final_model.pth"),
    ("best_model", "best_model.pth"),
    ("best_perf_checkpoint", "best_perf_checkpoint.pth"),
)
_FINETUNE_ARTIFACTS = (
    ("final_model", "finetuned_final_model.pth"),
    ("best_model", "finetuned_best_model.pth"),
    ("best_perf_checkpoint", "finetuned_best_perf_checkpoint.pth"),
)
_DECODER_ARTIFACTS = (
    ("best_decoder", "best_decoder.pth"),
    ("final_decoder", "final_decoder.pth"),
)


class Orchestrator:
    def __init__(self, config: OccamConfig, dry_run: bool = False):
        self.cfg = config
        self.dry_run = dry_run
        self.ledger = Ledger.load(config.run_dir)

    # -- planning --------------------------------------------------------
    def plan(self) -> List[Tuple[str, str]]:
        """[(stage, action)] where action is 'run', 'reuse' or 'blocked'."""
        resume = (self.cfg.run.get("resume") or "auto").lower()
        if resume not in ("auto", "never", "force"):
            raise ConfigError(f"run.resume must be auto|never|force, got {resume!r}")

        out = []
        for stage in self.cfg.selected_stages:
            h = self.cfg.stage_hash(stage)
            parent = self._resolved_parent(stage)
            if resume == "force":
                out.append((stage, "run"))
                continue
            if self.ledger.is_reusable(stage, h, parent):
                out.append((stage, "never" if resume == "never" else "reuse"))
                continue
            mismatch = self.ledger.hash_mismatch(stage, h)
            parent_moved = self.ledger.parent_mismatch(stage, parent)
            status = self.ledger.status(stage)
            if resume == "never" and status != StageStatus.PENDING:
                out.append((stage, "blocked"))
            elif (mismatch or parent_moved) and status == StageStatus.COMPLETED:
                out.append((stage, "blocked"))
            else:
                out.append((stage, "run"))
        return out

    def _resolved_parent(self, stage: str) -> Optional[str]:
        """The stage's resolved ``init_from``, or None if it has/needs none."""
        block = self.cfg.stages.get(stage) or {}
        raw = block.get("init_from")
        if not raw:
            return None
        try:
            resolved = resolve_artifact_ref(raw, self.ledger.data, f"stages.{stage}.init_from")
        except ConfigError:
            return None          # upstream has not run; not reusable regardless
        if resolved and not os.path.isabs(resolved):
            resolved = os.path.join(self.cfg.repo_root, resolved)
        return resolved

    def describe_plan(self) -> str:
        lines = [
            f"config      : {self.cfg.path}",
            f"name        : {self.cfg.name}",
            f"run dir     : {self.cfg.run_dir}",
            f"config hash : {self.cfg.config_hash[:16]}",
            f"resume      : {self.cfg.run.get('resume', 'auto')}",
            f"runner      : {self.cfg.run.get('runner', 'inprocess')}",
            "",
            "stages:",
        ]
        for stage, action in self.plan():
            h = self.cfg.stage_hash(stage)[:8]
            note = ""
            if action == "reuse":
                arts = self.ledger.artifacts(stage)
                note = "  <- reusing " + ", ".join(sorted(arts)) if arts else ""
            elif action == "blocked":
                recorded = self.ledger.hash_mismatch(stage, self.cfg.stage_hash(stage))
                if recorded:
                    note = f"  <- BLOCKED: recorded hash {recorded[:8]} != {h}"
                elif self.ledger.parent_mismatch(stage, self._resolved_parent(stage)):
                    note = "  <- BLOCKED: parent checkpoint moved"
                else:
                    note = f"  <- BLOCKED: status is {self.ledger.status(stage)}"
            lines.append(f"  {stage:<10s} {action:<8s} hash={h}{note}")
        return "\n".join(lines)

    # -- execution -------------------------------------------------------
    def run(self) -> int:
        from occam import _wandb_guard
        from occam.ddp import stage_fence

        _wandb_guard.ensure_no_shadow(self.cfg.repo_root)

        plan = self.plan()
        blocked = [s for s, a in plan if a == "blocked"]
        if blocked:
            self._report_blocked(blocked)
            return 2

        os.makedirs(self.cfg.run_dir, exist_ok=True)
        resolved = self.cfg.write_resolved()
        print(f"[occam] resolved config -> {resolved}")
        self.ledger.set_run_metadata(
            name=self.cfg.name,
            config_path=self.cfg.path,
            config_hash=self.cfg.config_hash,
        )

        if (self.cfg.run.get("resume") or "auto").lower() == "force":
            for stage, action in plan:
                if action == "run":
                    self._archive_stage_dir(stage)
                    self.ledger.reset(stage)

        for stage, action in plan:
            if action in ("reuse", "never"):
                print(f"\n[occam] {stage}: reusing completed stage "
                      f"({', '.join(sorted(self.ledger.artifacts(stage)))})")
                continue
            print(f"\n{'=' * 70}\n[occam] stage: {stage}\n{'=' * 70}")
            try:
                self._run_stage(stage)
            except Exception as exc:  # noqa: BLE001
                self.ledger.fail(stage, traceback.format_exc())
                print(f"[occam] stage {stage} FAILED: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                raise StageFailed(f"stage {stage} failed") from exc
            finally:
                stage_fence(stage)

        print(f"\n{'=' * 70}")
        print("[occam] all stages complete")
        print(self.ledger.summary(self.cfg.selected_stages))
        return 0

    def _report_blocked(self, blocked: List[str]) -> None:
        print("\n[occam] refusing to run: stage(s) blocked\n", file=sys.stderr)
        for stage in blocked:
            h = self.cfg.stage_hash(stage)
            recorded = self.ledger.hash_mismatch(stage, h)
            parent_moved = self.ledger.parent_mismatch(stage, self._resolved_parent(stage))
            e = self.ledger.entry(stage)
            print(f"  {stage}: status={e.get('status')} "
                  f"recorded_hash={(recorded or '-')[:8]} current_hash={h[:8]}",
                  file=sys.stderr)
            if parent_moved:
                print(f"    parent checkpoint moved:\n"
                      f"      recorded: {parent_moved}\n"
                      f"      now:      {self._resolved_parent(stage)}\n"
                      f"    The recorded weights descend from a different checkpoint.\n"
                      f"    Use --resume force to retrain from the new parent.",
                      file=sys.stderr)
            if recorded:
                diff = self._diff_stage_config(stage, e.get("config"))
                if diff:
                    print("    fields that changed since that run:", file=sys.stderr)
                    for line in diff:
                        print(f"      {line}", file=sys.stderr)
                print("\n    This stage completed under different hyperparameters. "
                      "Reusing its\n    weights would misattribute them. Choose one:\n"
                      "      --resume force        re-run it (old dir is archived)\n"
                      "      revert the config     to match the recorded run",
                      file=sys.stderr)

    def _diff_stage_config(self, stage: str, recorded: Optional[Mapping[str, Any]]) -> List[str]:
        if not recorded:
            return []
        current = self.cfg.stages.get(stage) or {}
        out = []

        def walk(a, b, trail=""):
            keys = sorted(set(a or {}) | set(b or {}))
            for k in keys:
                av, bv = (a or {}).get(k, "<absent>"), (b or {}).get(k, "<absent>")
                path = f"{trail}.{k}" if trail else k
                if isinstance(av, dict) or isinstance(bv, dict):
                    walk(av if isinstance(av, dict) else {},
                         bv if isinstance(bv, dict) else {}, path)
                elif av != bv:
                    out.append(f"{path}: recorded {av!r} -> now {bv!r}")

        walk(recorded, current)
        return out[:40]

    def _archive_stage_dir(self, stage: str) -> None:
        d = self.cfg.stage_dir(stage)
        if not os.path.isdir(d) or not os.listdir(d):
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = f"{d}.bak.{stamp}"
        shutil.move(d, dest)
        print(f"[occam] archived previous {stage} dir -> {dest}")

    def _prepare_stage_dir(self, stage: str) -> str:
        """Create the stage dir, refusing to write into a non-empty one.

        This is the guard against the failure that cost the research tree two
        misattributed checkpoints: five array tasks wrote one timestamp-named
        directory and the surviving weights could not be matched to a recipe.
        """
        d = self.cfg.stage_dir(stage)
        if os.path.isdir(d) and os.listdir(d):
            status = self.ledger.status(stage)
            if status in (StageStatus.RUNNING, StageStatus.FAILED):
                print(f"[occam] {stage}: resuming into existing dir (status {status})")
            else:
                raise StageFailed(
                    f"{d} is not empty and stage status is '{status}'.\n"
                    f"  Occam refuses to write into a populated run directory -- "
                    f"that is how\n  the research tree ended up with checkpoints "
                    f"nobody could attribute to a recipe.\n"
                    f"  Use --resume force to archive it and start clean."
                )
        os.makedirs(d, exist_ok=True)
        return d

    def _collect(self, stage_dir: str, spec) -> Dict[str, str]:
        """Find a stage's artifacts, searching the dir root and checkpoints/."""
        found = {}
        for name, filename in spec:
            for candidate in (
                os.path.join(stage_dir, "checkpoints", filename),
                os.path.join(stage_dir, filename),
            ):
                if os.path.isfile(candidate):
                    found[name] = candidate
                    break
            if name not in found:
                # Output managers create a versioned subdirectory; search it.
                for root, _dirs, files in os.walk(stage_dir):
                    if filename in files:
                        found[name] = os.path.join(root, filename)
                        break
        return found

    def _begin(self, stage: str) -> str:
        from occam.config.loader import _git_sha

        self.ledger.begin(
            stage,
            config_hash=self.cfg.stage_hash(stage),
            git_sha=_git_sha(self.cfg.repo_root),
            config_snapshot=self.cfg.stages.get(stage),
        )
        return self._prepare_stage_dir(stage)

    def _run_stage(self, stage: str) -> None:
        if stage == "pretrain":
            return self._run_pretrain()
        if stage == "finetune":
            return self._run_finetune()
        if stage == "decoder":
            return self._run_decoder()
        raise ConfigError(f"no runner for stage {stage!r}")

    # -- stage 1 ---------------------------------------------------------
    def _run_pretrain(self) -> None:
        from occam.mae.stages import run_pretrain

        cfg = self.cfg.build_pretrain_config()
        stage_dir = self._begin("pretrain")
        trainer, info, _train_loader, _val_loader = run_pretrain(
            train_adata_path=self.cfg.data["train"],
            val_adata_path=self.cfg.data.get("val"),
            config=cfg,
            gene_name_col=self.cfg.data.get("train_gene_col"),
            val_gene_name_col=self.cfg.data.get("val_gene_col"),
            save_dir=stage_dir,
            device=self.cfg.device,
            use_output_manager=False,
        )
        artifacts = self._collect(stage_dir, _PRETRAIN_ARTIFACTS)
        if "final_model" not in artifacts:
            raise StageFailed(
                f"pretrain produced no final_model.pth under {stage_dir}; "
                f"found {sorted(artifacts) or 'nothing'}"
            )
        self.ledger.complete("pretrain", artifacts, metrics={
            "best_val_loss": getattr(trainer, "best_val_loss", None),
            "epochs": cfg.epochs,
        })
        print(f"[occam] pretrain artifacts: {json.dumps(artifacts, indent=2)}")

    # -- stage 2 ---------------------------------------------------------
    def _run_finetune(self) -> None:
        from occam.mae.stages import run_finetune

        ft_cfg, meta = self.cfg.build_finetune_config()
        init_from = resolve_artifact_ref(
            ft_cfg.checkpoint, self.ledger.data, "stages.finetune.init_from")
        if not os.path.isfile(init_from):
            raise StageFailed(f"finetune init_from does not exist: {init_from}")
        ft_cfg.checkpoint = init_from

        stage_dir = self._begin("finetune")
        self._write_overlay_audit(stage_dir, ft_cfg, init_from, meta)

        trainer, _info = run_finetune(
            checkpoint_path=init_from,
            train_adata_path=self.cfg.data["train"],
            val_adata_path=self.cfg.data.get("val"),
            ft_config=ft_cfg,
            gene_name_col=self.cfg.data.get("train_gene_col"),
            val_gene_name_col=self.cfg.data.get("val_gene_col"),
            save_dir=stage_dir,
            device=self.cfg.device,
            # Deterministic run dirs, not the global exp_<NNN> registry.
            use_output_manager=False,
        )
        artifacts = self._collect(stage_dir, _FINETUNE_ARTIFACTS)
        if "final_model" not in artifacts:
            raise StageFailed(
                f"finetune produced no finetuned_final_model.pth under {stage_dir}; "
                f"found {sorted(artifacts) or 'nothing'}"
            )
        self.ledger.complete("finetune", artifacts, metrics={
            "best_val_loss": getattr(trainer, "best_val_loss", None),
            "epochs": ft_cfg.epochs,
        }, extra={"init_from": init_from})
        print(f"[occam] finetune artifacts: {json.dumps(artifacts, indent=2)}")

    def _write_overlay_audit(self, stage_dir: str, ft_cfg, init_from: str,
                             meta: Mapping[str, Any]) -> None:
        """Record, per field, what the YAML said vs what the parent supplied.

        Needed because ``FinetuneConfig.resolve()`` overlays onto the PARENT
        CHECKPOINT's config, not onto this config's ``pretrain`` block. Without
        this file the claim that the YAML is the complete record of the run
        would be false for stage 2.
        """
        import torch

        from occam.mae.config import PretrainConfig

        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        parent_raw = ck.get("config")
        if not isinstance(parent_raw, dict):
            parent_raw = dict(getattr(parent_raw, "__dict__", {}) or {})
        parent = PretrainConfig.from_dict(parent_raw)

        yaml_pretrain, _ = self.cfg.stage_attrs("pretrain") \
            if "pretrain" in self.cfg.stages else ({}, {})

        field_map = getattr(type(ft_cfg), "_FIELD_MAP", {})
        rows = {}
        for ft_field, base_field in sorted(field_map.items()):
            ft_value = getattr(ft_cfg, ft_field, None)
            inherited = getattr(parent, base_field, None)
            rows[ft_field] = {
                "maps_to": base_field,
                "yaml_value": ft_value,
                "inherited_from_parent": inherited,
                "effective": inherited if ft_value is None else ft_value,
                "source": "parent_checkpoint" if ft_value is None else "config",
            }

        # Compare the parent's config against this config's own pretrain block.
        #
        # A `null` in the YAML means "unspecified -- use the default", and the
        # checkpoint records the value that default RESOLVED to (e.g.
        # mae_decoder_hidden: null in the config vs 1024 in the checkpoint).
        # Those agree, so an unspecified field cannot conflict. Only a value
        # the config actually states can.
        conflicts, fatal = [], []
        for base_field, yaml_value in sorted(yaml_pretrain.items()):
            if yaml_value is None:
                continue
            inherited = getattr(parent, base_field, None)
            if inherited == yaml_value:
                continue
            entry = {"field": base_field, "config_pretrain": yaml_value,
                     "parent_checkpoint": inherited}
            conflicts.append(entry)
            if base_field in ARCH_AFFECTING_FIELDS:
                fatal.append(entry)

        # The gradual-unmasking source, made explicit rather than implied.
        effective_mask_prob = None
        if getattr(ft_cfg, "unmasking_mode", None) == "gradual":
            effective_mask_prob = getattr(parent, "mask_probability", None)

        audit = {
            "init_from": init_from,
            "parent_run_name": parent_raw.get("wandb_run_name"),
            "overlay": rows,
            "conflicts_with_config_pretrain_block": conflicts,
            "fatal_conflicts": fatal,
            "gradual_unmasking": {
                "mode": getattr(ft_cfg, "unmasking_mode", None),
                "config_mask_probability": getattr(ft_cfg, "mask_probability", None),
                "effective_mask_probability": effective_mask_prob,
                "source": "parent_checkpoint.mask_probability",
                "note": ("Under mode=gradual the ramp starts from the PARENT "
                         "checkpoint's mask_probability; the config value above "
                         "is inert."),
            },
        }
        out_dir = os.path.join(stage_dir, "config")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "overlay_audit.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2, sort_keys=True, default=str)
        print(f"[occam] overlay audit -> {path}")

        if effective_mask_prob is not None:
            zero_at = getattr(ft_cfg, "unmasking_zero_at", 1.0)
            print(f"[occam] finetune unmasking=gradual: mask_prob "
                  f"{effective_mask_prob} -> 0.0 by "
                  f"{zero_at:.0%} of training (source: parent checkpoint, NOT "
                  f"stages.finetune.masking.mask_probability="
                  f"{getattr(ft_cfg, 'mask_probability', None)})")

        for entry in conflicts:
            print(f"[occam] WARNING: '{entry['field']}' differs between this "
                  f"config's pretrain block ({entry['config_pretrain']!r}) and the "
                  f"parent checkpoint ({entry['parent_checkpoint']!r}); the parent "
                  f"wins for inherited fields")
        if fatal:
            raise StageFailed(
                "architecture-affecting fields differ between the config's "
                "pretrain block and the parent checkpoint:\n  "
                + "\n  ".join(
                    f"{e['field']}: config={e['config_pretrain']!r} "
                    f"parent={e['parent_checkpoint']!r}" for e in fatal
                )
                + "\n\nThis config does not describe the checkpoint it is "
                  "finetuning. Fix the config, or point init_from at the right "
                  "checkpoint."
            )

    # -- stage 3 ---------------------------------------------------------
    def _run_decoder(self) -> None:
        from occam.decoder_train.stages import run_decoder

        dec_cfg, meta = self.cfg.build_decoder_config()
        init_from = resolve_artifact_ref(
            dec_cfg.encoder_checkpoint, self.ledger.data, "stages.decoder.init_from")
        if not os.path.isabs(init_from):
            init_from = os.path.join(self.cfg.repo_root, init_from)
        if not os.path.isfile(init_from):
            raise StageFailed(f"decoder init_from does not exist: {init_from}")
        dec_cfg.encoder_checkpoint = init_from

        stage_dir = self._begin("decoder")
        if not getattr(dec_cfg, "gvf_cache_dir", None):
            dec_cfg.gvf_cache_dir = os.path.join(stage_dir, "gvf_cache")

        eval_data = getattr(dec_cfg, "eval_data", None) or None
        trainer = run_decoder(
            encoder_checkpoint=init_from,
            train_data=self.cfg.data["train"],
            output_dir=stage_dir,
            config=dec_cfg,
            val_data=self.cfg.data.get("val"),
            eval_data=eval_data,
            gene_col=getattr(dec_cfg, "train_gene_col", None),
            eval_gene_col=getattr(dec_cfg, "eval_gene_col", None),
            device=self.cfg.device,
        )
        artifacts = self._collect(stage_dir, _DECODER_ARTIFACTS)
        if not artifacts:
            raise StageFailed(f"decoder produced no checkpoints under {stage_dir}")
        self.ledger.complete("decoder", artifacts, metrics={
            "best_val_loss": getattr(trainer, "best_val_loss", None),
            "best_combined_score": getattr(trainer, "best_combined_score", None),
            "epochs": dec_cfg.epochs,
        }, extra={"init_from": init_from})
        print(f"[occam] decoder artifacts: {json.dumps(artifacts, indent=2)}")
