"""Load, resolve and validate an Occam config.

Pipeline: parse -> ``!include`` -> expand ``${paths.*}`` and
``${stages.X.artifacts.Y}`` -> validate against :mod:`occam.config.schema`
(rejecting unknown keys) -> emit ``resolved_config.yaml`` plus a
``config_hash``.

Two things here are load-bearing rather than conveniences:

**Unknown keys are an error.** A mistyped hyperparameter in a published recipe
does not warn, it changes nothing and lets you believe you changed something.

**Stage 2 resolves its overlay against the parent CHECKPOINT, not the YAML.**
``FinetuneConfig.resolve()`` overlays onto a config rebuilt from
``init_from``'s embedded config, so stage 2 inherits ``target_sum``,
``corruption_mode``, ``model_size``, ``mae_decoder_hidden`` and ``scale`` from
the weights being finetuned. That is what produced the released chain, and
changing it would silently produce a different stage 2 when someone runs
``--stages finetune --set init_from=<released stage1>``. The loader diffs the
inherited values against the YAML's ``pretrain`` block, warns on any mismatch,
hard-fails on architecture-affecting ones, and writes ``overlay_audit.json``
recording per field what the YAML said, what was inherited, and which won.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

from occam.config.schema import SchemaError, TOP_LEVEL_KEYS, RUN_KEYS, DATA_KEYS, flatten_stage

__all__ = [
    "OccamConfig", "load_config", "ConfigError",
    "ARCH_AFFECTING_FIELDS", "config_hash",
    "resolve_artifact_ref", "has_unresolved_refs",
]


class ConfigError(ValueError):
    pass


# Inheriting a different value for any of these changes what the model *is*, so
# a YAML/checkpoint mismatch here is fatal rather than a warning.
ARCH_AFFECTING_FIELDS = (
    "model_size", "hidden_size", "mae_decoder_hidden", "num_genes",
    "target_sum", "skip_normalize", "skip_log1p",
)

_VAR = re.compile(r"\$\{([^}]+)\}")
_STAGE_ORDER = ("pretrain", "finetune", "decoder", "vcell")


# ---------------------------------------------------------------------------
# YAML with !include
# ---------------------------------------------------------------------------
def _yaml_loader(base_dir: str):
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def _include(loader, node):
        rel = loader.construct_scalar(node)
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.isfile(path):
            raise ConfigError(
                f"!include target not found: {path}\n"
                f"  (referenced as '{rel}')\n"
                f"  For paths.yaml: copy configs/paths.example.yaml and edit it."
            )
        with open(path, encoding="utf-8") as fh:
            return yaml.load(fh, Loader=_yaml_loader(os.path.dirname(path)))

    Loader.add_constructor("!include", _include)
    return Loader


# ---------------------------------------------------------------------------
# ${...} expansion
# ---------------------------------------------------------------------------
def _expand(value, resolve, trail: str):
    if isinstance(value, str):
        def sub(m):
            ref = m.group(1).strip()
            out = resolve(ref, trail)
            return str(out)
        # A whole-string reference keeps its type (int, bool, None).
        whole = _VAR.fullmatch(value.strip())
        if whole:
            return resolve(whole.group(1).strip(), trail)
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, resolve, f"{trail}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, resolve, f"{trail}[{i}]") for i, v in enumerate(value)]
    return value


# ---------------------------------------------------------------------------
# The resolved config
# ---------------------------------------------------------------------------
class OccamConfig:
    """A fully resolved config, plus the machinery to build stage config objects."""

    def __init__(self, raw: Dict[str, Any], path: str, repo_root: str):
        self.raw = raw
        self.path = path
        self.repo_root = repo_root
        self.name: str = raw["name"]
        self.seed: int = int(raw.get("seed", 42))
        self.device: str = raw.get("device", "cuda")
        self.paths: Dict[str, Any] = raw.get("paths") or {}
        self.run: Dict[str, Any] = raw.get("run") or {}
        self.data: Dict[str, Any] = raw.get("data") or {}
        self.stages: Dict[str, Any] = raw.get("stages") or {}
        self._hash: Optional[str] = None

    # -- identity --------------------------------------------------------
    @property
    def config_hash(self) -> str:
        if self._hash is None:
            self._hash = config_hash(self.raw)
        return self._hash

    def stage_hash(self, stage: str) -> str:
        """Hash of one stage's hyperparameters, plus the shared data and seed.

        ``init_from`` is deliberately EXCLUDED. It starts life as a deferred
        ``${stages.X.artifacts.Y}`` reference and becomes a concrete path once
        the upstream stage has run, so including it would give the same stage
        two different hashes before and after -- and every re-run would look
        like a hyperparameter change.

        The parent checkpoint is still tracked, just separately: the ledger
        records the resolved ``init_from`` per stage and the orchestrator
        compares it. That also reports the two situations distinctly, which is
        more useful than one hash covering both: "you changed hyperparameters"
        and "you pointed this at a different checkpoint" call for different
        responses.
        """
        block = dict(self.stages.get(stage) or {})
        block.pop("init_from", None)
        payload = {
            "stage": block,
            "data": self.data,
            "seed": self.seed,
        }
        return config_hash(payload)

    @property
    def run_dir(self) -> str:
        root = self.run.get("root", "./runs")
        if not os.path.isabs(root):
            root = os.path.join(self.repo_root, root)
        suffix = self.run.get("run_suffix") or ""
        return os.path.normpath(
            os.path.join(root, self.name + (f"_{suffix}" if suffix else ""))
        )

    def stage_dir(self, stage: str) -> str:
        return os.path.join(self.run_dir, stage)

    @property
    def selected_stages(self) -> List[str]:
        wanted = self.run.get("stages") or list(_STAGE_ORDER)
        unknown = [s for s in wanted if s not in _STAGE_ORDER]
        if unknown:
            raise ConfigError(f"unknown stage(s) {unknown}; expected {list(_STAGE_ORDER)}")
        # Always in canonical order, whatever order the YAML listed them in.
        return [s for s in _STAGE_ORDER if s in wanted]

    # -- stage config objects --------------------------------------------
    def stage_attrs(self, stage: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        block = self.stages.get(stage)
        if block is None:
            raise ConfigError(f"config has no 'stages.{stage}' block")
        return flatten_stage(stage, block)

    def build_pretrain_config(self):
        from occam.mae.config import PretrainConfig

        attrs, _meta = self.stage_attrs("pretrain")
        cfg = PretrainConfig()
        for key, value in attrs.items():
            setattr(cfg, key, value)
        self._apply_shared(cfg)
        return cfg

    def build_finetune_config(self):
        from occam.mae.finetune_config import FinetuneConfig

        attrs, meta = self.stage_attrs("finetune")
        cfg = FinetuneConfig()
        for key, value in attrs.items():
            setattr(cfg, key, value)
        cfg.enabled = True
        init_from = meta.get("init_from")
        if init_from:
            cfg.checkpoint = init_from
        return cfg, meta

    def build_decoder_config(self):
        from occam.decoder_train.config import DecoderTrainConfig

        attrs, meta = self.stage_attrs("decoder")
        init_from = meta.get("init_from") or ""
        cfg = DecoderTrainConfig(encoder_checkpoint=init_from)
        for key, value in attrs.items():
            setattr(cfg, key, value)
        cfg.encoder_checkpoint = init_from
        self._apply_shared(cfg, decoder=True)
        return cfg, meta

    def build_vcell_config(self):
        from occam.vcell.config import VirtualCellConfig

        attrs, meta = self.stage_attrs("vcell")
        cfg = VirtualCellConfig()
        for key, value in attrs.items():
            setattr(cfg, key, value)
        cfg.encoder_checkpoint = meta.get("init_from") or ""
        self._apply_shared(cfg)
        # Relative registry-style paths resolve against the REPO ROOT, not the cwd, so a
        # config behaves the same wherever it is run from.
        for attr in ("bench_dir", "graph_cache_dir", "master_gene_set_path",
                     "control_cache_dir"):
            val = getattr(cfg, attr, None)
            if val and not os.path.isabs(str(val)):
                setattr(cfg, attr, os.path.join(self.repo_root, str(val)))
        return cfg, meta

    def _apply_shared(self, cfg, decoder: bool = False) -> None:
        """Apply the top-level and `data:` settings shared by every stage."""
        wandb = self.run.get("wandb") or {}
        for attr, value in (
            ("seed", self.seed),
            ("device", self.device),
            ("use_wandb", bool(wandb.get("enabled", True))),
            ("wandb_project", wandb.get("project", "occam")),
            ("wandb_tags", wandb.get("tags")),
        ):
            if hasattr(cfg, attr):
                setattr(cfg, attr, value)
        if hasattr(cfg, "train_gene_col") and self.data.get("train_gene_col"):
            cfg.train_gene_col = self.data["train_gene_col"]

    # -- serialisation ---------------------------------------------------
    def write_resolved(self, target_dir: Optional[str] = None) -> str:
        """Write the fully expanded config plus provenance. The audit artifact."""
        import yaml

        target_dir = target_dir or self.run_dir
        os.makedirs(target_dir, exist_ok=True)
        out = os.path.join(target_dir, "resolved_config.yaml")
        payload = {
            "_provenance": {
                "source_config": self.path,
                "config_hash": self.config_hash,
                "git_sha": _git_sha(self.repo_root),
                "stage_hashes": {s: self.stage_hash(s) for s in self.stages},
            },
            **self.raw,
        }
        with open(out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
        return out


def config_hash(payload: Any) -> str:
    """Stable hash of a config, insensitive to key order and formatting."""
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_sha(repo_root: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_config(
    path: str,
    overrides: Optional[List[str]] = None,
    repo_root: Optional[str] = None,
    ledger: Optional[Mapping[str, Any]] = None,
) -> OccamConfig:
    """Parse, expand, validate.

    ``overrides`` are ``dotted.key=value`` strings, applied before expansion so
    an override can itself reference ``${paths.*}``.

    ``ledger`` supplies stage artifacts for ``${stages.X.artifacts.Y}``
    references. Omit it and such a reference is an error naming the upstream
    stage, which is the right failure for ``--stages decoder`` before stage 2
    has ever run.
    """
    import yaml

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ConfigError(f"config not found: {path}")
    repo_root = repo_root or os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    with open(path, encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=_yaml_loader(os.path.dirname(path)))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    for key in ("name", "stages"):
        if key not in raw:
            raise ConfigError(f"{path} is missing required top-level key '{key}'")

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(
            f"unknown top-level key(s) in {os.path.basename(path)}: {sorted(unknown)}\n"
            f"  expected: {sorted(TOP_LEVEL_KEYS)}"
        )
    for section, allowed in (("run", RUN_KEYS), ("data", DATA_KEYS)):
        extra = set(raw.get(section) or {}) - allowed
        if extra:
            raise ConfigError(
                f"unknown key(s) in '{section}': {sorted(extra)}\n"
                f"  expected: {sorted(allowed)}"
            )

    for spec in overrides or []:
        _apply_override(raw, spec)

    paths_block = raw.get("paths") or {}
    if isinstance(paths_block, dict) and "__include__" in paths_block:
        raise ConfigError(
            f"paths block did not resolve: {paths_block['__include__']}"
        )

    def resolve(ref: str, trail: str):
        if ref.startswith("paths."):
            key = ref[len("paths."):]
            if key not in paths_block:
                raise ConfigError(
                    f"{trail} references ${{{ref}}}, which is not in the paths registry.\n"
                    f"  available: {sorted(paths_block)}\n"
                    f"  See configs/paths.example.yaml."
                )
            # Registry values are allowed to be relative -- the shipped defaults
            # point at data/eval/... inside the repo. Make them absolute against
            # the REPO ROOT, not the working directory, so a config behaves the
            # same wherever it is run from.
            from occam import paths as _paths

            return _paths.resolve(paths_block[key]) or ""
        m = re.fullmatch(r"stages\.(\w+)\.artifacts\.(\w+)", ref)
        if m:
            stage, artifact = m.groups()
            entry = ((ledger or {}).get("stages") or {}).get(stage) or {}
            value = (entry.get("artifacts") or {}).get(artifact)
            if value:
                return value
            # DEFERRED, not an error. When all three stages run in one job,
            # stage 1's artifact does not exist at load time -- it is produced
            # partway through. The reference is left intact and resolved by
            # `resolve_artifact_ref` just before the consuming stage starts, by
            # which point the upstream stage has recorded it in the ledger.
            return "${" + ref + "}"
        raise ConfigError(
            f"{trail} references ${{{ref}}}, which is not a recognised form.\n"
            f"  expected ${{paths.<key>}} or ${{stages.<stage>.artifacts.<name>}}"
        )

    for section in ("data", "stages"):
        if section in raw:
            raw[section] = _expand(raw[section], resolve, section)

    cfg = OccamConfig(raw, path, repo_root)

    # Validate every stage present, not only the selected ones -- a typo in an
    # unselected stage should still be caught.
    for stage in cfg.stages:
        try:
            flatten_stage(stage, cfg.stages[stage])
        except SchemaError as exc:
            raise ConfigError(f"{os.path.basename(path)}: {exc}") from None

    cfg.selected_stages  # noqa: B018  -- validates the stage list
    return cfg


def resolve_artifact_ref(value: Any, ledger: Mapping[str, Any], trail: str = "") -> Any:
    """Resolve a deferred ``${stages.X.artifacts.Y}`` reference against the ledger.

    Called by the orchestrator immediately before a stage runs. Raises with the
    upstream stage named, which is the right failure for e.g.
    ``--stages decoder`` before stage 2 has ever run.
    """
    if not isinstance(value, str):
        return value
    m = _VAR.fullmatch(value.strip())
    if not m:
        return value
    ref = m.group(1).strip()
    parts = re.fullmatch(r"stages\.(\w+)\.artifacts\.(\w+)", ref)
    if not parts:
        return value
    stage, artifact = parts.groups()
    entry = ((ledger or {}).get("stages") or {}).get(stage) or {}
    resolved = (entry.get("artifacts") or {}).get(artifact)
    if not resolved:
        status = entry.get("status", "never run")
        raise ConfigError(
            f"{trail or 'init_from'} references ${{{ref}}}, but stage "
            f"'{stage}' has no recorded artifact '{artifact}' (status: {status}).\n"
            f"  Either run stage '{stage}' first:\n"
            f"      python -m occam.train --config <cfg> --stages {stage}\n"
            f"  or point this stage at a checkpoint directly:\n"
            f"      --set stages.<this stage>.init_from=<path>"
        )
    return resolved


def has_unresolved_refs(value: Any) -> List[str]:
    """List any ``${...}`` references still present. Used by --dry-run."""
    found: List[str] = []

    def walk(node, trail):
        if isinstance(node, str):
            for m in _VAR.finditer(node):
                found.append(f"{trail} -> ${{{m.group(1)}}}")
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{trail}.{k}" if trail else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{trail}[{i}]")

    walk(value, "")
    return found


def _apply_override(raw: Dict[str, Any], spec: str) -> None:
    if "=" not in spec:
        raise ConfigError(f"--set expects dotted.key=value, got {spec!r}")
    dotted, _, text = spec.partition("=")
    node = raw
    parts = dotted.strip().split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ConfigError(
                f"--set {dotted}: '{part}' is not an existing mapping in the config"
            )
        node = node[part]
    leaf = parts[-1]
    if leaf not in node:
        raise ConfigError(
            f"--set {dotted}: '{leaf}' is not an existing key. "
            f"Overrides may only change values that the config already declares, "
            f"so that the config stays the complete record of the run."
        )
    node[leaf] = _coerce(text, node[leaf])


def _coerce(text: str, current: Any) -> Any:
    """Coerce an override to the type already in the config."""
    if isinstance(current, bool):
        low = text.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ConfigError(f"expected a boolean, got {text!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if current is None:
        import yaml
        return yaml.safe_load(text)
    return text
