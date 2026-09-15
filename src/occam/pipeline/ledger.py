"""The run ledger: what each stage produced, and whether it can be reused.

``runs/<name>/ledger.json`` is the record that lets stage 2 find stage 1's
checkpoint, and lets ``--resume auto`` decide what to skip.

The design point is the **per-stage config hash**. A stage is skipped only if it
completed *and* its hash is unchanged. If the hash moved, the orchestrator
refuses and prints the field-level diff rather than silently reusing weights
produced by different hyperparameters. That is the failure this whole repo
exists to prevent: the research tree lost track of which hyperparameters
produced which weights, twice, because provenance lived in mutable sidecar
files instead of in the artifacts.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["Ledger", "StageStatus"]


class StageStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    """Read/write ``runs/<name>/ledger.json``, one entry per stage."""

    SCHEMA = 1

    def __init__(self, path: str, data: Optional[Dict[str, Any]] = None):
        self.path = path
        self.data: Dict[str, Any] = data or {
            "schema": self.SCHEMA,
            "created": _now(),
            "stages": {},
        }

    # -- io --------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: str) -> "Ledger":
        path = os.path.join(run_dir, "ledger.json")
        if not os.path.isfile(path):
            return cls(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(
                f"{path} has schema {data.get('schema')}, expected {cls.SCHEMA}"
            )
        return cls(path, data)

    def save(self) -> None:
        """Atomic write -- a crash mid-save must not corrupt the ledger."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data["updated"] = _now()
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2, sort_keys=True, default=str)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # -- queries ---------------------------------------------------------
    def entry(self, stage: str) -> Dict[str, Any]:
        return self.data.setdefault("stages", {}).setdefault(
            stage, {"status": StageStatus.PENDING}
        )

    def status(self, stage: str) -> str:
        return self.entry(stage).get("status", StageStatus.PENDING)

    def artifacts(self, stage: str) -> Dict[str, str]:
        return dict(self.entry(stage).get("artifacts") or {})

    def artifact(self, stage: str, name: str) -> Optional[str]:
        return self.artifacts(stage).get(name)

    def is_reusable(self, stage: str, config_hash: str,
                    parent: Optional[str] = None) -> bool:
        """Completed, same hyperparameters, same parent, artifacts still on disk.

        ``parent`` is the resolved ``init_from``. Pass ``None`` to skip that
        check (stage 1 has no parent, and an unresolvable parent means the
        upstream stage has not run, in which case this stage is not reusable
        for other reasons anyway).
        """
        e = self.entry(stage)
        if e.get("status") != StageStatus.COMPLETED:
            return False
        if e.get("config_hash") != config_hash:
            return False
        if parent is not None and e.get("init_from") not in (None, parent):
            return False
        for path in (e.get("artifacts") or {}).values():
            if not os.path.exists(path):
                return False
        return True

    def parent_mismatch(self, stage: str, parent: Optional[str]) -> Optional[str]:
        """The recorded parent, if it differs from ``parent``. Else ``None``."""
        if parent is None:
            return None
        recorded = self.entry(stage).get("init_from")
        return recorded if recorded and recorded != parent else None

    def hash_mismatch(self, stage: str, config_hash: str) -> Optional[str]:
        """The recorded hash, if it differs from ``config_hash``. Else ``None``."""
        e = self.entry(stage)
        recorded = e.get("config_hash")
        if recorded and recorded != config_hash:
            return recorded
        return None

    def missing_artifacts(self, stage: str) -> List[str]:
        return [name for name, path in self.artifacts(stage).items()
                if not os.path.exists(path)]

    # -- mutations -------------------------------------------------------
    def begin(self, stage: str, config_hash: str, git_sha: Optional[str] = None,
              config_snapshot: Optional[Mapping[str, Any]] = None) -> None:
        e = self.entry(stage)
        e.update({
            "status": StageStatus.RUNNING,
            "started": _now(),
            "finished": None,
            "config_hash": config_hash,
            "git_sha": git_sha,
        })
        if config_snapshot is not None:
            e["config"] = dict(config_snapshot)
        e.pop("error", None)
        self.save()

    def complete(self, stage: str, artifacts: Mapping[str, str],
                 metrics: Optional[Mapping[str, Any]] = None,
                 extra: Optional[Mapping[str, Any]] = None) -> None:
        e = self.entry(stage)
        e.update({
            "status": StageStatus.COMPLETED,
            "finished": _now(),
            "artifacts": {k: v for k, v in artifacts.items() if v},
            "metrics": dict(metrics or {}),
        })
        if extra:
            e.update(extra)
        self.save()

    def fail(self, stage: str, error: str) -> None:
        e = self.entry(stage)
        e.update({
            "status": StageStatus.FAILED,
            "finished": _now(),
            "error": error[:4000],
        })
        self.save()

    def reset(self, stage: str) -> None:
        self.data.setdefault("stages", {})[stage] = {"status": StageStatus.PENDING}
        self.save()

    def set_run_metadata(self, **kwargs) -> None:
        self.data.update({k: v for k, v in kwargs.items() if v is not None})
        self.save()

    # -- reporting -------------------------------------------------------
    def summary(self, stages: List[str]) -> str:
        rows = []
        for s in stages:
            e = self.entry(s)
            status = e.get("status", StageStatus.PENDING)
            arts = len(e.get("artifacts") or {})
            h = (e.get("config_hash") or "")[:8]
            rows.append(f"  {s:<10s} {status:<10s} artifacts={arts}  hash={h or '-'}")
        return "\n".join(rows)
