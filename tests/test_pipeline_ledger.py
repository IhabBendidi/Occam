"""Ledger and stage-planning logic.

The property that matters: a stage is reused only when it completed under the
same hyperparameters and its artifacts still exist. Everything else -- a
changed hash, a deleted checkpoint, a failed run -- must not silently reuse
weights. Misattributed weights are the failure this repo was built to fix.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from occam.pipeline.ledger import Ledger, StageStatus


class _Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="occam_ledger_")
        return self.d

    def __exit__(self, *exc):
        shutil.rmtree(self.d, ignore_errors=True)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("x")
    return path


class TestLedgerRoundTrip:
    def test_absent_ledger_starts_empty(self):
        with _Tmp() as d:
            led = Ledger.load(d)
            assert led.status("pretrain") == StageStatus.PENDING
            assert led.artifacts("pretrain") == {}

    def test_completed_stage_persists(self):
        with _Tmp() as d:
            art = _touch(os.path.join(d, "pretrain", "final_model.pth"))
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.complete("pretrain", {"final_model": art}, metrics={"best_val_loss": 0.5})

            reread = Ledger.load(d)
            assert reread.status("pretrain") == StageStatus.COMPLETED
            assert reread.artifact("pretrain", "final_model") == art
            assert reread.entry("pretrain")["metrics"]["best_val_loss"] == 0.5

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        with _Tmp() as d:
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.save()
            leftovers = [f for f in os.listdir(d) if f.endswith(".tmp")]
            assert not leftovers, f"temp files left behind: {leftovers}"
            with open(os.path.join(d, "ledger.json")) as fh:
                assert json.load(fh)["schema"] == Ledger.SCHEMA


class TestReuseRules:
    def test_reusable_when_hash_matches_and_artifacts_exist(self):
        with _Tmp() as d:
            art = _touch(os.path.join(d, "pretrain", "final_model.pth"))
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.complete("pretrain", {"final_model": art})
            assert led.is_reusable("pretrain", "h1")

    def test_not_reusable_when_hash_changed(self):
        with _Tmp() as d:
            art = _touch(os.path.join(d, "pretrain", "final_model.pth"))
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.complete("pretrain", {"final_model": art})
            assert not led.is_reusable("pretrain", "h2")
            assert led.hash_mismatch("pretrain", "h2") == "h1"

    def test_not_reusable_when_an_artifact_was_deleted(self):
        """A checkpoint deleted to save disk must not read as a usable stage."""
        with _Tmp() as d:
            art = _touch(os.path.join(d, "pretrain", "final_model.pth"))
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.complete("pretrain", {"final_model": art})
            os.unlink(art)
            assert not led.is_reusable("pretrain", "h1")
            assert led.missing_artifacts("pretrain") == ["final_model"]

    def test_not_reusable_when_failed(self):
        with _Tmp() as d:
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.fail("pretrain", "boom")
            assert not led.is_reusable("pretrain", "h1")
            assert led.status("pretrain") == StageStatus.FAILED
            assert "boom" in led.entry("pretrain")["error"]

    def test_running_stage_is_not_reusable(self):
        with _Tmp() as d:
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            assert led.status("pretrain") == StageStatus.RUNNING
            assert not led.is_reusable("pretrain", "h1")

    def test_reset_clears_the_entry(self):
        with _Tmp() as d:
            art = _touch(os.path.join(d, "pretrain", "final_model.pth"))
            led = Ledger.load(d)
            led.begin("pretrain", config_hash="h1")
            led.complete("pretrain", {"final_model": art})
            led.reset("pretrain")
            assert led.status("pretrain") == StageStatus.PENDING
            assert led.artifacts("pretrain") == {}


class TestArtifactResolution:
    def test_deferred_reference_resolves_from_the_ledger(self):
        from occam.config.loader import resolve_artifact_ref

        led = {"stages": {"pretrain": {"status": "completed",
                                       "artifacts": {"final_model": "/tmp/f.pth"}}}}
        got = resolve_artifact_ref("${stages.pretrain.artifacts.final_model}", led)
        assert got == "/tmp/f.pth"

    def test_unrun_stage_raises_naming_the_stage(self):
        from occam.config.loader import ConfigError, resolve_artifact_ref

        with pytest.raises(ConfigError):
            resolve_artifact_ref("${stages.finetune.artifacts.final_model}", {"stages": {}})

    def test_error_message_tells_you_what_to_do(self):
        from occam.config.loader import ConfigError, resolve_artifact_ref

        try:
            resolve_artifact_ref("${stages.finetune.artifacts.final_model}",
                                 {"stages": {}}, "stages.decoder.init_from")
            raise AssertionError("should have raised")
        except ConfigError as exc:
            text = str(exc)
            assert "finetune" in text
            assert "--stages finetune" in text
            assert "init_from" in text

    def test_literal_paths_pass_through_untouched(self):
        from occam.config.loader import resolve_artifact_ref

        assert resolve_artifact_ref("/abs/path.pth", {}) == "/abs/path.pth"
        assert resolve_artifact_ref(None, {}) is None


class TestPlanning:
    def _config(self, tmp, stages=("pretrain", "finetune", "decoder")):
        from occam.config.loader import load_config

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = load_config(os.path.join(repo, "configs", "occam_small.yaml"))
        cfg.run["root"] = tmp
        cfg.run["stages"] = list(stages)
        return cfg

    def test_all_stages_run_on_a_fresh_tree(self):
        from occam.pipeline.orchestrator import Orchestrator

        with _Tmp() as d:
            cfg = self._config(d)
            plan = dict(Orchestrator(cfg).plan())
            assert plan == {"pretrain": "run", "finetune": "run", "decoder": "run"}

    def test_completed_stage_is_reused(self):
        from occam.pipeline.orchestrator import Orchestrator

        with _Tmp() as d:
            cfg = self._config(d)
            art = _touch(os.path.join(cfg.stage_dir("pretrain"), "final_model.pth"))
            led = Ledger.load(cfg.run_dir)
            led.begin("pretrain", config_hash=cfg.stage_hash("pretrain"))
            led.complete("pretrain", {"final_model": art})

            plan = dict(Orchestrator(cfg).plan())
            assert plan["pretrain"] == "reuse"
            assert plan["finetune"] == "run"

    def test_changed_hyperparameters_block_rather_than_reuse(self):
        """The core safety property."""
        from occam.pipeline.orchestrator import Orchestrator

        with _Tmp() as d:
            cfg = self._config(d)
            art = _touch(os.path.join(cfg.stage_dir("pretrain"), "final_model.pth"))
            led = Ledger.load(cfg.run_dir)
            led.begin("pretrain", config_hash="a-different-hash",
                      config_snapshot=cfg.stages["pretrain"])
            led.complete("pretrain", {"final_model": art})

            plan = dict(Orchestrator(cfg).plan())
            assert plan["pretrain"] == "blocked"

    def test_force_reruns_everything(self):
        from occam.pipeline.orchestrator import Orchestrator

        with _Tmp() as d:
            cfg = self._config(d)
            art = _touch(os.path.join(cfg.stage_dir("pretrain"), "final_model.pth"))
            led = Ledger.load(cfg.run_dir)
            led.begin("pretrain", config_hash=cfg.stage_hash("pretrain"))
            led.complete("pretrain", {"final_model": art})

            cfg.run["resume"] = "force"
            plan = dict(Orchestrator(cfg).plan())
            assert plan["pretrain"] == "run"

    def test_stage_order_is_canonical_regardless_of_yaml_order(self):
        with _Tmp() as d:
            cfg = self._config(d, stages=("decoder", "pretrain", "finetune"))
            assert cfg.selected_stages == ["pretrain", "finetune", "decoder"]

    def test_stage_hash_changes_when_a_hyperparameter_changes(self):
        with _Tmp() as d:
            cfg = self._config(d)
            before = cfg.stage_hash("decoder")
            cfg.stages["decoder"]["optim"]["lr"] = 1.0e-05
            assert cfg.stage_hash("decoder") != before

    def test_stage_hash_is_insensitive_to_other_stages(self):
        """Editing stage 1 must not invalidate a completed stage 3."""
        with _Tmp() as d:
            cfg = self._config(d)
            before = cfg.stage_hash("decoder")
            cfg.stages["pretrain"]["optim"]["epochs"] = 7
            assert cfg.stage_hash("decoder") == before
