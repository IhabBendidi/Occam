"""The `vcell` stage's schema must map onto real config attributes, and reject the rest.

`validate_stage` checks that YAML keys are known; it cannot check that a schema entry names
an attribute `VirtualCellConfig` actually has. A typo'd mapping would therefore pass
validation and then silently set nothing -- which is the exact failure mode the loader's
docstring says this repo refuses to accept ("a mistyped hyperparameter does not warn, it
changes nothing and lets you believe you changed something").
"""

from __future__ import annotations

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "configs", "rhj_fewshot.yaml")


def test_every_schema_attr_exists_on_the_config_object():
    pytest.importorskip("torch")
    from occam.config.schema import VCELL_SCHEMA
    from occam.vcell.config import VirtualCellConfig

    cfg = VirtualCellConfig()
    missing = []
    for group, mapping in VCELL_SCHEMA.items():
        for yaml_key, attr in mapping.items():
            if attr is None:        # consumed by the loader/orchestrator, not the config
                continue
            if not hasattr(cfg, attr):
                missing.append(f"{group or '<root>'}.{yaml_key} -> {attr}")
    assert not missing, f"schema names attributes that do not exist: {missing}"


def test_the_stage_is_registered_and_ordered_last():
    from occam.config.loader import _STAGE_ORDER
    from occam.config.schema import STAGE_SCHEMAS

    assert "vcell" in STAGE_SCHEMAS
    assert _STAGE_ORDER[-1] == "vcell", (
        "vcell consumes the decoder's output, so it must sort after it"
    )


def test_an_unknown_key_is_rejected_with_a_suggestion():
    from occam.config.schema import SchemaError, validate_stage

    with pytest.raises(SchemaError) as exc:
        validate_stage("vcell", {"interaction": {"film_mod": "full"}})
    assert "film_mode" in str(exc.value), "the error should suggest the real key"

    with pytest.raises(SchemaError):
        validate_stage("vcell", {"not_a_group": {"x": 1}})


def test_the_h1_panel_cannot_be_switched_on_from_this_schema():
    """No arm may accidentally drag in the hours-long panel scorer."""
    from occam.config.schema import SchemaError, validate_stage

    with pytest.raises(SchemaError):
        validate_stage("vcell", {"eval": {"panel_eval_enabled": True}})


@pytest.mark.skipif(not os.path.isfile(CONFIG), reason="config missing")
def test_the_reference_config_resolves_and_pins_the_benchmark():
    pytest.importorskip("torch")
    from occam.config.loader import load_config

    cfg = load_config(CONFIG)
    assert cfg.selected_stages == ["vcell"]
    vc, meta = cfg.build_vcell_config()

    # the decisions that define this benchmark
    assert vc.bench_id == "rhj_fewshot"
    assert vc.bench_sha256, "the artifact hash must be pinned in the config"
    assert vc.bench_axis_scope == "input_and_loss"
    assert vc.bench_eval_cell_lines == ["HEPG2"], "val must not draw from RPE1 or Jurkat"
    assert vc.bench_build_test_datasets is False
    # single cell in, single cell out
    assert vc.target_pb_k == 1 and vc.ctrl_avg_k == 1
    assert vc.n_controls_per_sample == 1, "5 would average controls into a pseudobulk"
    # the constraint that otherwise kills every nb arm on batch 1
    assert vc.source_weight == "none" and vc.loss_norm == "pooled"
    # the streaming settings that are not free choices
    assert vc.stream_v2 is True and vc.stream_interleave == "proportional"
    assert vc.freeze_encoder is True
    # film_mean_free needs the full node table as its reference
    assert vc.film_mean_free is True and vc.data_perts_only is False
    assert vc.film_scale == 1.0, "fixed by decision, not swept"
    assert os.path.isabs(vc.bench_dir), "registry paths resolve against the repo root"
    assert meta["init_from"].endswith("occam_small_encoder_decoder.pth")


@pytest.mark.skipif(not os.path.isfile(CONFIG), reason="config missing")
def test_the_config_declares_the_hash_the_artifact_actually_has():
    """Catches a config left pointing at a previous build of the benchmark."""
    pytest.importorskip("torch")
    from occam.config.loader import load_config
    from occam.vcell.benchmark import load_benchmark

    cfg = load_config(CONFIG)
    vc, _ = cfg.build_vcell_config()
    if not os.path.isfile(os.path.join(vc.bench_dir, "benchmark.json")):
        pytest.skip("benchmark not built")
    # load_benchmark raises on a mismatch; that is the assertion.
    bench = load_benchmark(vc.bench_dir, expect_id=vc.bench_id,
                           expect_sha256=vc.bench_sha256)
    assert len(bench.axis_ensembl) == 6983
    assert bench.eval_cell_lines == ["HEPG2"]
