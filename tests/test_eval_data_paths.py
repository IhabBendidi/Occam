"""The evaluation bundle must resolve with no configuration, from anywhere.

The point of the bundle is that someone downloads it into `data/eval/` and the
benchmarks just run. That requires two properties, both easy to regress:

1. The shipped registry uses **relative** paths, so nothing is tied to one
   machine's filesystem.
2. Those relative paths resolve against the **repo root**, not the working
   directory, so it does not matter where the command is run from.
"""

from __future__ import annotations

import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "configs", "paths.example.yaml")
MANIFEST = os.path.join(REPO, "data", "eval", "manifest.json")


def _load_example():
    yaml = pytest.importorskip("yaml")
    with open(EXAMPLE, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# In the released bundle, resolved relative to the repo root.
BUNDLE_KEYS = ("eval_dir", "scib_dir")



def test_shipped_registry_uses_relative_paths_for_the_bundle():
    """A released config must not bake in one cluster's absolute paths."""
    reg = _load_example()
    for key in BUNDLE_KEYS:
        value = reg.get(key)
        assert value, f"{key} missing from configs/paths.example.yaml"
        assert not os.path.isabs(value), (
            f"{key} = {value!r} is absolute. The evaluation bundle keys must be "
            f"relative so the released config works on any machine."
        )
        assert value.startswith("data/eval"), (
            f"{key} = {value!r} should point inside data/eval/"
        )


def test_configs_resolve_every_eval_path_to_an_absolute_file():
    """Nothing unexpanded or empty survives into a stage config."""
    from occam.config.loader import load_config

    cfg = load_config(os.path.join(REPO, "configs", "occam_small.yaml"))
    dec, _meta = cfg.build_decoder_config()
    assert dec.eval_data, "the decoder stage has no eval panel"
    for path in dec.eval_data:
        assert path, "an empty filename survived into eval_data"
        assert os.path.isabs(path), f"{path} should have been made absolute"
        assert "${" not in path, f"{path} still contains an unexpanded reference"


def test_relative_paths_resolve_against_the_repo_not_the_cwd():
    from occam import paths

    resolved = paths.resolve("data/eval/pert")
    assert resolved == os.path.join(REPO, "data", "eval", "pert")
    assert os.path.isabs(resolved)

    # Unchanged when already absolute.
    assert paths.resolve("/tmp/x.h5ad") == "/tmp/x.h5ad"
    assert paths.resolve("") is None


def test_environment_overrides_also_resolve():
    from occam import paths

    old = os.environ.get("OCCAM_EVAL_DIR")
    try:
        os.environ["OCCAM_EVAL_DIR"] = "data/eval/pert"
        assert paths.get("eval_dir") == os.path.join(REPO, "data", "eval", "pert")
        os.environ["OCCAM_EVAL_DIR"] = "/somewhere/else"
        assert paths.get("eval_dir") == "/somewhere/else"
    finally:
        if old is None:
            os.environ.pop("OCCAM_EVAL_DIR", None)
        else:
            os.environ["OCCAM_EVAL_DIR"] = old


def test_dataset_filenames_match_what_the_exporter_writes():
    """The registry's expected filenames and the exporter's outputs must agree."""
    from occam import paths

    expected_pert = {n for n, _ in paths._BENCHMARK_FILES}
    expected_scib = {n for n, _, _, _ in paths._SCIB_FILES}
    assert expected_pert == {"nadig_2024_hepg2", "nadig_2024_jurkat",
                             "replogle_2022_rpe1_e"}
    assert expected_scib == {"pancreas_scib", "pbmc_12k", "pbmc_68k"}

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_eval_data", os.path.join(REPO, "scripts", "export_eval_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exported = {d[0] for d in mod.DATASETS}
    assert expected_pert | expected_scib <= exported, (
        f"the registry expects datasets the exporter does not write: "
        f"{(expected_pert | expected_scib) - exported}"
    )


@pytest.mark.skipif(not os.path.isfile(MANIFEST),
                    reason="evaluation bundle not exported; "
                           "run scripts/export_eval_data.py")
def test_every_manifest_entry_is_where_the_registry_looks_for_it():
    from occam import paths

    paths.load_registry(EXAMPLE)
    with open(MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    by_name = {d["name"]: d for d in man["datasets"]}

    for name, path in paths.eval_datasets():
        if name not in by_name:
            continue
        assert os.path.normpath(path) == os.path.normpath(
            os.path.join(REPO, by_name[name]["file"])), (
            f"registry looks for {name} at {path}, exporter wrote "
            f"{by_name[name]['file']}"
        )
    for name, path, _lab, _bat in paths.scib_datasets():
        if name not in by_name:
            continue
        assert os.path.normpath(path) == os.path.normpath(
            os.path.join(REPO, by_name[name]["file"]))


@pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="bundle not exported")
def test_export_kept_obs_and_var_and_dropped_only_obsm():
    with open(MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    for d in man["datasets"]:
        assert d["obs_columns"], f"{d['name']} recorded no obs columns"
        assert d["var_columns"], f"{d['name']} recorded no var columns"
        assert set(d.get("dropped", {})) <= {"obsm"}, (
            f"{d['name']} dropped something other than obsm: "
            f"{sorted(d.get('dropped', {}))}"
        )
