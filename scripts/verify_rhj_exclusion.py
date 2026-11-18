#!/usr/bin/env python
"""Prove the per-(file, perturbation) exclusion actually bit. Both halves of it.

    python scripts/verify_rhj_exclusion.py --config configs/rhj_fewshot.yaml
    python scripts/verify_rhj_exclusion.py --config ... --checkpoint runs/.../best_model.pth

⛔ THIS IS THE GATE. Do not launch the grid until it passes.

Builds the streaming dataset exactly as training does -- same config, same metadata, same
eligibility gate -- and reads `audit_eligible_perts()`, which reports the rows the gate
actually admits. No GPU, no model, no training: `_static_ok` is the only per-row gate the
v2 iterator consults, so a zero there is conclusive, and the row cap and chunking downstream
can only remove rows, never restore an excluded one.

WHY BOTH HALVES
---------------
The negative half -- "no held-out HepG2 row trained" -- is satisfied by a *global*
perturbation-name exclusion too, and a global exclusion would also have deleted those same
728 perturbations from Jurkat and RPE1, i.e. two contexts of training data, silently, turning
a few-shot benchmark into a zero-shot one that still looks fine.

So the positive half is the load-bearing assertion: **the same symbols must still be eligible
from the other two files.** Only a per-(file, perturbation) exclusion passes both.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Check:
    def __init__(self) -> None:
        self.failures = []
        self.passed = 0

    def ok(self, label, detail=""):
        self.passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))

    def fail(self, label, detail):
        self.failures.append(f"{label}: {detail}")
        print(f"  FAIL  {label}\n          {detail}")

    def expect(self, cond, label, detail=""):
        self.ok(label, detail) if cond else self.fail(label, detail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rhj_fewshot.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="also cross-check a trained checkpoint's recorded split_info")
    ap.add_argument("--out", default=None, help="write the audit as JSON here")
    args = ap.parse_args(argv)

    from occam.config.loader import load_config
    from occam.vcell.benchmark import load_benchmark
    from occam.vcell.dataset import prepare_virtual_cell_data, VirtualCellStreamingDataset
    from occam.vcell.graph import load_pert2id

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO, args.config)
    cfg = load_config(cfg_path)
    vc, _meta = cfg.build_vcell_config()

    print("rhj_fewshot exclusion audit")
    print(f"  config     : {cfg_path}")
    print(f"  benchmark  : {vc.bench_dir}\n")

    import torch
    ck = torch.load(os.path.join(REPO, _meta["init_from"]) if not os.path.isabs(
        _meta["init_from"]) else _meta["init_from"], map_location="cpu", weights_only=False)
    encoder_genes = [str(g) for g in ck["gene_names"]]
    del ck

    bench = load_benchmark(vc.bench_dir, expect_id=vc.bench_id,
                           expect_sha256=vc.bench_sha256,
                           data_paths=list(vc.data_paths),
                           encoder_gene_names=encoder_genes)
    print(f"  {bench.summary()}\n")

    pert2id, _ = load_pert2id(vc.master_gene_set_path)
    print("[1] building the dataset exactly as training does")
    file_metas, matcher = prepare_virtual_cell_data(
        data_paths=list(vc.data_paths),
        encoder_gene_names=encoder_genes,
        pert2id=pert2id,
        target_sum=vc.target_sum,
        gene_name_col=vc.gene_name_col,
        pert_gene_col=vc.pert_gene_col,
        seed=vc.seed,
        restrict_gene_ids=(bench.axis_set
                           if vc.bench_axis_scope == "input_and_loss" else None),
        exclude_file_perts=bench.exclude_file_perts,
        build_pools=False,
    )

    train_perts = set(bench.split["train"])
    lines = sorted({cl for fm in file_metas for cl in set(fm.obs_cell_lines)})
    common = dict(
        file_metas=file_metas, pert2id=pert2id, split_pert_names=train_perts,
        control_matcher=matcher, encoder_gene_names=encoder_genes,
        target_sum=vc.target_sum, n_controls=1, batch_size=8,
        train_cell_lines=lines,
        min_cells_per_pert=int(getattr(vc, "min_cells_per_pert_stream", 0) or 0),
        max_cells_per_pert=int(getattr(vc, "max_cells_per_pert_stream", 0) or 0),
    )

    # Audit the class the config ACTUALLY trains with. v1 and v2 implement the same
    # exclusion in different places -- v1 in its row loop, v2 in `_static_ok` -- and
    # auditing the wrong one would prove nothing about the run.
    if bool(getattr(vc, "stream_v2", False)):
        from occam.vcell.stream_v2 import StreamingDatasetV2

        # Index/mask arguments only. The audit never iterates, so no control embeddings and
        # no GPU are needed; `_static_ok` is built in __init__.
        mask_table = torch.stack([fm.gene_mask for fm in file_metas])
        ds = StreamingDatasetV2(
            **common,
            pool_index={cl: (0, 1) for cl in lines},
            mask_table=mask_table,
            mask_id_of_meta={i: i for i in range(len(file_metas))},
            interleave=str(getattr(vc, "stream_interleave", "roundrobin")),
            ctrl_avg_k=int(getattr(vc, "ctrl_avg_k", 1)),
        )
        which = "StreamingDatasetV2._static_ok"
    else:
        ds = VirtualCellStreamingDataset(**common)
        which = "VirtualCellStreamingDataset.__iter__ predicates"
    print(f"  auditing: {which}")
    audit = ds.audit_eligible_perts()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(audit, fh, indent=2, sort_keys=True)
        print(f"  wrote {args.out}")

    holdout = sorted(bench.exclude_file_perts)[0]
    held = set(bench.exclude_file_perts[holdout])
    others = [n for n in audit if n != holdout]

    print(f"\n[2] the invariant ({len(held)} held-out perturbations, holdout file {holdout})")
    c = Check()

    # --- negative half -------------------------------------------------
    leaked = {p: audit[holdout].get(p, 0) for p in held if audit[holdout].get(p, 0) > 0}
    c.expect(not leaked, "no held-out perturbation is eligible in the holdout file",
             f"{len(leaked)} leaked, e.g. {sorted(leaked)[:5]}" if leaked
             else f"all {len(held)} held out")

    # --- positive half: the one a global exclusion would fail ----------
    for name in sorted(others):
        absent = sorted(p for p in held if audit[name].get(p, 0) == 0)
        c.expect(not absent,
                 f"held-out perturbations STILL train from {name}",
                 f"{len(absent)} absent, e.g. {absent[:5]}" if absent
                 else f"all {len(held)} eligible, "
                      f"{sum(audit[name].get(p, 0) for p in held):,} rows")

    # --- the holdout file's training set is the intended complement ----
    trained = {p for p, n in audit[holdout].items() if n > 0}
    c.expect(not (trained & held), "holdout training set is disjoint from the held-out set")
    expected_train = (set(bench.split["train"])
                      & set(json.load(open(os.path.join(
                          bench.dir, "files", holdout.replace(".h5ad", ".json"))))[
                          "per_pert_cells"])) - held
    c.expect(trained == expected_train,
             "holdout training perturbations are exactly the benchmark's",
             f"{len(trained)} trained vs {len(expected_train)} expected")

    # --- nothing outside the eligible set slipped through --------------
    stray = {n: sorted(set(audit[n]) - set(bench.split["train"]))[:5]
             for n in audit if set(audit[n]) - set(bench.split["train"])}
    c.expect(not stray, "no ineligible perturbation is eligible anywhere", f"{stray}")

    # --- cell counts line up -------------------------------------------
    rec = json.load(open(os.path.join(bench.dir, "files",
                                      holdout.replace(".h5ad", ".json"))))
    withheld_cells = sum(rec["per_pert_cells"].get(p, 0) for p in held)
    trained_cells = sum(audit[holdout].values())
    c.expect(trained_cells == rec["n_perturbed"] - withheld_cells
             - sum(rec["per_pert_cells"].get(p, 0)
                   for p in set(rec["per_pert_cells"]) - set(bench.split["train"]) - held),
             "holdout perturbed-cell count matches the manifest",
             f"{trained_cells:,} eligible, {withheld_cells:,} withheld")

    # --- optional: what the checkpoint itself recorded ------------------
    if args.checkpoint:
        print("\n[3] checkpoint cross-check")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        si = ckpt.get("split_info") or {}
        bm = ckpt.get("benchmark") or {}
        recorded = set(si.get("val", ())) | set(si.get("test_unseen_pert", ()))
        c.expect(recorded == held, "split_info val+test == the held-out set",
                 f"{len(recorded)} vs {len(held)}")
        c.expect(held <= set(si.get("train", ())),
                 "every held-out symbol is in split_info['train']",
                 "they train from the other two contexts")
        c.expect(bm.get("gene_axis_sha256") == bench.axis_sha256,
                 "checkpoint gene_axis_sha256 matches the artifact")
        c.expect(list(ckpt.get("val_cell_lines") or []) == list(bench.eval_cell_lines),
                 "val was drawn from the benchmark's eval cell lines only",
                 f"{ckpt.get('val_cell_lines')}")

    print(f"\n{'=' * 70}")
    if c.failures:
        print(f"FAILED  {len(c.failures)} check(s), {c.passed} passed")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print(f"OK  all {c.passed} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
