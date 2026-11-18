#!/usr/bin/env python3
"""Build coverage-checked official Occam comparison tables for every frozen panel.

Usage: python scripts/official_score_tables.py

Each split is averaged within its training seed; displayed uncertainty is the
sample SD (ddof=1) of the five seed means. Arms are sorted independently on
each panel by the sum of their ranks on the four headline metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/release/final"
NORMAN = ROOT / "data/release/norman_allsource"
OUT = ROOT / "data/release/official_comparisons"
METRICS = (
    "pearson_delta",
    "mae",
    "overlap_at_N",
    "discrimination_score_l1",
)
BENCHMARKS = (
    "rhj_fewshot",
    "h1_seen",
    "k562_seen",
    "k562_seen_h1set",
    "norman_combo",
)
NORMAN_PANELS = (
    "additive",
    "nonadditive",
    "middle",
    "all",
    "additive_raw",
    "nonadditive_raw",
    "middle_raw",
)
COMPETITORS = {"txpert_single", "txpert", "state", "cellflow",
               "perturbdiff_genept", "perturbdiff"}
LABELS = {
    "experimental": "Experimental",
    "sum_of_singles": "Sum of singles",
    "transfer": "Transfer",
    "meanpert": "Mean perturbation",
    "ctrlmean": "Control mean",
    "ctrlcell": "Control cell",
    "txpert_single": "TxPert",
    "txpert": "TxPert",
    "state": "STATE",
    "cellflow": "CellFlow",
    "perturbdiff_genept": "PerturbDiff (GenePT)",
    "perturbdiff": "PerturbDiff",
    "ABL_N_ALL": "Occam ABL_N_ALL (official)",
    "NF32ALLA": "Occam NF32ALLA (official)",
}


def check_coverage(rows: pd.DataFrame, coverage: pd.DataFrame, benchmark: str,
                   panel: str, model: str) -> None:
    seeds = set(range(42, 47))
    assert set(rows.seed) == seeds, (benchmark, panel, model, "seeds")
    assert len(rows) == 25 and not rows.duplicated(["seed", "split"]).any(), (
        benchmark, panel, model, "rows")
    for seed in seeds:
        assert set(rows[rows.seed.eq(seed)].split) == set(range(5)), (
            benchmark, panel, model, seed)
    c = coverage[(coverage.benchmark.eq(benchmark)) &
                 (coverage.panel.eq(panel)) & (coverage.model.eq(model))]
    assert len(c) == 1 and int(c.iloc[0].complete_seeds) == 5, (
        benchmark, panel, model, "ledger")
    assert all(int(c.iloc[0][str(seed)]) == 5 for seed in seeds)
    assert rows[list(METRICS)].notna().all().all(), (benchmark, panel, model, "missing metric")


def selected_rows(final: pd.DataFrame, new: pd.DataFrame,
                  benchmark: str, panel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = final[final.benchmark.eq(benchmark) & final.panel.eq(panel)]
    old = old[old.model_kind.isin(["baseline", "competitor"])]
    official = "NF32ALLA" if benchmark == "norman_combo" else "ABL_N_ALL"
    source = new if benchmark == "norman_combo" else final
    ours = source[source.benchmark.eq(benchmark) & source.panel.eq(panel) &
                  source.model.eq(official)]
    assert set(old[old.model_kind.eq("competitor")].model) == (
        COMPETITORS & set(old.model)), (benchmark, panel, "competitors")
    assert len(old[old.model_kind.eq("competitor")].model.unique()) == 4
    assert set(old[old.model_kind.eq("baseline")].model) == (
        {"experimental", "sum_of_singles", "meanpert", "ctrlmean", "ctrlcell"}
        if benchmark == "norman_combo" else
        {"experimental", "transfer", "meanpert", "ctrlmean", "ctrlcell"}
    ), (benchmark, panel, "baselines")
    return pd.concat([old, ours], ignore_index=True), ours


def format_metric(value: float, sd: float, metric: str) -> str:
    digits = 5 if metric == "mae" else 4
    return f"{value:.{digits}f} ± {sd:.{digits}f}"


def main() -> None:
    final = pd.read_csv(FINAL / "scores_by_seed_split.csv")
    new = pd.read_csv(NORMAN / "scores_by_seed_split.csv")
    final_coverage = pd.read_csv(FINAL / "coverage.csv")
    new_coverage = pd.read_csv(NORMAN / "coverage.csv")
    summary = []
    selected_splits = []
    panel_documents = {}
    sections = [
        "# Official Occam comparisons by frozen benchmark panel",
        "",
        "Uncalibrated `model_nocal` scores. Every row has five training seeds (42–46) "
        "and five splits (0–4) per seed. For each metric, average the splits within "
        "each seed, then report the mean and sample SD (`ddof=1`) across the five "
        "seed means. Larger is better except MAE. Discrimination L1 chance is 0.5.",
        "",
        "Rows are ranked **within each panel** by the sum of four metric ranks: "
        "Pearson Δ, overlap@N, and discrimination L1 descending; MAE ascending. "
        "A smaller rank sum is better. Tied metric values receive average ranks; "
        "remaining ties are resolved by Pearson Δ, then MAE. This ordering is a "
        "descriptive four-metric comparison, not a new benchmark score.",
        "",
        "Official Occam arms: `ABL_N_ALL` (GO+STRING + GenePT + ESM2, "
        "smooth-L1 + 0.5 delta-cos) for RHJ/H1/K562; `NF32ALLA` (the same "
        "sources, NF32 schedule, smooth-L1 + 0.0255 delta-cos, "
        "`decode_seq_mean`) for Norman. Norman's `additive` and `nonadditive` "
        "panels are the headline strata. Its `_raw`, `middle`, and `all` panels "
        "are displayed separately and are not pooled into those strata.",
        "",
    ]
    for benchmark in BENCHMARKS:
        panels = NORMAN_PANELS if benchmark == "norman_combo" else ("all",)
        for panel in panels:
            rows, ours = selected_rows(final, new, benchmark, panel)
            assert not ours.empty, (benchmark, panel, "official missing")
            rows["calibration"] = "nocal"
            rows["score_source"] = rows.model.map(
                lambda model: "norman_allsource" if model == "NF32ALLA" else "final")
            selected_splits.append(rows)
            for model, arm in rows.groupby("model"):
                ledger = new_coverage if model == "NF32ALLA" else final_coverage
                check_coverage(arm, ledger, benchmark, panel, model)
            seed_means = rows.groupby(["model_kind", "model", "seed"], as_index=False)[list(METRICS)].mean()
            grouped = seed_means.groupby(["model_kind", "model"], as_index=False)
            means = grouped[list(METRICS)].mean()
            sds = grouped[list(METRICS)].std(ddof=1)
            table = means.copy()
            for metric in METRICS:
                table[f"{metric}_sd"] = sds[metric]
                table[f"{metric}_rank"] = table[metric].rank(
                    ascending=(metric == "mae"), method="average")
            table["rank_sum"] = table[[f"{m}_rank" for m in METRICS]].sum(axis=1)
            table["benchmark"] = benchmark
            table["panel"] = panel
            table["variant"] = table.model.map(
                {"NF32ALLA": "decode_seq_mean", "ABL_N_ALL": ""}).fillna("")
            table["calibration"] = "nocal"
            table["score_source"] = table.model.map(
                lambda model: "norman_allsource" if model == "NF32ALLA" else "final")
            table["seeds"] = 5
            table["splits_per_seed"] = 5
            table = table.sort_values(
                ["rank_sum", "pearson_delta", "mae", "model"],
                ascending=[True, False, True, True]).reset_index(drop=True)
            table["position"] = range(1, len(table) + 1)
            summary.append(table)
            panel_lines = [
                f"## {benchmark} · {panel}",
                "",
                "| Rank | Arm | Coverage | Rank sum ↓ | Pearson Δ ↑ | MAE ↓ | Overlap@N ↑ | Discrimination L1 ↑ |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
            for _, r in table.iterrows():
                name = LABELS.get(r.model, r.model)
                cells = [format_metric(r[m], r[f"{m}_sd"], m) for m in METRICS]
                panel_lines.append(
                    f"| {r.position} | {name} | 5 × 5 | {r.rank_sum:g} | " +
                    " | ".join(cells) + " |")
            panel_lines.append("")
            sections.extend(panel_lines)
            panel_documents[f"{benchmark}__{panel}.md"] = "\n".join([
                f"# {benchmark} · {panel}",
                "",
                "Uncalibrated scores; 5 seeds × 5 splits for every row. Values are "
                "mean ± sample SD of seed means. Rows are ordered by the sum of "
                "within-panel ranks on all four metrics; lower rank sum is better.",
                "",
                *panel_lines[2:],
            ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "official_tables.md").write_text("\n".join(sections))
    for filename, content in panel_documents.items():
        (OUT / filename).write_text(content)
    pd.concat(summary, ignore_index=True).to_csv(OUT / "official_tables.csv", index=False)
    pd.concat(selected_splits, ignore_index=True).to_csv(
        OUT / "selected_scores_by_seed_split.csv", index=False)
    print(f"Wrote {len(summary)} panel tables, {sum(len(t) for t in summary)} rows to {OUT}")


if __name__ == "__main__":
    main()
