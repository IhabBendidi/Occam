"""Rebuild metric views from the coverage-checked official score table."""

import csv
from pathlib import Path

from export_split_metrics import arm_label, write_tables

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/release/official_comparisons/selected_scores_by_seed_split.csv"
    with (ROOT / "data/release/final/scores_by_seed_split.csv").open() as stream:
        metrics = next(csv.reader(stream))[9:]
    rows = []
    with source.open() as stream:
        for row in csv.DictReader(stream):
            arm, mode, tag = arm_label(row["eval_root"], row["benchmark"])
            baseline = row["model_kind"] == "baseline"
            common = dict(
                benchmark=row["benchmark"], eval_root=row["eval_root"], arm=arm,
                compose_mode=mode, config_tag=tag,
                scored_arm=row["model"] if baseline else "model_nocal",
                arm_kind="reference" if baseline else row["model_kind"],
                calibration="n/a" if baseline else "nocal", split=int(row["split"]),
                panel=row["panel"], n_perts=int(float(row["n_perts"])),
            )
            for metric in metrics:
                if row.get(metric, ""):
                    rows.append(dict(common, metric=metric, value=float(row[metric])))
    write_tables(rows, str(ROOT / "data/release/metrics"))
    print(f"Exported {len(rows):,} metric values from the official release.")


if __name__ == "__main__":
    main()
