"""Collect existing figures and plot small summaries from archived CSV/JSON only."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from common import PAPER, ROOT


def main():
    out = PAPER / "figures"
    out.mkdir(exist_ok=True)
    existing = {
        "E题/Q1/figures/q1_typical_medium.png": "q1_alignment_example.png",
        "E题/Q2/figures/fig_missing_ratio.png": "q2_missing_ratio.png",
        "E题/Q2/figures/fig_missing_type.png": "q2_missing_modality.png",
        "E题/Q2/figures/fig_missing_position.png": "q2_missing_position.png",
        "E题/Q2/figures/fig_missing_span_length.png": "q2_missing_length.png",
        "E题/Q3/outputs/q3_v2/attachment4/explanation_cards/01.png": "q3_evidence_example.png",
    }
    for origin, name in existing.items():
        path = ROOT / origin
        if path.is_file():
            shutil.copyfile(path, out / name)
            print(f"{name} <- {origin}")
        else:
            print(f"SKIP {name}: {origin} absent")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("SKIP q2_ablation/q3_modality_importance/q3_faithfulness: matplotlib absent")
        return
    specs = [
        (PAPER / "tables/q2_ablation.csv", "experiment", "macro_f1", "q2_ablation.png", "Q2 Macro-F1", "scene"),
        (PAPER / "generated/q3_modality_importance_plot.csv", "modality", "mean_lomo_importance", "q3_modality_importance.png", "Mean LOMO importance", None),
        (PAPER / "generated/q3_faithfulness_plot.csv", "condition", "macro_f1", "q3_faithfulness.png", "Q3 Macro-F1", None),
    ]
    for path, x, y, name, label, group in specs:
        if not path.is_file():
            print(f"SKIP {name}: {path.name} absent")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"SKIP {name}: no rows")
            continue
        fig, ax = plt.subplots(figsize=(7, 3.6))
        if group:
            categories = list(dict.fromkeys(row[x] for row in rows))
            groups = list(dict.fromkeys(row[group] for row in rows))
            width = 0.8 / len(groups)
            for index, value in enumerate(groups):
                selected = {row[x]: row for row in rows if row[group] == value}
                ax.bar([i - .4 + width * (index + .5) for i in range(len(categories))],
                       [float(selected[c][y]) for c in categories], width=width, label=value)
            ax.set_xticks(range(len(categories)), categories, rotation=25, ha="right")
            ax.legend()
        else:
            ax.bar([row[x] for row in rows], [float(row[y]) for row in rows])
            ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
        print(name)


if __name__ == "__main__":
    main()
