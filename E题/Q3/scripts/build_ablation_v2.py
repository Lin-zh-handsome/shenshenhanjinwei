import csv
import json
from pathlib import Path


ROOT = Path("outputs/q3_v2")
VARIANTS = {
    "B0": "00_base_predictor",
    "B1": "01_a1_reproduction",
    "B2": "02_position",
    "B3": "03_gate_pooling",
    "B4": "04_gate_regularized",
    "B5": "05_router_diagnostic",
}
FIELDS = [
    "variant", "accuracy", "macro_f1", "mae", "pearson", "evidence_budget",
    "keep_only_accuracy", "keep_only_macro_f1", "keep_only_mae", "keep_only_pearson",
    "remove_accuracy", "remove_macro_f1", "remove_mae", "remove_pearson",
    "random_remove_accuracy", "random_remove_macro_f1", "random_remove_mae", "random_remove_pearson",
    "confidence_drop_top", "confidence_drop_random", "top_vs_random_f1_drop", "router_lomo_agreement",
]


def build():
    rows = []
    for variant, folder in VARIANTS.items():
        metrics = json.loads((ROOT / folder / "valid_metrics.json").read_text(encoding="utf-8"))
        row = {"variant": variant, "accuracy": metrics["accuracy"],
               "macro_f1": metrics["f1_macro"], "mae": metrics["mae"],
               "pearson": metrics["pearson"]}
        if variant != "B0":
            fidelity = json.loads((ROOT / "faithfulness" / variant / "budget_025" /
                                   "valid_faithfulness_v2.json").read_text(encoding="utf-8"))
            row["evidence_budget"] = fidelity["evidence_budget"]
            for label, source in (("keep_only", "sufficiency"), ("remove", "comprehensiveness"),
                                  ("random_remove", "random_remove")):
                for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                    row[f"{label}_{metric}"] = fidelity[f"{source}_{metric}"]
            for key in ("confidence_drop_top", "confidence_drop_random", "top_vs_random_f1_drop"):
                row[key] = fidelity[key]
            if variant == "B5":
                row["router_lomo_agreement"] = fidelity["router_vs_lomo"]["main_modality_agreement"]
        rows.append(row)
    with (ROOT / "ablation_v2.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    for row in build():
        print(row["variant"], row["accuracy"], row["macro_f1"], flush=True)
