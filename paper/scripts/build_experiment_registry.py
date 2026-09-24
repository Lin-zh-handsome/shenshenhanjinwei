"""Index existing experiment artifacts without running models or inferring outcomes."""
from __future__ import annotations

import csv
from pathlib import Path

from common import PAPER, ROOT, SOURCES, write_csv

FIELDS = ["problem", "experiment_id", "experiment_name", "status", "model", "dataset_split",
          "feature_version", "config_path", "metrics_path", "checkpoint_path", "prediction_path",
          "purpose", "paper_role", "notes"]


def rel(path: Path | None) -> str:
    return path.relative_to(ROOT).as_posix() if path and path.is_file() else ""


def first(directory: Path, names: list[str]) -> Path | None:
    return next((directory / name for name in names if (directory / name).is_file()), None)


def add(rows, problem, name, directory, role, purpose, config=None, feature="aligned_50"):
    metrics = first(directory, ["metrics.json", "valid_metrics.json", "test_metrics.json", "valid_faithfulness_v2.json"])
    prediction = first(directory, ["valid_predictions.csv", "valid_missing_predictions.csv", "attachment4_predictions_explanations.csv", "attachment3_predictions_compact.csv", "valid_explanations.csv"])
    checkpoint = next(iter(sorted(directory.glob("*.pt"))), None)
    if not any((metrics, prediction, checkpoint)):
        return
    rows.append({"problem": problem, "experiment_id": name, "experiment_name": name,
                 "status": "RECORDED", "model": name, "dataset_split": "attachment4_unlabeled" if name == "attachment4_v2" else "valid" if metrics or prediction else "",
                 "feature_version": feature, "config_path": rel(config or first(directory, ["config.yaml"])),
                 "metrics_path": rel(metrics), "checkpoint_path": rel(checkpoint), "prediction_path": rel(prediction),
                 "purpose": purpose, "paper_role": role,
                 "notes": "Only files in repository are indexed; an empty checkpoint path does not imply no remote weight exists."})


def main():
    rows = []
    q1 = ROOT / "E题/Q1"
    rows.append({"problem": "Q1", "experiment_id": "q1_100_samples", "experiment_name": "Q1 fixed 50-window features",
                 "status": "RECORDED", "model": "feature extraction and temporal alignment", "dataset_split": "attachment1_unlabeled",
                 "feature_version": "Q1_K50_128_74_52", "config_path": SOURCES["q1_config"],
                 "metrics_path": SOURCES["q1_validation"], "checkpoint_path": "", "prediction_path": "",
                 "purpose": "100-sample feature extraction", "paper_role": "FINAL", "notes": rel(q1 / "q1_features.npz")})
    q2 = ROOT / "E题/Q2/outputs"
    for directory in sorted((q2 / "q2_best_ablation").iterdir()):
        if directory.is_dir():
            name = directory.name
            role = "FINAL" if name == "R3_reconstruction" else "BASELINE" if name.startswith("R0") else "ABLATION"
            add(rows, "Q2", name, directory, role, "clean and synthetic mixed-missing validation",
                ROOT / "E题/Q2/configs" / ("best_R3_reconstruction.yaml" if role == "FINAL" else "best_r0_selected_bert.yaml" if role == "BASELINE" else f"best_{name}.yaml"))
    for directory in sorted(q2.rglob("metrics.json")):
        if "q2_best_ablation" not in directory.parts:
            add(rows, "Q2", directory.parent.name, directory.parent, "HISTORICAL", "prior Q2 experiment")
    v2 = ROOT / "E题/Q3/outputs/q3_v2"
    q3_configs = {
        "00_base_predictor": "q3_v2_base.yaml", "01_a1_reproduction": "q3_v2_base.yaml",
        "02_position": "q3_v2_position.yaml", "03_gate_pooling": "q3_v2_gate_pooling.yaml",
        "04_gate_regularized": "q3_v2_gate_regularized.yaml", "05_router_diagnostic": "q3_v2_router_diagnostic.yaml",
    }
    for directory in sorted(v2.iterdir()):
        if directory.is_dir() and directory.name[:2].isdigit():
            name = directory.name
            add(rows, "Q3", name, directory, "FINAL" if name.startswith("01_") else "BASELINE" if name.startswith("00_") else "ABLATION",
                "v2 validation prediction", ROOT / "E题/Q3/configs" / q3_configs[name])
    for path in sorted((v2 / "faithfulness").rglob("valid_faithfulness_v2.json")):
        name = "/".join(path.parent.relative_to(v2 / "faithfulness").parts)
        add(rows, "Q3", name, path.parent, "FINAL" if name == "final_b1_router/budget_025" else "ABLATION",
            "faithfulness on validation", ROOT / "E题/Q3/configs/q3_v2_final.yaml")
    add(rows, "Q3", "attachment4_v2", v2 / "attachment4", "FINAL", "unlabeled attachment4 inference",
        ROOT / "E题/Q3/configs/q3_v2_final.yaml")
    add(rows, "Q3", "legacy_q3", ROOT / "E题/Q3/outputs/q3", "HISTORICAL", "prior Q3 results",
        ROOT / "E题/Q3/configs/q3_aligned.yaml")
    for row in rows:
        if row["problem"] == "Q3" and row["paper_role"] == "FINAL":
            row["checkpoint_path"] = "E题/Q3/outputs/q3_v2/final_model.pt"
        if row["problem"] == "Q2" and row["experiment_id"] == "R3_reconstruction":
            row["checkpoint_path"] = "paper_package/Q2/model_parameters/best_robust_delta.pt"
            row["notes"] = "Compact R3 delta only; also requires best_bert_last4_compact.pt and public BERT base weights."
    write_csv(PAPER / "EXPERIMENT_REGISTRY.csv", rows, FIELDS)
    print(f"Registered {len(rows)} existing experiment artifact groups")


if __name__ == "__main__":
    main()
