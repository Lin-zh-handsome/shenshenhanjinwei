"""Summarize Q2 clean-validation experiments without reading the test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "outputs/q2_score_push"
RUN_A = ROOT / "outputs/q2_v2/01_run_a_oracle_text"
OLD = ROOT / "outputs/q2_v2/00_baseline_reproduction"


def read_metrics(path: Path) -> dict:
    file = path / "metrics.json"
    if path == OLD:
        file = path / "valid_metrics.json"
    with file.open(encoding="utf-8") as handle:
        item = json.load(handle)
    if path == OLD:
        item = {"valid": item["clean"], "selected_epoch": item.get("best_joint_epoch"), "test_evaluated": False}
    return item


def row(name: str, directory: Path) -> dict | None:
    if not ((directory / "metrics.json").exists() or (directory == OLD and (directory / "valid_metrics.json").exists())):
        return None
    item = read_metrics(directory)
    valid = item.get("valid", item)
    classes = item.get("class_metrics", [])
    by_name = {str(part.get("class_name", "")).lower(): part for part in classes}
    return {
        "experiment": name,
        "accuracy": valid.get("accuracy"),
        "macro_f1": valid.get("f1_macro", valid.get("macro_f1")),
        "weighted_f1": valid.get("f1_weighted", valid.get("weighted_f1")),
        "neg_f1": by_name.get("negative", {}).get("f1"),
        "neutral_f1": by_name.get("neutral", {}).get("f1"),
        "pos_f1": by_name.get("positive", {}).get("f1"),
        "neutral_precision": by_name.get("neutral", {}).get("precision"),
        "neutral_recall": by_name.get("neutral", {}).get("recall"),
        "mae": valid.get("mae"),
        "pearson": valid.get("pearson"),
        "trainable_params": item.get("trainable_params"),
        "epoch": item.get("selected_epoch"),
        "gpu_memory_mb": item.get("gpu_memory_mb"),
        "validation_only": not item.get("test_evaluated", False),
        "directory": str(directory.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensemble() -> dict | None:
    seed_paths = [RUN_A, SCORES / "seeds_oracle/2026", SCORES / "seeds_oracle/3407"]
    if not all((path / "valid_predictions.csv").exists() for path in seed_paths):
        return None
    tables = []
    for path in seed_paths:
        with (path / "valid_predictions.csv").open(newline="", encoding="utf-8") as handle:
            tables.append({record["id"]: record for record in csv.DictReader(handle)})
    ids = list(tables[0])
    if any(set(ids) != set(table) for table in tables[1:]):
        raise ValueError("Validation sample IDs differ across seeds")
    output_rows = []
    for sample_id in ids:
        records = [table[sample_id] for table in tables]
        labels = {int(record["true_class"]) for record in records}
        targets = np.array([float(record["true_reg"]) for record in records])
        if len(labels) != 1 or not np.allclose(targets, targets[0]):
            raise ValueError(f"Validation labels differ for {sample_id}")
        probabilities = np.array([[float(record[f"p_{name}"]) for name in ("negative", "neutral", "positive")] for record in records]).mean(axis=0)
        output_rows.append({
            "id": sample_id,
            "true_class": labels.pop(),
            "pred_class": int(probabilities.argmax()),
            "p_negative": probabilities[0],
            "p_neutral": probabilities[1],
            "p_positive": probabilities[2],
            "true_reg": targets[0],
            "pred_reg": np.mean([float(record["pred_reg"]) for record in records]),
        })
    true_class = [record["true_class"] for record in output_rows]
    pred_class = [record["pred_class"] for record in output_rows]
    true_reg = np.array([record["true_reg"] for record in output_rows])
    pred_reg = np.array([record["pred_reg"] for record in output_rows])
    precision, recall, class_f1, support = precision_recall_fscore_support(true_class, pred_class, labels=[0, 1, 2], zero_division=0)
    result = {
        "run_name": "Oracle_Three_Seed_Ensemble",
        "seeds": [42, 2026, 3407],
        "valid": {
            "accuracy": accuracy_score(true_class, pred_class),
            "f1_macro": f1_score(true_class, pred_class, average="macro"),
            "f1_weighted": f1_score(true_class, pred_class, average="weighted"),
            "mae": mean_absolute_error(true_reg, pred_reg),
            "pearson": float(np.corrcoef(true_reg, pred_reg)[0, 1]),
        },
        "class_metrics": [
            {"class_id": index, "class_name": name, "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(class_f1[index]), "support": int(support[index])}
            for index, name in enumerate(("Negative", "Neutral", "Positive"))
        ],
        "test_evaluated": False,
    }
    output_dir = SCORES / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "valid_predictions.csv", output_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    seed_rows = [row("seed_42", seed_paths[0]), row("seed_2026", seed_paths[1]), row("seed_3407", seed_paths[2])]
    write_csv(output_dir / "seed_stability.csv", seed_rows)
    numeric = ("accuracy", "macro_f1", "weighted_f1", "neutral_f1", "mae", "pearson")
    summary = {key: {"mean": float(np.mean([item[key] for item in seed_rows])), "std": float(np.std([item[key] for item in seed_rows], ddof=1))} for key in numeric}
    with (output_dir / "seed_stability.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()
    if args.ensemble:
        ensemble()
    run_dirs = [
        ("Old SRF-MSA", OLD),
        ("Run A Oracle", RUN_A),
        ("C1 position", SCORES / "C1_position"),
        ("C1 position small init", SCORES / "C1_position_smallinit"),
        ("C2 temporal", SCORES / "C2_temporal"),
        ("C2 temporal dropout .2", SCORES / "C2_temporal_dropout02"),
        ("C3 mean pool", SCORES / "C3_pool_mean"),
        ("C3 attention max pool", SCORES / "C3_pool_attention_max"),
        ("C4 text centered fusion", SCORES / "C4_text_centered_fusion"),
        ("C5 CE", SCORES / "C5_loss/ce"),
        ("C5 CE smoothing", SCORES / "C5_loss/ce_smooth"),
        ("C5 sqrt weighted CE", SCORES / "C5_loss/weighted_ce_sqrt"),
        ("C5 focal", SCORES / "C5_loss/focal"),
        ("C5 regression .3", SCORES / "C5_regression/0p3"),
        ("C5 regression .8", SCORES / "C5_regression/0p8"),
        ("C5 lr 2e-4", SCORES / "C5_lr2e4"),
        ("C6 EMA", SCORES / "C6_ema"),
        ("Deployable BERT frozen", SCORES / "deployable_bert/frozen_bridge"),
        ("Deployable BERT last 4", SCORES / "deployable_bert/partial_last4"),
        ("Deployable BERT distill .1", SCORES / "deployable_bert/distill_0p1"),
        ("Deployable BERT distill .05", SCORES / "deployable_bert/distill_0p05"),
        ("Deployable BERT distill .2", SCORES / "deployable_bert/distill_0p2"),
        ("Mild Neutral oversampling", SCORES / "neutral_sampler"),
        ("3-seed ensemble", SCORES / "ensemble"),
    ]
    rows = [item for name, path in run_dirs if (item := row(name, path)) is not None]
    write_csv(SCORES / "score_push_ablation.csv", rows)
    modalities = [item for code in ("T", "TA", "TV") if (item := row(code, SCORES / "modalities_oracle" / code)) is not None]
    modalities.append(row("TAV", RUN_A))
    write_csv(SCORES / "modality_ablation_clean.csv", modalities)
    print(f"Wrote {len(rows)} experiment rows and {len(modalities)} modality rows")


if __name__ == "__main__":
    main()
