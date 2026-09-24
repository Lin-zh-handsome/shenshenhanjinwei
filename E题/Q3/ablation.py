import argparse
import json
from pathlib import Path

import pandas as pd

from faithfulness_eval import run_faithfulness
from train import train_model
from utils.io import load_config
from evaluate import evaluate_split


def run_ablation(cfg):
    output = Path(cfg["paths"]["output_dir"])
    rows = []
    for variant in (f"A{i}" for i in range(6)):
        directory = output if variant == "A5" else output / "ablations" / variant
        checkpoint = directory / "best_joint.pt"
        if not checkpoint.is_file():
            train_model(cfg, variant, directory)
        metrics_path = directory / "valid_metrics.json"
        if metrics_path.is_file() and metrics_path.stat().st_mtime >= checkpoint.stat().st_mtime:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = evaluate_split(cfg, checkpoint, "valid", directory)
        if variant == "A0":
            fidelity = {"suff_cls": None, "suff_reg": None, "f1_drop_after_evidence_removal": None,
                        "mae_increase_after_evidence_removal": None, "selected_ratio": None}
        else:
            fidelity_path = directory / "valid_faithfulness.json"
            if fidelity_path.is_file() and fidelity_path.stat().st_mtime >= checkpoint.stat().st_mtime:
                result = json.loads(fidelity_path.read_text(encoding="utf-8"))
            else:
                result = run_faithfulness(cfg, checkpoint, output_dir=directory)
            fidelity = {k: result[k] for k in ("suff_cls", "suff_reg", "f1_drop_after_evidence_removal",
                                                "mae_increase_after_evidence_removal", "selected_ratio")}
        rows.append({"variant": variant, **{k: metrics[k] for k in ("accuracy", "f1_macro", "mae", "pearson")}, **fidelity})
        pd.DataFrame(rows).to_csv(output / "ablation_table.csv", index=False)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q3_aligned.yaml")
    args = p.parse_args()
    run_ablation(load_config(args.config))
