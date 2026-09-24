import argparse
from pathlib import Path

from faithfulness_eval_v2 import run_faithfulness
from utils.io import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-base", required=True)
    parser.add_argument("--config-position", required=True)
    parser.add_argument("--config-pool", required=True)
    parser.add_argument("--config-regularized", required=True)
    parser.add_argument("--config-router", required=True)
    args = parser.parse_args()
    jobs = [
        ("B1", args.config_base, "01_a1_reproduction", (0.15, 0.20, 0.25)),
        ("B2", args.config_position, "02_position", (0.25,)),
        ("B3", args.config_pool, "03_gate_pooling", (0.15, 0.20, 0.25)),
        ("B4", args.config_regularized, "04_gate_regularized", (0.25,)),
        ("B5", args.config_router, "05_router_diagnostic", (0.25,)),
    ]
    root = Path("outputs/q3_v2")
    for variant, config_path, folder, budgets in jobs:
        cfg = load_config(config_path)
        checkpoint = root / folder / "best_joint.pt"
        for budget in budgets:
            label = f"budget_{round(budget * 100):03d}"
            output = root / "faithfulness" / variant / label
            if (output / "valid_faithfulness_v2.json").is_file():
                continue
            print(variant, label, flush=True)
            run_faithfulness(cfg, checkpoint, output, budget=budget)


if __name__ == "__main__":
    main()
