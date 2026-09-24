import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from data.feature_baseline import (compute_train_position_means, load_feature_means,
                                   load_position_means)
from explain.modality_ablation import MODALITIES, modality_importance
from utils.io import batch_to_device, get_device, load_checkpoint, load_config, save_json


@torch.no_grad()
def run(cfg, checkpoint, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(cfg["paths"]["output_dir"])
    position_file = baseline_dir / "train_feature_position_baselines.npz"
    if not position_file.is_file():
        train = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "train")
        compute_train_position_means(train, position_file)
    global_means = load_feature_means(baseline_dir / "train_feature_baselines.npz")
    position_means = load_position_means(position_file)
    device = get_device(cfg)
    model, _ = load_checkpoint(checkpoint, cfg, device)
    valid = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "valid")
    rows = []
    for original in DataLoader(valid, batch_size=cfg["training"]["batch_size"], shuffle=False):
        batch = batch_to_device(original, device)
        a = modality_importance(model, batch, global_means, cfg)["importance"].argmax(dim=1).cpu().tolist()
        b = modality_importance(model, batch, position_means, cfg)["importance"].argmax(dim=1).cpu().tolist()
        rows.extend({"sample": sample, "main_global": MODALITIES[x], "main_position": MODALITIES[y],
                     "agreement": bool(x == y)} for sample, x, y in zip(original["id"], a, b))
    pd.DataFrame(rows).to_csv(output_dir / "lomo_baseline_stability.csv", index=False)
    summary = {"sample_count": len(rows), "agreement_rate": sum(r["agreement"] for r in rows) / len(rows),
               "position_baseline_source": "train_only"}
    save_json(output_dir / "lomo_baseline_stability.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.checkpoint, args.output_dir)
