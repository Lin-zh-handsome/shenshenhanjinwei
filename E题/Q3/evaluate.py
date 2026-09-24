import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from utils.io import batch_to_device, get_device, load_checkpoint, load_config, model_inputs, save_json
from utils.metrics import compute_metrics
from utils.plotting import confusion_figure, regression_figure


@torch.no_grad()
def evaluate_split(cfg, checkpoint, split, output_dir=None):
    output = Path(output_dir or cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg)
    model, _ = load_checkpoint(checkpoint, cfg, device)
    ds = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], split)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], num_workers=cfg["runtime"]["num_workers"])
    rows = []
    for batch in loader:
        original = batch
        batch = batch_to_device(batch, device)
        out = model(**model_inputs(batch), temperature=model.inference_temperature)
        prob = out["cls_prob"].float().cpu().numpy()
        reg = out["reg_pred"].float().cpu().numpy()
        router = out["router_alpha"].float().cpu().numpy()
        for i, sample_id in enumerate(original["id"]):
            rows.append({"id": sample_id, "true_class": int(original["y_cls"][i]),
                         "pred_class": int(prob[i].argmax()), "true_reg": float(original["y_reg"][i]),
                         "pred_reg": float(reg[i]), "prob_negative": float(prob[i, 0]),
                         "prob_neutral": float(prob[i, 1]), "prob_positive": float(prob[i, 2]),
                         "router_text": float(router[i, 0]), "router_audio": float(router[i, 1]),
                         "router_vision": float(router[i, 2])})
    table = pd.DataFrame(rows)
    metrics = compute_metrics(table.true_class, table[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy(), table.true_reg, table.pred_reg)
    table.to_csv(output / f"{split}_predictions.csv", index=False)
    save_json(output / f"{split}_metrics.json", metrics)
    if split == "test":
        confusion_figure(table.true_class, table.pred_class, output / "fig_confusion_matrix.png")
        regression_figure(table.true_reg, table.pred_reg, output / "fig_regression_scatter.png")
    print(split, metrics, flush=True)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q3_aligned.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["valid", "test"], required=True)
    args = p.parse_args()
    evaluate_split(load_config(args.config), args.checkpoint, args.split)
