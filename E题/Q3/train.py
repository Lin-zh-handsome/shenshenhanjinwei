import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.aligned_dataset import AlignedMoseiDataset
from data.feature_baseline import compute_train_feature_means
from losses.explainable_multitask_loss import ExplainableMultitaskLoss
from models.fer_msa import FERMSA
from utils.io import batch_to_device, get_device, load_config, model_inputs, save_checkpoint
from utils.metrics import compute_metrics, selection_score
from utils.seed import set_seed


@torch.no_grad()
def evaluate_loader(model, loader, device, temperature):
    model.eval()
    labels, probs, targets, predictions, gates = [], [], [], [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(**model_inputs(batch), temperature=temperature)
        labels.extend(batch["y_cls"].cpu().tolist())
        probs.extend(out["cls_prob"].float().cpu().numpy())
        targets.extend(batch["y_reg"].cpu().tolist())
        predictions.extend(out["reg_pred"].float().cpu().tolist())
        if model.variant != "A0":
            selected = torch.stack([out[f"gate_{k}"] for k in ("text", "audio", "vision")], dim=1)
            gates.extend(selected[batch["valid_mask"][:, None, :].expand_as(selected)].float().cpu().tolist())
    result = compute_metrics(labels, probs, targets, predictions)
    result["gate_mean"] = float(np.mean(gates)) if gates else None
    result["gate_std"] = float(np.std(gates)) if gates else None
    return result


def train_model(cfg, variant="A5", output_dir=None, seed=None):
    if seed is None:
        seed = cfg["seed"]
    set_seed(seed)
    output = Path(output_dir or cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_data = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "train")
    valid_data = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "valid")
    if variant == "A5":
        compute_train_feature_means(train_data, output / "train_feature_baselines.npz")
    batch_size = cfg["training"]["batch_size"]
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=cfg["runtime"]["num_workers"])
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=cfg["runtime"]["num_workers"])
    device = get_device(cfg)
    model = FERMSA(cfg, variant).to(device)
    counts = np.bincount(train_data.cls, minlength=3)
    class_weights = len(train_data) / (3 * counts)
    criterion = ExplainableMultitaskLoss(cfg, class_weights, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    steps = cfg["training"]["epochs"] * len(loader)
    warmup = int(steps * cfg["training"]["warmup_ratio"])
    def lr_factor(step):
        return (step + 1) / max(1, warmup) if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = bool(cfg["runtime"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = {"cls": -float("inf"), "reg": float("inf"), "joint": -float("inf")}
    history = []
    wait = 0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_losses = []
        start = cfg["model"]["gate_temperature_start"]
        end = cfg["model"]["gate_temperature_end"]
        temperature = start if epoch < 5 else start + (end - start) * min(1.0, (epoch - 4) / max(1, cfg["training"]["epochs"] - 5))
        for batch in tqdm(loader, desc=f"{variant} epoch {epoch + 1}", leave=False):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(**model_inputs(batch), temperature=temperature)
                loss, _ = criterion(out, batch, epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.detach()))
        metrics = evaluate_loader(model, valid_loader, device, temperature)
        score = selection_score(metrics, cfg["selection"])
        row = {"epoch": epoch + 1, "variant": variant, "train_loss": float(np.mean(epoch_losses)), "temperature": temperature, "joint_score": score, **metrics}
        history.append(row)
        print(row, flush=True)
        if metrics["f1_macro"] > best["cls"]:
            best["cls"] = metrics["f1_macro"]
            save_checkpoint(output / "best_cls.pt", model, cfg, variant, epoch + 1, metrics, temperature)
        if metrics["mae"] < best["reg"]:
            best["reg"] = metrics["mae"]
            save_checkpoint(output / "best_reg.pt", model, cfg, variant, epoch + 1, metrics, temperature)
        if score > best["joint"]:
            best["joint"] = score
            wait = 0
            save_checkpoint(output / "best_joint.pt", model, cfg, variant, epoch + 1, metrics, temperature)
        else:
            wait += 1
        pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
        if wait >= cfg["training"]["early_stop_patience"]:
            break
    return history, best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q3_aligned.yaml")
    p.add_argument("--variant", choices=[f"A{i}" for i in range(6)], default="A5")
    p.add_argument("--output-dir")
    args = p.parse_args()
    train_model(load_config(args.config), args.variant, args.output_dir)
