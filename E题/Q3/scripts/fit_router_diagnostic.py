import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from data.feature_baseline import load_feature_means
from explain.modality_ablation import modality_importance
from utils.io import batch_to_device, get_device, load_checkpoint, load_config, model_inputs


def fit_router(cfg, checkpoint, output_path, epochs=5):
    device = get_device(cfg)
    cfg = dict(cfg)
    cfg["model"] = {**cfg["model"], "router_diagnostic": True}
    model, saved = load_checkpoint(checkpoint, cfg, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.router.route.parameters():
        parameter.requires_grad_(True)
    means = load_feature_means(Path(cfg["paths"]["output_dir"]) / "train_feature_baselines.npz")
    ds = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "train")
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.router.route.parameters(), lr=2e-4)
    for epoch in range(epochs):
        losses = []
        for original in loader:
            batch = batch_to_device(original, device)
            with torch.no_grad():
                target = modality_importance(model, batch, means, cfg)["importance"].detach()
                joined = model(**model_inputs(batch), temperature=model.inference_temperature,
                               return_explain=True)["router_input"].detach()
            logits = model.router.route(joined)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"diagnostic epoch {epoch + 1}: loss={sum(losses) / len(losses):.6f}", flush=True)
    result = {key: saved[key] for key in ("variant", "epoch", "metrics", "temperature") if key in saved}
    result["model"] = model.state_dict()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_router(load_config(args.config), args.checkpoint, args.output)
