import json
from pathlib import Path

import torch
import yaml


MODEL_INPUTS = ("text", "audio", "vision", "valid_mask")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]).resolve())
    return cfg


def get_device(cfg):
    preference = cfg.get("device", "auto")
    return torch.device("cuda" if preference == "auto" and torch.cuda.is_available() else "cpu" if preference == "auto" else preference)


def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def model_inputs(batch):
    return {k: batch[k] for k in MODEL_INPUTS}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_checkpoint(path, model, cfg, variant, epoch, metrics, temperature):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "variant": variant, "epoch": epoch, "metrics": metrics, "temperature": temperature}, path)


def load_checkpoint(path, cfg, device):
    from models.fer_msa import FERMSA
    saved = torch.load(path, map_location=device, weights_only=False)
    model = FERMSA(cfg, variant=saved.get("variant", "A5")).to(device)
    state = saved["model"]
    if any(key.startswith("router.fuse.") for key in state):
        state = {("fusion." + key.removeprefix("router.")) if key.startswith("router.fuse.") else key: value
                 for key, value in state.items()}
    model.load_state_dict(state)
    epoch = saved.get("epoch", cfg["training"]["epochs"])
    start = cfg["model"]["gate_temperature_start"]
    end = cfg["model"]["gate_temperature_end"]
    inferred = start if epoch <= 5 else start + (end - start) * min(1.0, (epoch - 5) / max(1, cfg["training"]["epochs"] - 5))
    model.inference_temperature = saved.get("temperature", inferred)
    model.eval()
    return model, saved
