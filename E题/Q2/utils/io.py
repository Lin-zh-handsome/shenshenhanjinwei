from pathlib import Path
import json
import yaml
import torch


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def output_dir(cfg):
    path = Path(cfg['paths']['output_dir'])
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def device_for(cfg):
    return torch.device('cuda' if cfg.get('device') == 'auto' and torch.cuda.is_available() else cfg.get('device', 'cpu'))


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)
