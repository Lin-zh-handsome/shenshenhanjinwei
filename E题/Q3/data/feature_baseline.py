from pathlib import Path

import numpy as np


def compute_train_feature_means(dataset, output_path):
    sums = {k: np.zeros(w, dtype=np.float64) for k, w in (("text", 768), ("audio", 74), ("vision", 35))}
    count = 0
    for i in range(len(dataset)):
        mask = dataset.masks[i]
        count += int(mask.sum())
        for k in sums:
            sums[k] += dataset.data[k][i, mask].sum(axis=0, dtype=np.float64)
    means = {f"{k}_mean": (s / count).astype(np.float32) for k, s in sums.items()}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **means)
    return means


def load_feature_means(path):
    with np.load(path) as d:
        return {k.removesuffix("_mean"): d[k] for k in d.files}


def compute_train_position_means(dataset, output_path):
    """Use the train split only; each valid grid position has its own mean."""
    sums = {k: np.zeros((50, w), dtype=np.float64) for k, w in
            (("text", 768), ("audio", 74), ("vision", 35))}
    counts = np.zeros(50, dtype=np.int64)
    for i in range(len(dataset)):
        mask = dataset.masks[i]
        counts += mask.astype(np.int64)
        for k in sums:
            sums[k][mask] += dataset.data[k][i, mask]
    means = {}
    for k, values in sums.items():
        global_mean = values.sum(axis=0) / max(1, counts.sum())
        position = values / np.maximum(counts[:, None], 1)
        position[counts == 0] = global_mean
        means[k] = position.astype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **means)
    return means


def load_position_means(path):
    with np.load(path) as data:
        return {k: data[k] for k in ("text", "audio", "vision")}
