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
