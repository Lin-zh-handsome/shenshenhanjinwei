import argparse
from pathlib import Path

from data.aligned_dataset import AlignedMoseiDataset
from data.feature_baseline import compute_train_feature_means, compute_train_position_means
from utils.io import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "train")
    output = Path(cfg["paths"]["output_dir"])
    compute_train_feature_means(dataset, output / "train_feature_baselines.npz")
    compute_train_position_means(dataset, output / "train_feature_position_baselines.npz")
    print(f"Train-only baselines: {len(dataset)} samples -> {output}", flush=True)


if __name__ == "__main__":
    main()
