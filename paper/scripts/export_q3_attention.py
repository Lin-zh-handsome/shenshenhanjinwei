"""Export actual Q3 temporal self-attention for one saved Attachment 4 sample.

This script only loads an existing checkpoint and runs one forward pass. The
result is an internal attention visualization, not a faithfulness explanation.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MODALITIES = ("text", "audio", "vision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--sample-id", default="07")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.q3_dir.resolve()))
    from data.attachment4_dataset import Attachment4AlignedDataset
    from utils.io import load_checkpoint, load_config

    cfg = load_config(args.config)
    model, _ = load_checkpoint(args.checkpoint, cfg, torch.device("cpu"))
    dataset = Attachment4AlignedDataset(args.feature_dir, args.video_dir)
    sample = next((dataset[i] for i, path in enumerate(dataset.files)
                   if path.stem == args.sample_id), None)
    if sample is None:
        raise ValueError(f"Sample {args.sample_id} was not found")

    valid = sample["valid_mask"].unsqueeze(0)
    batch = {name: sample[name].unsqueeze(0) for name in MODALITIES}
    batch["valid_mask"] = valid
    captured = {}
    handles = []
    for name in MODALITIES:
        layer = model.encoders[name].encoder.layers[-1]

        def capture(_module, inputs, modality=name):
            captured[modality] = inputs[0].detach()

        handles.append(layer.register_forward_pre_hook(capture))

    with torch.inference_mode():
        model(**batch, temperature=model.inference_temperature)
        for handle in handles:
            handle.remove()

        indices = torch.where(valid[0])[0]
        matrices = {}
        for name in MODALITIES:
            layer = model.encoders[name].encoder.layers[-1]
            x = layer.norm1(captured[name])  # norm_first=True in the final model
            _, weights = layer.self_attn(
                x, x, x, key_padding_mask=~valid,
                need_weights=True, average_attn_weights=False,
            )
            matrices[name] = weights[0][:, indices][:, :, indices].cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = indices.cpu().numpy()
    for name, heads in matrices.items():
        destination = args.output_dir / f"q3_attention_{args.sample_id}_{name}_mean.csv"
        with destination.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["query_grid\\key_grid", *labels.tolist()])
            writer.writerows([[int(labels[i]), *row.tolist()]
                              for i, row in enumerate(heads.mean(axis=0))])

    means = [matrices[name].mean(axis=0) for name in MODALITIES]
    shared_max = max(float(matrix.max()) for matrix in means)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    for axis, name, matrix in zip(axes, MODALITIES, means):
        image = axis.imshow(matrix, origin="lower", cmap="viridis",
                            vmin=0, vmax=shared_max, aspect="auto")
        axis.set_title(f"{name.title()} | 4-head mean")
        axis.set_xlabel("Key grid index")
    axes[0].set_ylabel("Query grid index")
    fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.78)
    fig.savefig(args.output_dir / f"q3_attention_{args.sample_id}_modalities.png", dpi=200)
    plt.close(fig)

    text_heads = matrices["text"]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.2), constrained_layout=True)
    shared_max = float(text_heads.max())
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(text_heads[index], origin="lower", cmap="viridis",
                            vmin=0, vmax=shared_max, aspect="auto")
        axis.set_title(f"Text head {index + 1}")
        axis.set_xlabel("Key grid index")
        axis.set_ylabel("Query grid index")
    fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.78)
    fig.savefig(args.output_dir / f"q3_attention_{args.sample_id}_text_heads.png", dpi=200)
    plt.close(fig)
    print(f"Exported sample {args.sample_id}, {len(labels)} valid positions, "
          f"last temporal encoder layer, four heads per modality")


if __name__ == "__main__":
    main()
