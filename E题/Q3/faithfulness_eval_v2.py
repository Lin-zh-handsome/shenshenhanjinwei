import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.aligned_dataset import AlignedMoseiDataset
from data.feature_baseline import load_feature_means
from explain.evidence_segments import evidence_spans
from explain.local_occlusion import local_evidence_scores
from explain.modality_ablation import MODALITIES, modality_importance
from utils.io import batch_to_device, get_device, load_checkpoint, load_config, model_inputs, save_json
from utils.metrics import compute_metrics


def selected_positions(evidence, valid_len, cfg):
    candidates = np.asarray(evidence["candidate_positions"], dtype=np.int64)
    k = min(len(candidates), max(1, math.ceil(cfg["explanation"]["evidence_ratio"] * valid_len)))
    scores = np.asarray(evidence["score"])
    return candidates[np.argsort(scores[candidates])[-k:]].tolist()


def matched_random_positions(selected, length, rng):
    """Match the selected position count and contiguous span lengths per modality."""
    ordered = sorted(selected)
    runs = []
    for pos in ordered:
        if not runs or pos != runs[-1][-1] + 1:
            runs.append([pos])
        else:
            runs[-1].append(pos)
    occupied = set()
    for run in sorted(runs, key=len, reverse=True):
        width = len(run)
        starts = [s for s in range(length - width + 1)
                  if all(t not in occupied for t in range(s, s + width))]
        if not starts:
            remaining = [t for t in range(length) if t not in occupied]
            occupied.update(rng.choice(remaining, size=width, replace=False).tolist())
        else:
            start = int(rng.choice(starts))
            occupied.update(range(start, start + width))
    return sorted(occupied)


def _variant(batch, means, positions, keep=False):
    result = {k: batch[k].clone() for k in MODALITIES}
    result["valid_mask"] = batch["valid_mask"]
    length = int(batch["valid_mask"][0].sum())
    for modality in MODALITIES:
        chosen = set(positions[modality])
        targets = [i for i in range(length) if i not in chosen] if keep else list(chosen)
        if targets:
            result[modality][0, targets] = torch.as_tensor(means[modality], dtype=result[modality].dtype, device=result[modality].device)
    return result


def _batch_variants(variants):
    return {k: torch.cat([v[k] for v in variants], dim=0) for k in (*MODALITIES, "valid_mask")}


def _safe_corr(func, x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value = float(func(x, y).statistic)
    return value if np.isfinite(value) else 0.0


@torch.no_grad()
def run_faithfulness(cfg, checkpoint, output_dir=None, write_files=True, budget=None):
    output = Path(output_dir or cfg["paths"]["output_dir"])
    if budget is not None:
        cfg = dict(cfg)
        cfg["explanation"] = {**cfg["explanation"], "evidence_ratio": budget}
    device = get_device(cfg)
    model, _ = load_checkpoint(checkpoint, cfg, device)
    ds = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "valid")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    means = load_feature_means(Path(cfg["paths"]["output_dir"]) / "train_feature_baselines.npz")
    rng = np.random.default_rng(cfg["seed"])
    base_rows, keep_rows, remove_rows, random_rows, curve_rows, detail_rows = [], [], [], [], [], []
    spans_all, selected_ratios = [], []
    curve_steps = cfg["explanation"]["deletion_steps"]
    for original in tqdm(loader, desc="valid faithfulness"):
        batch = batch_to_device(original, device)
        sample_id = original["id"][0]
        length = int(batch["valid_mask"][0].sum())
        lomo = modality_importance(model, batch, means, cfg)
        full = lomo["full"]
        local = local_evidence_scores(model, batch, means, cfg, full)
        selected = {m: selected_positions(local[m], length, cfg) for m in MODALITIES}
        selected_ratios.append(sum(len(x) for x in selected.values()) / (3 * length))
        spans = {m: evidence_spans(m, local[m], length, cfg, float(full["reg_pred"][0])) for m in MODALITIES}
        spans_all.extend(s for group in spans.values() for s in group)
        variant_names = ["keep", "remove"]
        variants = [_variant(batch, means, selected, keep=True), _variant(batch, means, selected)]
        for repeat in range(cfg["explanation"]["random_control_repeats"]):
            random_positions = {m: matched_random_positions(selected[m], length, rng) for m in MODALITIES}
            variants.append(_variant(batch, means, random_positions))
            variant_names.append(f"random_{repeat}")
        for step in curve_steps:
            positions = {m: np.argsort(-np.asarray(local[m]["score"])[:length], kind="stable")[:math.ceil(step * length)].tolist() for m in MODALITIES}
            variants.append(_variant(batch, means, positions))
            variant_names.append(f"curve_{step}")
        perturbed = model(**_batch_variants(variants), temperature=model.inference_temperature)
        predicted_class = int(full["cls_prob"][0].argmax())
        true_class = int(batch["y_cls"][0])
        true_reg = float(batch["y_reg"][0])
        base_p = full["cls_prob"][0].float().cpu().numpy()
        base_y = float(full["reg_pred"][0])
        importance = lomo["importance"][0].float().cpu().numpy()
        router = full["router_alpha"][0].float().cpu().numpy()
        base_rows.append({"id": sample_id, "true_class": true_class, "true_reg": true_reg,
                          "pred_class": predicted_class, "pred_reg": base_y, "prob": base_p,
                          "importance": importance, "router": router})
        for i, name in enumerate(variant_names):
            p = perturbed["cls_prob"][i].float().cpu().numpy()
            y = float(perturbed["reg_pred"][i])
            row = {"id": sample_id, "true_class": true_class, "true_reg": true_reg,
                   "pred_class": int(p.argmax()), "pred_reg": y, "prob": p,
                   "predicted_class_probability": float(p[predicted_class]),
                   "confidence_drop": float(base_p[predicted_class] - p[predicted_class]),
                   "reg_shift": abs(base_y - y)}
            if name == "keep":
                keep_rows.append(row)
            elif name == "remove":
                remove_rows.append(row)
            elif name.startswith("random_"):
                random_rows.append({"id": sample_id, "repeat": int(name.split("_")[1]),
                                    "top_confidence_drop": None, **row})
            else:
                curve_rows.append({"deletion_fraction": float(name.split("_")[1]), **row})
        detail_rows.append({"id": sample_id, "true_class": true_class, "pred_class": predicted_class,
                            "true_reg": true_reg, "pred_reg": base_y,
                            "full_confidence": float(base_p[predicted_class]),
                            **{f"importance_{m}": float(importance[j]) for j, m in enumerate(MODALITIES)},
                            **{f"router_{m}": float(router[j]) for j, m in enumerate(MODALITIES)},
                            "main_modality": MODALITIES[int(importance.argmax())],
                            "main_evidence_span": str(spans[MODALITIES[int(importance.argmax())]][0]) if spans[MODALITIES[int(importance.argmax())]] else "",
                            "error_score": 1.5 * (predicted_class != true_class) + abs(base_y - true_reg),
                            "removed_evidence_confidence": remove_rows[-1]["predicted_class_probability"]})
    def metrics(rows):
        return compute_metrics([r["true_class"] for r in rows], [r["prob"] for r in rows], [r["true_reg"] for r in rows], [r["pred_reg"] for r in rows])
    full_metrics, keep_metrics, remove_metrics = metrics(base_rows), metrics(keep_rows), metrics(remove_rows)
    router = np.stack([r["router"] for r in base_rows])
    importance = np.stack([r["importance"] for r in base_rows])
    alignment = {m: {"pearson": _safe_corr(pearsonr, router[:, i], importance[:, i]),
                     "spearman": _safe_corr(spearmanr, router[:, i], importance[:, i])} for i, m in enumerate(MODALITIES)}
    alignment["rank_agreement"] = float(np.mean(np.all(np.argsort(router, axis=1) == np.argsort(importance, axis=1), axis=1)))
    alignment["main_modality_agreement"] = float(np.mean(router.argmax(axis=1) == importance.argmax(axis=1)))
    curve = []
    for step in curve_steps:
        subset = [r for r in curve_rows if r["deletion_fraction"] == step]
        m = metrics(subset)
        curve.append({"deletion_fraction": step, "predicted_class_probability": float(np.mean([r["predicted_class_probability"] for r in subset])), **m})
    random_mean = float(np.mean([r["confidence_drop"] for r in random_rows]))
    top_mean = float(np.mean([r["confidence_drop"] for r in remove_rows]))
    random_metrics = [metrics([r for r in random_rows if r["repeat"] == repeat])
                      for repeat in range(cfg["explanation"]["random_control_repeats"])]
    random_metrics_mean = {key: float(np.mean([item[key] for item in random_metrics])) for key in full_metrics}
    summary = {
        "evidence_budget": float(cfg["explanation"]["evidence_ratio"]),
        "full_metrics": full_metrics, "keep_only_metrics": keep_metrics, "evidence_removed_metrics": remove_metrics,
        "suff_cls": float(np.mean([abs(b["prob"][b["pred_class"]] - k["prob"][b["pred_class"]]) for b, k in zip(base_rows, keep_rows)])),
        "suff_reg": float(np.mean([abs(b["pred_reg"] - k["pred_reg"]) for b, k in zip(base_rows, keep_rows)])),
        "comp_cls": top_mean, "comp_reg_shift": float(np.mean([r["reg_shift"] for r in remove_rows])),
        "f1_drop_after_evidence_removal": full_metrics["f1_macro"] - remove_metrics["f1_macro"],
        "mae_increase_after_evidence_removal": remove_metrics["mae"] - full_metrics["mae"],
        "random_confidence_drop": random_mean, "random_deletion_metrics_mean": random_metrics_mean,
        "top_minus_random_confidence_drop": top_mean - random_mean,
        "top_deletion_beats_random": bool(top_mean > random_mean),
        "top_vs_random_f1_drop": random_metrics_mean["f1_macro"] - remove_metrics["f1_macro"],
        "top_vs_random_mae_increase": remove_metrics["mae"] - random_metrics_mean["mae"],
        "selected_ratio": float(np.mean(selected_ratios)),
        "mean_span_count": float(len(spans_all) / (len(ds) * 3)),
        "mean_span_length": float(np.mean([s["length"] for s in spans_all])),
        "router_vs_lomo": alignment, "deletion_curve": curve,
        "deletion_auc_confidence": float(np.trapezoid([r["predicted_class_probability"] for r in curve], curve_steps)),
    }
    for name, values in (("full", full_metrics), ("sufficiency", keep_metrics),
                         ("comprehensiveness", remove_metrics), ("random_remove", random_metrics_mean)):
        for key in ("accuracy", "f1_macro", "mae", "pearson"):
            summary[f"{name}_{'macro_f1' if key == 'f1_macro' else key}"] = values[key]
    for key in ("accuracy", "f1_macro", "mae", "pearson"):
        label = "macro_f1" if key == "f1_macro" else key
        summary[f"sufficiency_delta_{label}"] = keep_metrics[key] - full_metrics[key]
    summary.update(
        confidence_drop_top=top_mean,
        confidence_drop_random=random_mean,
        macro_f1_drop_top=full_metrics["f1_macro"] - remove_metrics["f1_macro"],
        macro_f1_drop_random=full_metrics["f1_macro"] - random_metrics_mean["f1_macro"],
        mae_increase_top=remove_metrics["mae"] - full_metrics["mae"],
        mae_increase_random=random_metrics_mean["mae"] - full_metrics["mae"],
        regression_comprehensiveness_failed=bool(remove_metrics["mae"] < full_metrics["mae"]),
    )
    if write_files:
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(random_rows).assign(top_confidence_drop=lambda d: d["id"].map({r["id"]: r["confidence_drop"] for r in remove_rows})).drop(columns=["prob"]).to_csv(output / "valid_faithfulness_random_control.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(output / "valid_explanations.csv", index=False)
        pd.DataFrame(curve).to_csv(output / "valid_deletion_curve.csv", index=False)
        save_json(output / "valid_faithfulness_v2.json", summary)
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q3_aligned.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--budget", type=float, required=True)
    args = p.parse_args()
    run_faithfulness(load_config(args.config), args.checkpoint, args.output_dir, budget=args.budget)
