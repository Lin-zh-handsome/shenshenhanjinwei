import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from data.feature_baseline import load_feature_means
from explain.evidence_segments import evidence_spans
from explain.explanation_card import save_explanation_card
from explain.local_occlusion import local_evidence_scores
from explain.modality_ablation import MODALITIES, modality_importance
from utils.io import batch_to_device, get_device, load_checkpoint, load_config
from utils.plotting import save_plot


LABELS = ("Negative", "Neutral", "Positive")


@torch.no_grad()
def _test_importance(cfg, checkpoint, output, means):
    device = get_device(cfg)
    model, _ = load_checkpoint(checkpoint, cfg, device)
    dataset = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "test")
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=0)
    rows = []
    for original in loader:
        batch = batch_to_device(original, device)
        result = modality_importance(model, batch, means, cfg)
        importance = result["importance"].float().cpu().numpy()
        for i, sample_id in enumerate(original["id"]):
            rows.append({"id": sample_id, "true_class": int(original["y_cls"][i]),
                         **{f"importance_{m}": float(importance[i, j]) for j, m in enumerate(MODALITIES)}})
    table = pd.DataFrame(rows)
    table.to_csv(output / "test_importance.csv", index=False)
    return table


@torch.no_grad()
def _typical_valid_cards(cfg, checkpoint, output, means, valid):
    device = get_device(cfg)
    model, _ = load_checkpoint(checkpoint, cfg, device)
    dataset = AlignedMoseiDataset(cfg["paths"]["aligned_pkl"], "valid")
    by_id = {str(dataset.data["id"][i]): i for i in range(len(dataset))}
    correct = valid[valid.true_class == valid.pred_class]
    wrong = valid[valid.true_class != valid.pred_class]
    picks = {}
    if len(correct):
        picks["high_confidence_correct"] = correct.sort_values("full_confidence", ascending=False).iloc[0]
    delta = sum((valid[f"router_{m}"] - valid[f"importance_{m}"]).abs() for m in MODALITIES)
    picks["router_lomo_conflict"] = valid.loc[delta.idxmax()]
    if len(wrong):
        picks["error_case"] = wrong.sort_values("error_score", ascending=False).iloc[0]
    predictions = pd.read_csv(output / "valid_predictions.csv", dtype={"id": str}).set_index("id")
    for label, row in picks.items():
        sample_id = str(row["id"])
        sample = dataset[by_id[sample_id]]
        batch = {k: sample[k].unsqueeze(0).to(device) for k in (*MODALITIES, "valid_mask")}
        lomo = modality_importance(model, batch, means, cfg)
        local = local_evidence_scores(model, batch, means, cfg, lomo["full"])
        length = int(batch["valid_mask"][0].sum())
        words = sample["raw_text"].split()
        spans = {m: evidence_spans(m, local[m], length, cfg) for m in MODALITIES}
        for span in spans["text"]:
            a = math.floor(span["grid_start"] / length * len(words))
            b = math.ceil(span["grid_end_exclusive"] / length * len(words))
            span["text_fragment_approx"] = " ".join(words[a:b])
        pred = predictions.loc[sample_id]
        card_prediction = {"pred_label": LABELS[int(pred["pred_class"])], "pred_intensity": float(pred["pred_reg"]),
                           "prob_negative": float(pred["prob_negative"]), "prob_neutral": float(pred["prob_neutral"]),
                           "prob_positive": float(pred["prob_positive"]), "main_modality": MODALITIES[int(lomo["importance"][0].argmax())]}
        save_explanation_card(output / "explanation_cards" / f"valid_{label}_{sample_id.replace('/', '_')}.png",
                              sample_id, card_prediction, lomo["importance"][0].float().cpu().numpy(), local, spans)


def export_tables(cfg):
    output = Path(cfg["paths"]["output_dir"])
    with (output / "valid_metrics.json").open(encoding="utf-8") as f:
        valid_metrics = json.load(f)
    with (output / "test_metrics.json").open(encoding="utf-8") as f:
        test_metrics = json.load(f)
    with (output / "valid_faithfulness.json").open(encoding="utf-8") as f:
        fidelity = json.load(f)
    ablation = pd.read_csv(output / "ablation_table.csv")
    valid = pd.read_csv(output / "valid_explanations.csv", dtype={"id": str})
    attachment4 = pd.read_csv(output / "attachment4_predictions_explanations.csv", dtype={"sample_id": str})
    mapping_notes = pd.read_csv(output / "attachment4_mapping_notes.csv", dtype={"sample_id": str})
    means = load_feature_means(output / "train_feature_baselines.npz")
    checkpoint = output / "best_joint.pt"
    test = _test_importance(cfg, checkpoint, output, means)
    groups = []
    for split, table in (("valid", valid), ("test", test)):
        for label, subset in table.groupby("true_class"):
            groups.append({"split": split, "true_class": int(label), "count": len(subset),
                           **{f"mean_importance_{m}": float(subset[f"importance_{m}"].mean()) for m in MODALITIES}})
    pd.DataFrame(groups).to_csv(output / "modality_importance_by_class.csv", index=False)
    error_columns = ["id", "true_class", "pred_class", "true_reg", "pred_reg", "error_score",
                     "importance_text", "importance_audio", "importance_vision", "main_modality",
                     "main_evidence_span", "full_confidence", "removed_evidence_confidence"]
    valid.sort_values("error_score", ascending=False)[error_columns].to_csv(output / "valid_error_attribution.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (split, table) in zip(axes, (("valid", valid), ("test", test))):
        ax.boxplot([table[f"importance_{m}"] for m in MODALITIES], tick_labels=[m.title() for m in MODALITIES])
        ax.set(title=split, ylabel="LOMO importance", ylim=(0, 1))
    save_plot(fig, output / "fig_modality_importance_distribution.png")
    curve = pd.DataFrame(fidelity["deletion_curve"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(curve.deletion_fraction, curve.predicted_class_probability, marker="o")
    axes[0].set(xlabel="Deleted fraction", ylabel="Predicted class probability")
    axes[1].plot(curve.deletion_fraction, curve.f1_macro, marker="o", label="Macro F1")
    axes[1].plot(curve.deletion_fraction, curve.mae, marker="s", label="MAE")
    axes[1].set(xlabel="Deleted fraction", ylabel="Metric")
    axes[1].legend()
    save_plot(fig, output / "fig_deletion_curve.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["Keep", "Remove", "Random remove"], [fidelity["suff_cls"], fidelity["comp_cls"], fidelity["random_confidence_drop"]])
    axes[0].set(ylabel="Mean predicted-class probability change")
    axes[1].bar(["Full", "Keep", "Remove"], [fidelity["full_metrics"]["f1_macro"], fidelity["keep_only_metrics"]["f1_macro"], fidelity["evidence_removed_metrics"]["f1_macro"]])
    axes[1].set(ylabel="Macro F1")
    save_plot(fig, output / "fig_sufficiency_comprehensiveness.png")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True)
    for ax, modality in zip(axes, MODALITIES):
        ax.scatter(valid[f"router_{modality}"], valid[f"importance_{modality}"], s=8, alpha=0.4)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        corr = fidelity["router_vs_lomo"][modality]["spearman"]
        ax.set(title=f"{modality.title()} | Spearman {corr:.2f}", xlabel="Router alpha", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("LOMO importance")
    save_plot(fig, output / "fig_router_vs_ablation.png")
    _typical_valid_cards(cfg, checkpoint, output, means, valid)
    means_valid = {m: float(valid[f"importance_{m}"].mean()) for m in MODALITIES}
    frame_count_mismatches = int(mapping_notes.warning.str.contains("reported_frames=", na=False).sum())
    a0 = ablation.loc[ablation.variant == "A0"].iloc[0]
    a5 = ablation.loc[ablation.variant == "A5"].iloc[0]
    lines = ["# Q3 FER-MSA 结果汇总", "", "## 标准性能", "",
             f"- valid: Accuracy {valid_metrics['accuracy']:.4f}, Macro F1 {valid_metrics['f1_macro']:.4f}, MAE {valid_metrics['mae']:.4f}, Pearson {valid_metrics['pearson']:.4f}。",
             f"- test: Accuracy {test_metrics['accuracy']:.4f}, Macro F1 {test_metrics['f1_macro']:.4f}, MAE {test_metrics['mae']:.4f}, Pearson {test_metrics['pearson']:.4f}。",
             "", "## 解释忠实性（valid）", "",
             f"- Sufficiency classification change: {fidelity['suff_cls']:.4f}; regression change: {fidelity['suff_reg']:.4f}。",
             f"- 删除 top evidence 后 Macro F1 变化: {fidelity['f1_drop_after_evidence_removal']:.4f}; MAE 增量: {fidelity['mae_increase_after_evidence_removal']:.4f}。",
             f"- Top 删除置信度变化 {fidelity['comp_cls']:.4f}，随机对照 {fidelity['random_confidence_drop']:.4f}；top 是否大于随机：{fidelity['top_deletion_beats_random']}。",
             f"- Top 删除后 Macro F1 {fidelity['evidence_removed_metrics']['f1_macro']:.4f}，随机删除平均 {fidelity['random_deletion_metrics_mean']['f1_macro']:.4f}；两种比较口径分别呈现。",
             f"- Deletion AUC（删除曲线下面积，分类置信度）{fidelity['deletion_auc_confidence']:.4f}。",
             f"- 平均已选位置比例 {fidelity['selected_ratio']:.4f}，每模态平均段数 {fidelity['mean_span_count']:.3f}。",
             f"- Router 与 LOMO 主模态一致率 {fidelity['router_vs_lomo']['main_modality_agreement']:.4f}。",
             "", "## 模态作用（valid LOMO 平均）", "",
             *(f"- {m}: {means_valid[m]:.4f}。" for m in MODALITIES),
             "", "## 消融与附件 4", "",
             f"- 已记录 {len(ablation)} 组 A0–A5 消融；详见 ablation_table.csv。",
             f"- A0 valid Macro F1 {a0['f1_macro']:.4f}，A5 valid Macro F1 {a5['f1_macro']:.4f}；完整模型在该指标上未超过 A0。",
             f"- 附件 4 aligned 无标签样本输出 {len(attachment4)} 行预测；这些样本不报告预测正确率。",
             f"- {frame_count_mismatches}/{len(mapping_notes)} 个附件 4 视频的报告帧数与实际可解码帧数不同；秒级回投按可解码帧数计算，并在 mapping notes 逐条记录。",
             "- 输入 grid 索引精确；秒级和文本回投采用均匀比例近似，缺少官方逐位置/逐词时间戳。",
             "- 解释指标不构成 ground-truth localization accuracy（真实标注定位准确率）。", ""]
    (output / "q3_results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported Q3 figures and report for {len(attachment4)} Attachment 4 samples", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q3_aligned.yaml")
    args = p.parse_args()
    export_tables(load_config(args.config))
