from collections import Counter

from common import PAPER, SOURCES, read_csv, read_json, write_csv, write_table


def main():
    ablation = read_csv("q3_ablation")
    for row in ablation:
        for metric in ("accuracy", "macro_f1", "mae", "pearson", "keep_only_macro_f1", "remove_macro_f1", "random_remove_macro_f1"):
            if row.get(metric):
                row[metric] = float(row[metric])
    write_table(PAPER / "tables/table_q3_ablation.md", "Q3 v2 消融", ablation,
                ["variant", "accuracy", "macro_f1", "mae", "pearson", "keep_only_macro_f1", "remove_macro_f1", "random_remove_macro_f1"],
                ["版本", "Accuracy", "Macro-F1", "MAE", "Pearson", "仅保留证据 F1", "删除证据 F1", "随机删除 F1"],
                ["q3_ablation"], "最终 B1 + 诊断 Router 单独记录；B5 是 B4 加诊断 Router，不代表最终预测器。")
    faith = read_json("q3_faithfulness")
    scenes = [("Full", "full"), ("Keep-only Evidence", "sufficiency"),
              ("Remove Evidence", "comprehensiveness"), ("Random Remove", "random_remove")]
    rows = [{"condition": name, "accuracy": faith.get(prefix + "_accuracy", ""),
             "macro_f1": faith.get(prefix + "_macro_f1", ""), "mae": faith.get(prefix + "_mae", ""),
             "pearson": faith.get(prefix + "_pearson", ""),
             "confidence_drop": faith.get("confidence_drop_top" if prefix == "comprehensiveness" else "confidence_drop_random", "")
             if prefix in ("comprehensiveness", "random_remove") else "",
             "source_file": SOURCES["q3_faithfulness"]} for name, prefix in scenes]
    write_csv(PAPER / "generated/q3_faithfulness_plot.csv", rows,
              ["condition", "accuracy", "macro_f1", "mae", "pearson", "confidence_drop", "source_file"])
    write_table(PAPER / "tables/table_q3_faithfulness.md", "Q3 最终模型解释忠实性", rows,
                ["condition", "accuracy", "macro_f1", "mae", "pearson", "confidence_drop"],
                ["条件", "Accuracy", "Macro-F1", "MAE", "Pearson", "置信度下降"], ["q3_faithfulness"],
                "验证集、预算 0.25；回归 MAE 删除证据后仅小幅上升，不宜夸大。")
    explanations = read_csv("q3_explanations")
    counts = Counter(row.get("main_modality", "") for row in explanations)
    importance = []
    for modality in ("text", "audio", "vision"):
        values = [float(row[f"importance_{modality}"]) for row in explanations if row.get(f"importance_{modality}")]
        importance.append({"modality": modality, "main_count": counts[modality],
                           "main_fraction": counts[modality] / len(explanations) if explanations else "",
                           "mean_lomo_importance": sum(values) / len(values) if values else "",
                           "source_file": SOURCES["q3_explanations"]})
    fields = ["modality", "main_count", "main_fraction", "mean_lomo_importance", "source_file"]
    write_csv(PAPER / "generated/q3_modality_importance_plot.csv", importance, fields)
    write_table(PAPER / "tables/table_q3_modality_importance.md", "Q3 验证集模态贡献", importance, fields[:-1],
                ["模态", "主模态样本数", "主模态比例", "平均 LOMO 贡献"], ["q3_explanations"],
                "由已保存的逐样本 LOMO 结果聚合；不是 Router 权重。")


if __name__ == "__main__":
    main()
