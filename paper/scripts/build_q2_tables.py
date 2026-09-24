from common import PAPER, SOURCES, read_csv, write_csv, write_table


def main():
    ablation = read_csv("q2_ablation")
    out = []
    for row in ablation:
        for scene in ("clean", "missing"):
            out.append({"experiment": row["experiment"], "scene": scene,
                        "accuracy": float(row[f"{scene}_accuracy"]), "macro_f1": float(row[f"{scene}_macro_f1"]),
                        "mae": float(row[f"{scene}_mae"]), "pearson": float(row[f"{scene}_pearson"]),
                        "final": "YES" if row["experiment"] == "R3_reconstruction" else "NO"})
    fields = ["experiment", "scene", "accuracy", "macro_f1", "mae", "pearson", "final"]
    write_csv(PAPER / "tables/q2_ablation.csv", out, fields)
    write_table(PAPER / "tables/table_q2_ablation.md", "Q2 R0–R5 消融", out, fields,
                ["实验", "输入", "Accuracy", "Macro-F1", "MAE", "Pearson", "最终"], ["q2_ablation"],
                "R5 的某些单项缺失指标更高；最终选择依据见 `E题/Q2/Q2_BEST_MODEL_ABLATION.md`。")
    mapping = [
        ("modality", "missing_modality", "missing_modality", "missing_type", "缺失模态"),
        ("ratio", "missing_ratio", "missing_ratio", "missing_ratio", "缺失比例"),
        ("position", "missing_position", "missing_position", "missing_position", "缺失位置"),
        ("length", "missing_length", "missing_length", "missing_duration", "缺失时长"),
    ]
    for dimension, key, plot, table, label in mapping:
        rows = read_csv("q2_" + key)
        for row in rows:
            for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                row[metric] = float(row[metric])
            row["source_file"] = SOURCES["q2_" + key]
        plot_fields = [dimension, "accuracy", "macro_f1", "mae", "pearson", "source_file"]
        write_csv(PAPER / f"generated/q2_{plot}_plot.csv", rows, plot_fields)
        write_table(PAPER / f"tables/table_q2_{table}.md", f"Q2 {label}分析", rows, plot_fields[:-1],
                    [label, "Accuracy", "Macro-F1", "MAE", "Pearson"], ["q2_" + key],
                    "同一最终 R3 模型在验证集上的人工连续缺失条件。")


if __name__ == "__main__":
    main()
