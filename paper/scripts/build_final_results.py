from common import PAPER, SOURCES, metric_block, read_json, write_csv, write_table


def main():
    q2 = read_json("q2_validation")
    q3 = read_json("q3_validation")
    rows = [{"problem": "Q1", "model": "50-window feature extraction and alignment", "split": "attachment1_unlabeled",
             **{key: "" for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")},
             "source_file": SOURCES["q1_validation"], "notes": "feature and alignment audit; no sentiment prediction metrics"}]
    for split, block in (("clean_valid", q2["clean_valid"]), ("missing_valid", q2["missing_valid"])):
        rows.append({"problem": "Q2", "model": "R3_Temporal_Reconstruction", "split": split,
                     **metric_block(block), "source_file": SOURCES["q2_validation"],
                     "notes": "official valid; missing is deterministic synthetic mixed spans" if split == "missing_valid" else "official valid"})
    rows.append({"problem": "Q3", "model": "FER-MSA v2 B1 + diagnostic Router", "split": "valid",
                 **metric_block(q3), "source_file": SOURCES["q3_validation"],
                 "notes": "B1 predictor; Router does not affect prediction"})
    fields = ["problem", "model", "split", "accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "source_file", "notes"]
    write_csv(PAPER / "FINAL_RESULTS.csv", rows, fields)
    write_table(PAPER / "tables/table_q2_main_results.md", "Q2 最终验证结果", rows[1:3], fields[:8],
                ["问题", "模型", "场景", "Accuracy", "Macro-F1", "Weighted-F1", "MAE", "Pearson"], ["q2_validation"])
    write_table(PAPER / "tables/table_q3_main_results.md", "Q3 最终验证结果", rows[3:], fields[:8],
                ["问题", "模型", "场景", "Accuracy", "Macro-F1", "Weighted-F1", "MAE", "Pearson"], ["q3_validation"])


if __name__ == "__main__":
    main()
