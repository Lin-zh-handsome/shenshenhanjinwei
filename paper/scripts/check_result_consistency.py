"""Compare overlapping recorded metrics; report differences, never rewrite sources."""
from __future__ import annotations

import csv
import json
import re

from common import PAPER, ROOT, SOURCES, metric_block, read_csv, read_json, markdown_table


def main():
    issues = []
    def compare(name, source_key, left, other_key, right, metrics):
        for field in metrics:
            a, b = left.get(field), right.get(field)
            if a in (None, "") or b in (None, ""):
                continue
            if abs(float(a) - float(b)) > 0.0001:
                issues.append({"experiment": name, "metric": field, "source value": a,
                               "other value": b, "source": SOURCES[source_key], "other": SOURCES[other_key]})

    q2 = read_json("q2_validation")
    ab = next(row for row in read_csv("q2_ablation") if row["experiment"] == "R3_reconstruction")
    for scene, block in (("clean", q2["clean_valid"]), ("missing", q2["missing_valid"])):
        compare("Q2 R3 " + scene, "q2_validation", metric_block(block), "q2_ablation",
                {k: ab[f"{scene}_{k}"] for k in ("accuracy", "macro_f1", "mae", "pearson")},
                ("accuracy", "macro_f1", "mae", "pearson"))
    q3 = metric_block(read_json("q3_validation"))
    final = read_json("q3_faithfulness")
    compare("Q3 B1 final", "q3_validation", q3, "q3_faithfulness",
            {k: final["full_" + k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
            ("accuracy", "macro_f1", "mae", "pearson"))
    b1 = next(row for row in read_csv("q3_ablation") if row["variant"] == "B1")
    compare("Q3 B1 ablation", "q3_validation", q3, "q3_ablation", b1,
            ("accuracy", "macro_f1", "mae", "pearson"))
    # Compare the preserved human-written result rows at their displayed precision.
    for label, scene in (("R3 原始参数，clean valid", "clean_valid"), ("R3 原始参数，混合缺失 valid", "missing_valid")):
        path = ROOT / "E题/Q2/README_Q2.md"
        row = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| " + label + " |")), "")
        if row:
            values = [value.strip() for value in row.strip("|").split("|")][1:6]
            actual = q2[scene]
            for field, displayed in zip(("accuracy", "f1_macro", "f1_weighted", "mae", "pearson"), values):
                if round(float(actual[field]), 4) != round(float(displayed), 4):
                    issues.append({"experiment": "Q2 README " + scene, "metric": field, "source value": actual[field],
                                   "other value": displayed, "source": SOURCES["q2_validation"], "other": "E题/Q2/README_Q2.md"})
    path = ROOT / "E题/Q3/Q3_FAITHFULNESS_V2_REPORT.md"
    row = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| B1 |")), "")
    if row:
        values = [value.strip() for value in row.strip("|").split("|")][-4:]
        for field, displayed in zip(("accuracy", "macro_f1", "mae", "pearson"), values):
            if round(float(q3[field]), 4) != round(float(displayed), 4):
                issues.append({"experiment": "Q3 report B1", "metric": field, "source value": q3[field],
                               "other value": displayed, "source": SOURCES["q3_validation"], "other": "E题/Q3/Q3_FAITHFULNESS_V2_REPORT.md"})
    readme_note = "当前根 README 不手写指标；保留的 Q2 README 与 Q3 v2 报告按展示的四位小数核对。"
    body = "# 结果数值一致性\n\n比较 Q2 R3 metrics/消融表，以及 Q3 B1 验证、消融和最终 faithfulness 文件的重叠指标。容差 0.0001；源文件未修改。\n\n"
    body += markdown_table(issues, ["experiment", "metric", "source value", "other value", "source", "other"]) if issues else "未发现上述同一实验、同一场景的数值冲突。\n"
    body += "\n" + readme_note + "\n"
    (PAPER / "RESULT_INCONSISTENCIES.md").write_text(body, encoding="utf-8")
    print(f"Metric inconsistencies: {len(issues)}")


if __name__ == "__main__":
    main()
