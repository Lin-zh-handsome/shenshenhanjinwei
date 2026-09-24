"""Document each existing repository file by role; unknown files are retained."""
from __future__ import annotations

import subprocess
from pathlib import Path

from common import PAPER, ROOT, SOURCES, markdown_table


def category(path: Path) -> str:
    n = path.name.lower()
    parts = [p.lower() for p in path.parts]
    if path.suffix.lower() in (".pt", ".pth", ".ckpt"):
        return "checkpoint"
    if "__pycache__" in parts or n.endswith(".pyc") or n in (".ds_store", "thumbs.db"):
        return "临时文件"
    if path.suffix.lower() == ".csv":
        return "CSV"
    if path.suffix.lower() in (".json", ".jsonl"):
        return "JSON"
    if path.suffix.lower() == ".md":
        return "Markdown 报告"
    if path.suffix.lower() in (".yaml", ".yml", ".toml", ".txt") and ("config" in parts or "config" in n or "requirement" in n):
        return "Config"
    if path.suffix.lower() == ".py":
        if "loss" in parts or "loss" in n: return "Loss"
        if "model" in parts or "model" in n: return "模型"
        if "data" in parts or "dataset" in n: return "数据读取"
        if "plot" in n or "figure" in n or "visual" in n: return "绘图脚本"
        if "ablation" in n: return "消融脚本"
        if "infer" in n: return "推理脚本"
        if "eval" in n or "valid" in n: return "评估脚本"
        if "train" in n: return "训练脚本"
        return "核心源码"
    if "output" in parts or "outputs" in parts or path.suffix.lower() in (".npz", ".png", ".pdf", ".svg", ".jpg"):
        return "实验结果"
    return "未分类"


def status(path: Path) -> str:
    p = path.as_posix()
    n = path.name.lower()
    if "__pycache__" in p or n.endswith(".pyc"):
        return "TEMP"
    if "/outputs/q3/" in p or "q3_aligned.yaml" in p or "faithfulness_eval.py" in p:
        return "HISTORICAL"
    if "oracle" in p.lower() or "diagnostic" in p.lower() or "legacy" in p.lower():
        return "LEGACY"
    if "R4_reliability" in p or "R5_consistency" in p or "B2/" in p or "B3/" in p or "B4/" in p or "B5/" in p:
        return "HISTORICAL"
    if p in SOURCES.values() or p in ("E题/Q1/q1_features.npz", "E题/Q3/outputs/q3_v2/final_model.pt"):
        return "PAPER_SOURCE"
    if "/outputs/" in p or p.startswith("E题/Q1/") and path.suffix.lower() in (".csv", ".npz", ".json", ".jsonl"):
        return "FINAL_RESULT" if ("R3_reconstruction" in p or "q3_v2/attachment4/" in p or "E题/Q1/" in p) else "HISTORICAL"
    if path.suffix.lower() == ".py" or "/configs/" in p or n.startswith("readme"):
        return "ACTIVE"
    return "UNKNOWN"


def main():
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "paper" not in p.relative_to(ROOT).parts]
    rows = []
    for path in sorted(files):
        rel = path.relative_to(ROOT)
        group = rel.parts[1] if len(rel.parts) > 1 and rel.parts[0] == "E题" else "root/other"
        cat = category(rel)
        role = status(rel)
        description = f"{group}；{cat}；{role}。" + ("论文直接来源。" if role == "PAPER_SOURCE" else "原文件保留，未移动。")
        rows.append({"文件": rel.as_posix(), "类别": cat, "状态": role, "作用/判断": description})
    from collections import Counter
    counts = Counter(row["状态"] for row in rows)
    body = "# 仓库文件盘点\n\n"
    body += "盘点范围：`E题/Q1`、`E题/Q2`、`E题/Q3`、根目录及 `paper_package`。状态是论文整理用途标记，非算法质量判定；UNKNOWN 原样保留。\n\n"
    body += "文件数：" + str(len(rows)) + "；" + "，".join(f"{k} {v}" for k, v in sorted(counts.items())) + "。\n\n"
    body += "类别包括核心源码、数据读取、模型、Loss、Config、训练、评估、推理、消融、绘图、实验结果、CSV、JSON、Markdown、checkpoint、临时文件、历史代码、论文来源。\n\n"
    body += markdown_table(rows, ["文件", "类别", "状态", "作用/判断"])
    (ROOT / "REPO_INVENTORY.md").write_text(body, encoding="utf-8")
    print(f"Inventoried {len(rows)} files; historical/legacy={counts['HISTORICAL'] + counts['LEGACY']}")


if __name__ == "__main__":
    main()
