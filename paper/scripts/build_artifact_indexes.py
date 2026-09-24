from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from common import PAPER, ROOT, SOURCES, markdown_table, read_csv, read_json, write_csv, write_table


def tracked_paths():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {x.decode("utf-8").replace("\\", "/") for x in output.split(b"\0") if x}


def main():
    tracked = tracked_paths()
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    checkpoints = []
    sizes = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size / 1_000_000
        category = "checkpoint" if path.suffix.lower() in (".pt", ".pth", ".ckpt") else "archive" if path.suffix.lower() == ".zip" else "result" if "/outputs/" in rel else "other"
        sizes.append({"path": rel, "size MB": f"{size:.3f}", "category": category})
        if category == "checkpoint":
            problem = next((q for q in ("Q1", "Q2", "Q3") if f"/{q}/" in "/" + rel), "other")
            final = "YES" if rel in ("E题/Q3/outputs/q3_v2/final_model.pt",
                                      "paper_package/Q2/model_parameters/best_bert_last4_compact.pt",
                                      "paper_package/Q2/model_parameters/best_robust_delta.pt") else "NO/REVIEW"
            purpose = "final Q3 predictor" if "q3_v2/final_model" in rel else "Q2 compact parameter component" if "model_parameters" in rel else "historical model; review before attachment"
            checkpoints.append({"problem": problem, "checkpoint": rel, "purpose": purpose,
                                "final?": final, "tracked by git?": "YES" if rel in tracked else "NO", "size MB": f"{size:.3f}"})
    (PAPER / "CHECKPOINT_INDEX.md").write_text("# Checkpoint 索引\n\n" + markdown_table(checkpoints,
        ["problem", "checkpoint", "purpose", "final?", "tracked by git?", "size MB"]), encoding="utf-8")
    sizes.sort(key=lambda row: float(row["size MB"]), reverse=True)
    (PAPER / "REPO_SIZE_REPORT.md").write_text("# 仓库体积前 30 个文件\n\n按当前工作树文件大小排序；未做哈希检查。\n\n" +
        markdown_table(sizes[:30], ["path", "size MB", "category"]), encoding="utf-8")
    q1 = read_json("q1_validation")
    shapes = [{"特征": key, "shape": " × ".join(map(str, q1["shapes"][key]))}
              for key in ("text", "audio", "vision", "time_edges")]
    write_table(PAPER / "tables/table_q1_feature_summary.md", "Q1 最终特征形状", shapes,
                ["特征", "shape"], ["特征", "形状"], ["q1_validation"])
    predictions = read_csv("q3_attachment4")
    cards = []
    for row in predictions:
        sample = row["sample_id"]
        figure = ROOT / "E题/Q3/outputs/q3_v2/attachment4/explanation_cards" / f"{sample}.png"
        cards.append({"sample_id": sample, "prediction_file": SOURCES["q3_attachment4"],
                      "explanation_file": SOURCES["q3_attachment4_evidence"],
                      "main_modality": row.get("main_modality", ""),
                      "evidence_modality": row.get("top_evidence_modality", ""),
                      "evidence_grid": f"[{row.get('top_evidence_start_grid','')},{row.get('top_evidence_end_grid','')})",
                      "approx_time": f"{row.get('approx_start_seconds','')}–{row.get('approx_end_seconds','')}",
                      "figure_path": figure.relative_to(ROOT).as_posix() if figure.is_file() else ""})
    write_csv(PAPER / "q3/Q3_EXPLANATION_CARD_INDEX.csv", cards,
              ["sample_id", "prediction_file", "explanation_file", "main_modality", "evidence_modality", "evidence_grid", "approx_time", "figure_path"])
    print(f"Indexed {len(checkpoints)} checkpoints and {len(cards)} Q3 cards")


if __name__ == "__main__":
    main()
