"""Read-only source helpers for paper tables. Run from any working directory."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SOURCES = json.loads((PAPER / "source_of_truth.json").read_text(encoding="utf-8"))


def source(key: str) -> Path:
    path = ROOT / SOURCES[key]
    if not path.is_file():
        raise FileNotFoundError(f"{key}: {path}")
    return path


def read_json(key: str):
    return json.loads(source(key).read_text(encoding="utf-8"))


def read_csv(key: str):
    with source(key).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], fields: list[str], labels: list[str] | None = None):
    labels = labels or fields
    def cell(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_table(path: Path, title: str, rows: list[dict], fields: list[str], labels: list[str], source_keys: list[str], note: str = ""):
    paths = [SOURCES[key] for key in source_keys]
    body = f"# {title}\n\n" + markdown_table(rows, fields, labels)
    body += "\nSource:\n" + "\n".join(f"- `{path}`" for path in paths) + "\n"
    if note:
        body += f"\n{note}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def metric_block(metrics: dict) -> dict:
    return {
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("f1_macro", metrics.get("macro_f1", "")),
        "weighted_f1": metrics.get("f1_weighted", metrics.get("weighted_f1", "")),
        "mae": metrics.get("mae", ""),
        "pearson": metrics.get("pearson", ""),
    }
