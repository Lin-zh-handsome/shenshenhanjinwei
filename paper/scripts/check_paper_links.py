"""Check local Markdown links and registered source paths; no model execution."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import unquote

from common import PAPER, ROOT, SOURCES


def main():
    missing = []
    for key, relative in SOURCES.items():
        if not (ROOT / relative).is_file():
            missing.append(f"source_of_truth:{key}: {relative}")
    for markdown in [ROOT / "README.md", *ROOT.glob("E题/Q?/README.md"), *PAPER.rglob("*.md")]:
        content = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
            target = unquote(target.split("#", 1)[0].strip("<>"))
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (markdown.parent / target).exists():
                missing.append(f"link:{markdown.relative_to(ROOT).as_posix()}: {target}")
        for target in re.findall(r"`(E题/[^`]+)`", content):
            if any(marker in target for marker in ("*", "{", "}", "…", ",")) or target.endswith("/"):
                continue
            if not (ROOT / target).exists():
                missing.append(f"source:{markdown.relative_to(ROOT).as_posix()}: {target}")
    for name in ("EXPERIMENT_REGISTRY.csv", "FINAL_RESULTS.csv"):
        with (PAPER / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for field in ("config_path", "metrics_path", "checkpoint_path", "prediction_path", "source_file"):
                    if row.get(field) and not (ROOT / row[field]).is_file():
                        missing.append(f"{name}:{field}: {row[field]}")
    if missing:
        print("\n".join(missing))
        raise SystemExit(f"{len(missing)} missing local targets")
    print("Local README/paper links, registered paths and source-of-truth files exist")


if __name__ == "__main__":
    main()
