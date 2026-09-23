#!/usr/bin/env python3
"""Create the Q1 dataset summary, alignment example, and representative timeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BASE = Path(os.environ.get("Q1_BASE_DIR", SCRIPT_DIR))
DEFAULT_MODEL_DIR = Path(os.environ.get("Q1_MODEL_DIR", REPO_ROOT / "models"))
REPRESENTATIVE_ID = "-3g5yACwYnA/13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BASE / "work")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE / "results")
    parser.add_argument(
        "--dictionary", type=Path, default=DEFAULT_BASE / "work" / "english_us_arpa_augmented.dict"
    )
    parser.add_argument("--representative-id", default=REPRESENTATIVE_ID)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_representative(
    output_dir: Path,
    sample_id: str,
    archive: Any,
    alignment_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    manifest_row: pd.Series,
) -> None:
    plt.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ids = archive["sample_ids"].astype(str)
    found = np.flatnonzero(ids == sample_id)
    if not len(found):
        raise ValueError(f"Representative sample is absent from feature archive: {sample_id}")
    idx = int(found[0])
    length = int(archive["lengths"][idx])
    edges = archive["time_windows_sec"][idx, :length]
    duration = float(archive["durations_video_sec"][idx])
    text_mask = archive["text_mask"][idx, :length]
    audio_mask = archive["audio_mask"][idx, :length]
    vision_mask = archive["vision_mask"][idx, :length]

    word_rows = [row for row in alignment_rows if row["sample_id"] == sample_id]
    run_features = metadata["feature_definitions"]
    blendshape_names = run_features.get("vision", [])
    blendshape_values = archive["vision_features"][idx, :length]
    valid_vision = np.flatnonzero(vision_mask)
    selected_blendshapes: list[int] = []
    name_preferences = ("jawOpen", "mouthSmile", "browDown", "eyeBlink")
    blendshape_cn = {
        "jawOpen": "jawOpen（张口）",
        "mouthSmileLeft": "mouthSmileLeft（左侧嘴角上扬）",
        "browDownLeft": "browDownLeft（左眉下压）",
        "eyeBlinkLeft": "eyeBlinkLeft（左眼眨动）",
    }
    for pattern in name_preferences:
        match = next((i for i, name in enumerate(blendshape_names) if pattern.lower() in name.lower()), None)
        if match is not None and match not in selected_blendshapes:
            selected_blendshapes.append(match)
    if valid_vision.size and len(selected_blendshapes) < 4:
        mean_scores = blendshape_values[valid_vision].mean(axis=0)
        for i in np.argsort(mean_scores)[::-1]:
            if int(i) not in selected_blendshapes:
                selected_blendshapes.append(int(i))
            if len(selected_blendshapes) == 4:
                break

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [5.8, 1.8, 2.2, 1.2]},
        constrained_layout=True,
    )
    ax = axes[0]
    word_rows = sorted(word_rows, key=lambda x: (x["start_sec_video"] is None, x["start_sec_video"] or 0))
    n_words = max(1, len(word_rows))
    ylabels = []
    for i, row in enumerate(word_rows):
        y = n_words - i - 1
        label = str(row["word"])
        if not row["aligned"]:
            ylabels.append(f"{label}（未对齐）")
            continue
        start = float(row["start_sec_video"])
        end = float(row["end_sec_video"])
        ax.broken_barh([(start, max(0.005, end - start))], (y - 0.31, 0.62), color="#285f9e")
        ax.text((start + end) / 2, y, label, ha="center", va="center", color="white", fontsize=7, clip_on=True)
        ylabels.append(label)
    ax.set_yticks(range(n_words), labels=list(reversed(ylabels)), fontsize=7)
    ax.set_ylim(-0.6, n_words - 0.4)
    ax.set_title("MFA（蒙特利尔强制对齐器）逐词时间对齐（蓝色条表示已对齐）")
    ax.grid(axis="x", alpha=0.25)

    bin_centers = edges.mean(axis=1)
    audio = archive["audio_features"][idx, :length]
    rms = audio[:, 13].astype(float)
    rms[~audio_mask] = np.nan
    axes[1].step(bin_centers, rms, where="mid", color="#bc5b21", linewidth=1.7)
    axes[1].set_ylabel("RMS（均方根能量）")
    axes[1].set_title("音频描述符：每个时间窗内短时 RMS（均方根能量）的均值")
    axes[1].grid(alpha=0.25)

    ax = axes[2]
    if valid_vision.size and selected_blendshapes:
        for feature_idx in selected_blendshapes:
            values = blendshape_values[:, feature_idx].astype(float)
            values[~vision_mask] = np.nan
            name = blendshape_names[feature_idx] if feature_idx < len(blendshape_names) else f"score_{feature_idx}"
            ax.plot(bin_centers, values, marker="o", markersize=3, label=blendshape_cn.get(name, name))
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "该片段未检出人脸", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("分数")
    ax.set_title("MediaPipe Face Landmarker（人脸模型）的 blendshape（面部形变）分数，不是情绪标签")
    ax.grid(alpha=0.25)

    mask_matrix = np.vstack([text_mask, audio_mask, vision_mask]).astype(float)
    axes[3].imshow(
        mask_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=(0, duration, 3, 0),
        cmap=plt.matplotlib.colors.ListedColormap(["#e3e7ec", "#2d8a60"]),
        vmin=0,
        vmax=1,
    )
    axes[3].set_yticks([0.5, 1.5, 2.5], labels=["文本", "音频", "视觉"])
    axes[3].set_title("模态有效窗掩码（绿色=有效；灰色=缺失或未对齐）")
    axes[3].set_xlabel("视频相对时间（秒）")
    axes[3].set_xlim(0, duration)
    axes[3].set_xticks(np.arange(0, math.ceil(duration * 2) / 2 + 0.01, 0.5), minor=True)
    axes[3].grid(which="minor", axis="x", alpha=0.14)

    fig.suptitle(
        f"Q1 典型样本：{sample_id} | {duration:.3f} 秒 | "
        f"文本 {archive['text_features'].shape[-1]} 维 / 音频 {archive['audio_features'].shape[-1]} 维 / "
        f"视觉 {archive['vision_features'].shape[-1]} 维",
        fontsize=13,
    )
    fig.savefig(output_dir / "q1_typical_sample.png", dpi=190)
    plt.close(fig)

    sample_alignment = [
        row
        for row in alignment_rows
        if row["sample_id"] == sample_id
    ]
    write_json_csv(output_dir / "q1_alignment_example.csv", sample_alignment)


def plot_source_frames(
    output_dir: Path,
    sample_id: str,
    record: dict[str, Any],
    archive: Any,
    archive_index: dict[str, int],
) -> None:
    import av

    array_index = archive_index[sample_id]
    length = int(archive["lengths"][array_index])
    edges = archive["time_windows_sec"][array_index, :length]
    face_mask = archive["vision_mask"][array_index, :length]
    candidates = np.flatnonzero(face_mask)
    if not candidates.size:
        candidates = np.arange(length)
    selected = np.unique(candidates[np.linspace(0, len(candidates) - 1, min(5, len(candidates))).round().astype(int)])
    targets = {int(position): float(edges[position].mean()) for position in selected}
    pending = list(targets)
    frames: dict[int, Any] = {}
    video_start = float(record.get("video_start_sec", 0.0))
    with av.open(record["source_path"]) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None and frame.pts is not None:
                timestamp = float(frame.pts * frame.time_base)
            if timestamp is None:
                continue
            clip_time = float(timestamp) - video_start
            while pending and clip_time + 1e-7 >= targets[pending[0]]:
                position = pending.pop(0)
                frames[position] = frame.to_image()
            if not pending:
                break
    if not frames:
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(3.8 * len(frames), 3.1), squeeze=False)
    for ax, (position, image) in zip(axes[0], sorted(frames.items())):
        start, end = map(float, edges[position])
        state = "人脸分数有效" if face_mask[position] else "该窗未检出人脸"
        ax.imshow(image)
        ax.set_title(f"时间窗 {position + 1}\n{start:.2f}–{end:.2f} 秒\n{state}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"原视频帧与 0.5 秒窗的对应关系：{sample_id}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "q1_typical_video_frames.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_run_logs(work_dir: Path, output_dir: Path) -> None:
    run_log = work_dir / "run_q1.log"
    if run_log.is_file():
        lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        starts = [i for i, line in enumerate(lines) if "Q1 stage=extract" in line]
        if starts:
            final_lines = lines[starts[-1] :]
            writes = [i for i, line in enumerate(final_lines) if " Wrote " in line]
            if writes:
                final_lines = final_lines[: writes[-1] + 1]
            (output_dir / "q1_final_run.log").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    for source_name, output_name in (
        ("mfa_align.log", "q1_mfa_align.log"),
        ("mfa_g2p.log", "q1_mfa_g2p.log"),
        ("mfa_align_status.json", "q1_mfa_align_status.json"),
    ):
        source = work_dir / source_name
        if source.is_file():
            shutil.copyfile(source, output_dir / output_name)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "q1_manifest.csv"
    feature_path = args.output_dir / "q1_features.npz"
    alignment_path = args.output_dir / "q1_alignment.jsonl"
    for path in (manifest_path, feature_path, alignment_path, args.work_dir / "records.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(manifest_path)
    records = read_jsonl(args.work_dir / "records.jsonl")
    record_by_id = {row["sample_id"]: row for row in records}
    alignment_rows = read_jsonl(alignment_path)
    metadata = json.loads((args.output_dir / "q1_run_info.json").read_text(encoding="utf-8"))
    archive = np.load(feature_path, allow_pickle=False)
    if not args.dictionary.is_file():
        args.dictionary = DEFAULT_MODEL_DIR / "english_us_arpa_dictionary.dict"
    archive_index = {sample_id: i for i, sample_id in enumerate(archive["sample_ids"].astype(str))}
    if len(manifest) != 100 or len(records) != 100:
        raise ValueError(f"Expected exactly 100 records; got manifest={len(manifest)}, records={len(records)}")

    dictionary_words: set[str] = set()
    with args.dictionary.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.split()
            if fields:
                dictionary_words.add(fields[0].lower())

    textgrid_missing: list[str] = []
    issue_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        rec = record_by_id[sample_id]
        textgrid_path = Path(rec.get("textgrid_path", ""))
        missing_grid = not textgrid_path.is_file()
        if missing_grid:
            textgrid_missing.append(sample_id)
        oov = sorted({word for word in rec.get("alignment_tokens", []) if word.lower() not in dictionary_words})
        reasons = []
        if missing_grid:
            reasons.append("MFA did not export a TextGrid")
        if int(row["aligned_word_count"]) == 0:
            reasons.append("no aligned transcript words")
        elif int(row["aligned_word_count"]) < int(row["text_word_count"]):
            reasons.append(f"partial word alignment {int(row['aligned_word_count'])}/{int(row['text_word_count'])}")
        if int(row["audio_valid_bins"]) == 0:
            reasons.append("no valid audio windows")
        if int(row["vision_valid_bins"]) == 0:
            reasons.append("no detected face windows")
        if str(row["prepare_status"]) != "ok":
            reasons.append(f"preparation status: {row['prepare_status']}")
        if reasons:
            array_index = archive_index[sample_id]
            n_bins = int(archive["lengths"][array_index])
            sample_audio_mask = archive["audio_mask"][array_index, :n_bins]
            sample_audio_rms = archive["audio_features"][array_index, :n_bins, 13]
            issue_rows.append(
                {
                    "sample_id": sample_id,
                    "prepare_status": row["prepare_status"],
                    "mfa_textgrid_missing": missing_grid,
                    "text_word_count": int(row["text_word_count"]),
                    "aligned_word_count": int(row["aligned_word_count"]),
                    "dictionary_oov_words": " ".join(oov),
                    "audio_valid_bins": int(row["audio_valid_bins"]),
                    "audio_rms_mean": (
                        float(sample_audio_rms[sample_audio_mask].mean()) if sample_audio_mask.any() else 0.0
                    ),
                    "vision_valid_bins": int(row["vision_valid_bins"]),
                    "issue": "; ".join(reasons),
                }
            )
    write_json_csv(args.output_dir / "q1_quality_issues.csv", issue_rows)

    npz_ids = archive["sample_ids"].astype(str)
    time_mask = archive["time_mask"]
    modality_masks = {
        "text": archive["text_mask"],
        "audio": archive["audio_mask"],
        "vision": archive["vision_mask"],
    }
    n_time_bins = int(time_mask.sum())
    mask_totals = {name: int(mask.sum()) for name, mask in modality_masks.items()}
    zero_alignment_rows = manifest.loc[manifest["aligned_word_count"] == 0]
    silent_unaligned = 0
    for _, row in zero_alignment_rows.iterrows():
        array_index = archive_index[str(row["sample_id"])]
        n_bins = int(archive["lengths"][array_index])
        valid_audio = archive["audio_mask"][array_index, :n_bins]
        rms = archive["audio_features"][array_index, :n_bins, 13]
        if valid_audio.any() and float(rms[valid_audio].mean()) == 0.0:
            silent_unaligned += 1
    word_total = int(manifest["text_word_count"].sum())
    aligned_total = int(manifest["aligned_word_count"].sum())
    zero_alignment = int((manifest["aligned_word_count"] == 0).sum())
    zero_face = int((manifest["vision_valid_bins"] == 0).sum())
    zero_audio = int((manifest["audio_valid_bins"] == 0).sum())
    duration_total = float(manifest["duration_video_sec"].sum())
    max_length = int(archive["lengths"].max())
    feature_shapes = metadata["feature_shapes"]
    feature_bytes = feature_path.stat().st_size

    summary = f"""# Q1 数据处理与特征生成报告

## 数据覆盖与时间结构

- 附件1标签表提供 {len(manifest)} 条样本，按 `video_id + clip_id` 与 100 个分层目录视频一一对应；代码只读取 `video_id`、`clip_id` 和英文 `text`（转写）三列，答案标签与 annotation（人工标注）不进入处理流程。
- 视频总有效时长 {duration_total:.2f} 秒，单条时长 {manifest['duration_video_sec'].min():.2f}–{manifest['duration_video_sec'].max():.2f} 秒；统一采用 0.5 秒窗，最后一个窗按实际视频长度截断。变长序列补齐到 {max_length} 窗，并保留每条长度与时间边界。
- 全集共有 {n_time_bins} 个真实时间窗；文本、音频、视觉有效窗分别为 {mask_totals['text']}、{mask_totals['audio']}、{mask_totals['vision']}。

## 特征定义

- **文本（128 维）**：清理方括号内提示和标点后，将转写切成英文词；用通用预训练 BERT（双向编码器表示）得到每词最后一层 WordPiece（子词）向量，再按词与时间窗的重叠时长加权汇总。BERT 不做情感微调；只保留 MFA（Montreal Forced Aligner，蒙特利尔强制对齐器）的逐词时间区间与音频时间重叠的窗口。
- **音频（40 维）**：FFmpeg 转 16 kHz 单声道；25 ms 窗、512 点 FFT（快速傅里叶变换）、10 ms 帧移提取 MFCC（梅尔频率倒谱系数）C0–C12、RMS（均方根能量）、ZCR（过零率）、YIN F0（YIN 基频）、谱质心、谱带宽、谱滚降和谱平坦度，共 20 个逐帧描述符；每个 0.5 秒窗记录逐帧均值和标准差。
- **视觉（52 维）**：按每个 0.5 秒窗中心抽取视频帧，MediaPipe Face Landmarker 输出 52 个 blendshape（面部形变）分数。分数是面部运动代理量，不作为情绪标签或人工 Action Unit（动作单元）真值。
- **缺失与补齐**：时间、文本、音频、人脸检测分别保存掩码。有效模态缺失值的数值占位为零但掩码为 false；序列补齐位由 time mask 标记，时间边界为 NaN。

## 对齐质量

- MFA 命令使用 beam（搜索波束宽度）{metadata.get('alignment_settings', {}).get('beam', 10)}、retry beam（重试搜索宽度）{metadata.get('alignment_settings', {}).get('retry_beam', 40)}；G2P（字素到音素）模型 `{metadata.get('alignment_settings', {}).get('g2p_model', 'english_us_arpa')}` 为基础字典外的转写词补充发音。导出 TextGrid（Praat 时间标注文件）{100 - len(textgrid_missing)}/100 个。总转写词数 {word_total}，逐词对齐 {aligned_total}，词级覆盖率 {aligned_total / word_total if word_total else 0:.1%}；{zero_alignment} 条样本当前没有可用词级对齐，其中 {silent_unaligned} 条音轨的有效 RMS 均值为 0；{zero_face} 条没有检出人脸窗口，{zero_audio} 条没有有效音频窗。
- 字典外词和逐样本问题见 `q1_quality_issues.csv`；缺失位置保持掩码，不以零分数冒充观测。
- 特征形状：{json.dumps(feature_shapes, ensure_ascii=False)}。`q1_manifest.csv` 提供每条样本路径、时长、窗数、词数、对齐覆盖率、模态有效窗和特征维度；`q1_alignment.jsonl` 保存全部逐词时间边界。

## 典型样本

样本 `{args.representative_id}` 的逐词对齐和三个模态时间窗映射见 `q1_alignment_example.csv` 与 `q1_typical_sample.png`；按窗中心选取的原视频帧见 `q1_typical_video_frames.png`。图中面板从上到下对应词级时间、短时 RMS、代表性面部形变分数以及三个模态的有效窗掩码。

## 可复现设置与限制

- 运行入口：在本目录执行 python run_q1.py --stage all --num-jobs 8 --window-sec 0.5 --beam 100 --retry-beam 400；统计与图表执行 python make_q1_report.py。
- 环境与模型记录见 `q1_run_info.json`；预训练模型只用于通用语言表征和面部形状描述，不使用其他情感数据集训练、微调、调参或统计。
- 竞赛附件总大小上限为 50 MB；本目录不包含原视频、外部数据或预训练权重。重跑时通过 Q1_DATA_DIR 与 Q1_MODEL_DIR 指定数据和模型资源目录。

## 参考工具文档

- [MFA dictionary format](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html)
- [Google BERT model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)
- [MediaPipe FaceLandmarker options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions)
- [MFA G2P usage](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/)
"""
    (args.output_dir / "Q1_论文正文.md").write_text(summary, encoding="utf-8")

    representative = manifest.loc[manifest["sample_id"] == args.representative_id]
    if representative.empty:
        raise ValueError(f"Representative sample not found in manifest: {args.representative_id}")
    plot_representative(
        args.output_dir,
        args.representative_id,
        archive,
        alignment_rows,
        metadata,
        representative.iloc[0],
    )
    plot_source_frames(
        args.output_dir,
        args.representative_id,
        record_by_id[args.representative_id],
        archive,
        archive_index,
    )
    copy_run_logs(args.work_dir, args.output_dir)

    versions = metadata.get("package_versions", {})
    readme = f"""# Q1 特征数据交付说明

本目录完成赛题第一问的 Attachment 1（附件1）100 条视频处理。

## 文件

- `q1_features.npz`：压缩多模态张量，包含 sample IDs（样本键）、逐窗时间边界、有效长度和模态掩码。
- `q1_manifest.csv`：100 行样本/模态摘要。
- `q1_alignment.jsonl`：词级起止时间及对齐状态。
- `q1_quality_issues.csv`：无对齐、无有效模态或字典外词的样本列表。
- `q1_final_run.log`、`q1_mfa_align.log`、`q1_mfa_g2p.log`、`q1_mfa_align_status.json`：最终运行和对齐日志。
- `Q1_论文正文.md`：Q1 方法、结果、质量状态和可复现说明。
- `q1_alignment_example.csv`、`q1_typical_sample.png`、`q1_typical_video_frames.png`：代表样本对齐、特征和原帧示例。
- `run_q1.py`、`make_q1_report.py`：完整特征生成与报告脚本。

## 特征轴顺序

`q1_features.npz` 中主要张量形状为 `{feature_shapes}`。每个样本按 `lengths` 读取真实窗数；各特征维定义和视觉 blendshape（面部形变分数）名称见 `q1_run_info.json`。mask（掩码）为 false 的模态值不得当作观测值。

## 复现

服务器环境为 Conda（环境管理器）`math`，主要包版本：`{json.dumps(versions, ensure_ascii=False)}`；外部工具版本：`{json.dumps(metadata.get('external_tool_versions', {}), ensure_ascii=False)}`。

```bash
conda activate math
mfa model download g2p english_us_arpa
python ./run_q1.py --stage all --num-jobs 8 --window-sec 0.5 --beam 100 --retry-beam 400
python ./make_q1_report.py
```

原始视频、比赛数据和预训练权重需从赛题数据包或允许渠道单独获取，本仓库不包含这些文件。通过 Q1_DATA_DIR 与 Q1_MODEL_DIR 配置相应目录。程序只从标签表读取视频键与转写文本，不读取 label（答案标签）和 annotation（人工标注）列。
"""
    (args.output_dir / "README_Q1.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "samples": len(manifest),
                "duration_sec": round(duration_total, 3),
                "time_bins": n_time_bins,
                "feature_shapes": feature_shapes,
                "textgrid_exported": 100 - len(textgrid_missing),
                "textgrid_missing": textgrid_missing,
                "zero_word_alignment": zero_alignment,
                "zero_face_samples": zero_face,
                "zero_audio_samples": zero_audio,
                "word_coverage": aligned_total / word_total if word_total else 0,
                "feature_bytes": feature_bytes,
                "output_files": [p.name for p in sorted(args.output_dir.iterdir()) if p.is_file()],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
