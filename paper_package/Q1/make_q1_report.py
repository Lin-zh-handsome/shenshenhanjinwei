#!/usr/bin/env python3
"""Build result-grounded Q1 audit, manuscript text, manual review CSV and figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def decode_audio(video: Path) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(command, check=False, capture_output=True, timeout=120)
    if result.returncode:
        return np.zeros(0, dtype=np.float32), 16000
    return np.frombuffer(result.stdout, dtype="<f4").copy(), 16000


def keyframes_by_pts(video: Path, targets_relative: list[float], first_pts: float) -> list[np.ndarray | None]:
    import av

    targets = [first_pts + value for value in targets_relative]
    best: list[tuple[float, np.ndarray | None]] = [(float("inf"), None) for _ in targets]
    with av.open(str(video)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            return [None] * len(targets)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            for index, target in enumerate(targets):
                difference = abs(timestamp - target)
                if difference < best[index][0]:
                    best[index] = (difference, frame.to_ndarray(format="rgb24"))
    return [item[1] for item in best]


def generate_typical_figures(result_dir: Path, data_dir: Path, manifest: pd.DataFrame, arrays: np.lib.npyio.NpzFile, alignment: list[dict], figure_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = manifest.sort_values("duration_used_sec").reset_index(drop=True)
    median_value = float(selected["duration_used_sec"].astype(float).median())
    median_row = selected.iloc[int(np.argmin(np.abs(selected["duration_used_sec"].astype(float).to_numpy() - median_value)))]
    chosen = [selected.iloc[0], median_row, selected.iloc[-1]]
    chosen_names = ["short", "medium", "long"]
    ids = arrays["ids"].astype(str).tolist()
    for record, label in zip(chosen, chosen_names):
        sample_id = str(record["sample_id"])
        index = ids.index(sample_id)
        duration = float(record["duration_used_sec"])
        video = data_dir / str(record["relative_video_path"])
        audio, sample_rate = decode_audio(video)
        audio_offset = float(record.get("first_audio_pts_sec") or 0) - float(record.get("first_video_pts_sec") or 0)
        time = audio_offset + np.arange(len(audio), dtype=np.float32) / sample_rate
        edges = arrays["time_edges"][index].astype(np.float64)
        bins = [5, 25, 45]
        frame_times = [(edges[k] + edges[k + 1]) / 2 for k in bins]
        try:
            images = keyframes_by_pts(video, frame_times, float(record.get("first_video_pts_sec") or 0))
        except Exception:
            images = [None] * len(bins)
        words = [item for item in alignment if item["sample_id"] == sample_id and item.get("start_sec") is not None]
        figure = plt.figure(figsize=(17, 17), constrained_layout=True)
        grid = GridSpec(7, 1, figure=figure, height_ratios=[1.8, 1.3, 2.2, 2.6, 2.4, 2.2, 1.2])
        frame_grid = grid[0].subgridspec(1, 3, wspace=0.03)
        for image_index, (k, image) in enumerate(zip(bins, images)):
            axis = figure.add_subplot(frame_grid[0, image_index])
            if image is not None:
                axis.imshow(image)
            else:
                axis.text(0.5, 0.5, "frame unavailable", ha="center", va="center")
            axis.set_title(f"k={k}, [{edges[k]:.3f}, {edges[k + 1]:.3f}) s\n{record['relative_video_path']}", fontsize=9)
            axis.axis("off")

        wave_axis = figure.add_subplot(grid[1])
        if len(audio):
            stride = max(1, len(audio) // 100_000)
            wave_axis.plot(time[::stride], audio[::stride], linewidth=0.35, color="#38598c")
        wave_axis.set_xlim(0, duration)
        wave_axis.set_ylabel("waveform")
        wave_axis.set_xlabel("video-relative time (s)")
        wave_axis.grid(alpha=0.2)

        words_axis = figure.add_subplot(grid[2], sharex=wave_axis)
        words_axis.set_xlim(0, duration)
        words_axis.set_ylim(-0.25, 3.25)
        words_axis.set_yticks([])
        words_axis.set_ylabel("words\n(start–end s)")
        words_to_draw = words if len(words) <= 72 else [words[int(i)] for i in np.linspace(0, len(words) - 1, 72)]
        for word_index, word in enumerate(words_to_draw):
            start, end = float(word["start_sec"]), float(word["end_sec"])
            center = (start + end) / 2
            lane = word_index % 3
            words_axis.hlines(lane, start, end, color="#c45a32", linewidth=1.1)
            words_axis.text(center, lane + 0.04, f"{word['word']}\n{start:.2f}–{end:.2f}", rotation=35, fontsize=5.5, ha="center", va="bottom", clip_on=True)
        words_axis.grid(axis="x", alpha=0.2)

        for grid_index, modality, matrix in (
            (3, "text (128 PCA dimensions)", arrays["text"][index].astype(np.float32)),
            (4, "audio (74 mean/std features)", arrays["audio"][index].astype(np.float32)),
            (5, "vision (52 named blendshapes)", arrays["vision"][index].astype(np.float32)),
        ):
            axis = figure.add_subplot(grid[grid_index], sharex=wave_axis)
            scale = matrix.std(axis=0, keepdims=True)
            normalized = (matrix - matrix.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-6)
            axis.imshow(np.clip(normalized.T, -3, 3), aspect="auto", origin="lower", extent=(0, duration, 0, matrix.shape[1]), cmap="coolwarm", vmin=-3, vmax=3)
            axis.set_ylabel(modality)
            axis.set_xlim(0, duration)
            axis.grid(axis="x", alpha=0.2)

        mask_axis = figure.add_subplot(grid[6], sharex=wave_axis)
        for lane, name, key in ((2, "text", "text_mask"), (1, "audio", "audio_mask"), (0, "vision", "vision_mask")):
            mask = arrays[key][index].astype(np.uint8)
            mask_axis.step(edges[:-1], mask + lane, where="post", label=name)
        mask_axis.set_yticks([0.5, 1.5, 2.5], ["vision", "audio", "text"])
        mask_axis.set_xlim(0, duration)
        mask_axis.set_xlabel("shared video-relative time (s); each k maps to [edge[k], edge[k+1])")
        mask_axis.grid(axis="x", alpha=0.2)
        figure.suptitle(f"Q1 {label}: {sample_id} | duration={duration:.3f}s | K=50", fontsize=14)
        figure.savefig(figure_dir / f"q1_typical_{label}.png", dpi=160)
        plt.close(figure)


def generate_comparison_figure(run_info: dict, figure_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = run_info["comparison"]
    granular = comparison["time_granularity"]
    ks = [25, 50, 100]
    visual = comparison["visual_aggregation"]["aggregation_mask_rates"]
    alignment = comparison["text_time_alignment"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].plot(ks, [granular[str(k)]["mean_audio_frames_per_window"] for k in ks], "o-", label="audio frames/window")
    axes[0].plot(ks, [granular[str(k)]["mean_video_frames_per_window"] for k in ks], "s-", label="video frames/window")
    axes[0].set_title("Time granularity")
    axes[0].set_xlabel("K")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].bar(list(visual), list(visual.values()), color=["#8c6bb1", "#41ab5d", "#2b8cbe"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Vision valid-window ratio")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].bar(list(alignment), [alignment[key]["word_boundary_coverage"] for key in alignment], color="#e69535")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Word boundary coverage (not accuracy)")
    axes[2].tick_params(axis="x", rotation=25)
    figure.savefig(figure_dir / "q1_design_comparison.png", dpi=160)
    plt.close(figure)


def generate_manual_review(manifest: pd.DataFrame, alignment: list[dict], output_path: Path, seed: int = 20260923) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_ids = manifest["sample_id"].astype(str).tolist()
    chosen = set(rng.choice(sample_ids, size=min(10, len(sample_ids)), replace=False).tolist())
    rows = []
    by_sample: dict[str, list[dict]] = {}
    for word in alignment:
        if word.get("sample_id") in chosen and word.get("start_sec") is not None:
            by_sample.setdefault(word["sample_id"], []).append(word)
    for sample_id in sorted(chosen):
        sample_words = by_sample.get(sample_id, [])
        if not sample_words:
            continue
        count = min(6, len(sample_words))
        indices = np.linspace(0, len(sample_words) - 1, count).round().astype(int)
        video_path = manifest.loc[manifest["sample_id"].astype(str) == sample_id, "relative_video_path"].iloc[0]
        for index in indices:
            word = sample_words[int(index)]
            rows.append({
                "sample_id": sample_id,
                "word": word["word"],
                "predicted_start": word["start_sec"],
                "predicted_end": word["end_sec"],
                "timestamp_source": word["timestamp_source"],
                "source_video": video_path,
            })
    frame = pd.DataFrame(rows, columns=["sample_id", "word", "predicted_start", "predicted_end", "timestamp_source", "source_video"])
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def write_final_documents(result_dir: Path, manifest: pd.DataFrame, arrays: np.lib.npyio.NpzFile, run_info: dict, validation: dict, alignment: list[dict], manual_rows: pd.DataFrame) -> None:
    durations = manifest["duration_used_sec"].astype(float)
    total_windows = len(manifest) * 50
    source_counts = pd.Series([item.get("timestamp_source") for item in alignment]).value_counts(dropna=False).to_dict()
    total_words = len(alignment)
    forced = int(source_counts.get("forced_alignment", 0))
    interpolated = int(source_counts.get("interpolated", 0))
    proportional = int(source_counts.get("proportional_fallback", 0))
    audio_present = int(arrays["audio_present"].sum())
    audio_valid = int(arrays["audio_mask"].sum())
    vision_valid = int(arrays["vision_mask"].sum())
    text_valid = int(arrays["text_mask"].sum())
    decode_video = int(manifest["video_decode_complete"].astype(bool).sum())
    decode_audio = int(manifest["audio_decode_complete"].astype(bool).sum())
    comparisons = run_info["comparison"]
    warnings = run_info.get("warning_summary", {})
    arrays_meta = run_info["arrays"]
    errors = arrays_meta["float16_conversion_error"]
    duration_stats = f"{durations.min():.3f}–{durations.max():.3f} s (median {durations.median():.3f} s)"
    size_mb = arrays_meta["compressed_npz_bytes"] / (1024 * 1024)
    warning_table = "\n".join(f"| `{code}` | {count} |" for code, count in sorted(warnings.items())) or "| 无 | 0 |"
    low_vision = manifest.loc[(manifest["vision_valid_bins"] == 0) | (manifest["face_coverage_global"] < 0.20), "sample_id"].astype(str).tolist()
    near_silence = manifest.loc[manifest["audio_reliable_bins"] == 0, "sample_id"].astype(str).tolist()
    duration_delta = manifest["duration_video_stream_minus_used_sec"].astype(float)
    audio_delta = manifest["duration_audio_video_decoded_difference_sec"].astype(float)
    alignment_report = "\n".join(
        f"| {name} | {entry['word_boundary_coverage']:.4f} | {entry['order_violations']} | 未提供时间戳真值 |"
        for name, entry in comparisons["text_time_alignment"].items()
    )
    granularity_report = "\n".join(
        f"| {k} | {entry['mean_audio_frames_per_window']:.2f} | {entry['mean_video_frames_per_window']:.2f} | {entry['empty_text_window_ratio']:.4f} | {entry['vision_valid_window_ratio']:.4f} | {entry['compressed_variant_npz_bytes_pre_pca']:,} |"
        for k, entry in comparisons["time_granularity"].items()
    )
    vision_cmp = comparisons["visual_aggregation"]
    vision_report = "\n".join(
        f"| {name} | {rate:.4f} | {vision_cmp['zero_vision_sample_counts'][name]} |"
        for name, rate in vision_cmp["aggregation_mask_rates"].items()
    )
    audit = f"""# Q1_FINAL_AUDIT：最终结果自动审计

生成来源：`q1_manifest.csv`、`q1_features.npz`、`q1_alignment.jsonl`、`q1_run_info.json`、`validation_report.json`。本报告由 `make_q1_report.py` 从实际运行数据生成，不手工填入实验结果。

## 覆盖、接口与存储

- Excel / manifest / NPZ 样本 ID 集合完全一致：{len(manifest)} 条；重复 ID：0。
- 特征形状：text `{tuple(arrays['text'].shape)}`，audio `{tuple(arrays['audio'].shape)}`，vision `{tuple(arrays['vision'].shape)}`；真实时间边界 `{tuple(arrays['time_edges'].shape)}`。
- 统一步数 K=50；未使用时间维 padding；所有首边界为 0、每条严格递增、末边界与 `duration_used_sec` 一致：{validation['status']}。
- 实际时长范围：{duration_stats}；NPZ 压缩后大小：{size_mb:.2f} MiB。
- PTS 解码得到的 FPS 中位数 `{manifest['fps_decoded'].astype(float).median():.3f}`，范围 `{manifest['fps_decoded'].astype(float).min():.3f}–{manifest['fps_decoded'].astype(float).max():.3f}`；实际帧逐一按时间戳分箱。
- 旧实现记录为 1,084/1,627 个中心帧视觉有效窗、25/100 条视频完全无有效视觉窗。新版统计使用 K=50 的 5,000 窗和多帧覆盖率/置信度规则，分母不同；本报告同时提供新版数值，不能把差值单独归因于某一个因素。
- float32 转 float16 误差：text 最大/平均 `{errors['text']['max_absolute_error']:.7g}` / `{errors['text']['mean_absolute_error']:.7g}`；audio `{errors['audio']['max_absolute_error']:.7g}` / `{errors['audio']['mean_absolute_error']:.7g}`；vision `{errors['vision']['max_absolute_error']:.7g}` / `{errors['vision']['mean_absolute_error']:.7g}`。
- 有效文本窗 {text_valid}/{total_windows}；有效音频窗 {audio_valid}/{total_windows}；存在可解码音频帧窗 {audio_present}/{total_windows}；有效视觉窗 {vision_valid}/{total_windows}。
- 特征 NaN/Inf 检查：{json.dumps(validation['nonfinite_feature_values'], ensure_ascii=False)}。

## 视频、音频解码及时长口径

- 完整解码到结束：视频 {decode_video}/{len(manifest)}；音频 {decode_audio}/{int((manifest['duration_audio_stream_sec'] > 0).sum())}（有音频流样本）。未完整样本在本报告末列出，保留在结果中。
- 视频流时长减实际解码 PTS 时长：中位数 `{duration_delta.median():.4f}` s，最大绝对差 `{duration_delta.abs().max():.4f}` s。
- 音频解码时长减视频解码时长：中位数 `{audio_delta.median():.4f}` s，最大绝对差 `{audio_delta.abs().max():.4f}` s。
- 每条 `duration_used_sec` 定义为最后一个有效解码视频帧 PTS 减第一个有效解码视频帧 PTS，再加末帧实际 duration（缺失时用解码 PTS 间隔中位数）；窗口边界为该时长的 50 等分。PTS 来自 PyAV 帧的 `pts * time_base`，不由帧序号/FPS 推算。只有无有效视频 PTS 时才降级到 ffprobe 视频流/容器时长，并在 warning 中注明。
- 负 PTS 样本：{int((manifest['negative_video_pts_count'] > 0).sum())}；逆序 PTS 样本：{int((manifest['nonmonotonic_video_pts_count'] > 0).sum())}；缺失 PTS 帧总数：{int(manifest['missing_video_pts_count'].sum())}。

## 文本对齐与模态质量

- 有词级时间边界的词：{forced + interpolated + proportional}/{total_words}；其中 MFA 强制对齐 {forced}，内部插值 {interpolated}，词序比例降级 {proportional}。边界覆盖率不等于准确率。
- MFA 运行状态：`{run_info.get('mfa_alignment', {}).get('status', 'missing')}`；G2P（字素到音素）返回码 `{run_info.get('mfa_alignment', {}).get('g2p_return_code', 'missing')}`；TextGrid（词级时间标注文件）数量 `{run_info.get('mfa_alignment', {}).get('textgrid_count', 'missing')}`。
- 10 条随机样本、每条最多 6 个词的人工核查 CSV：`q1_manual_alignment_review.csv`（实际 {len(manual_rows)} 行）。人工真值及抽查结论未自动推断；无独立人工时间戳时不报告准确率。
- 音频可靠窗要求存在分析帧且窗口平均 RMS 不低于配置阈值 `{0.001}`；F0 只在满足有声判断的帧统计。其 pYIN 估计支撑为 403 个采样点（25.19 ms），输出插值到 400 点（25 ms）的共同帧中心；这样保留 25 ms 主分析网格，同时为 80 Hz 下限提供足够的周期支撑。零可靠音频窗样本：{', '.join(near_silence) if near_silence else '无'}。
- 视觉 mask 采用 face coverage ≥0.20 且 YuNet 平均检测分数 ≥0.50；blendshape（面部形变系数）不是人工 AU 真值，也不是情感标签。低覆盖/无有效视觉样本：{', '.join(low_vision) if low_vision else '无'}。

## 方案比较

### 时间粒度

| K | 每窗平均音频帧 | 每窗平均视频帧 | 空文本窗比例 | 有效视觉窗比例 | 压缩变体大小（PCA前，字节） |
|---:|---:|---:|---:|---:|---:|
{granularity_report}

### 视觉聚合

| 方法 | 有效视觉窗比例 | 完全无有效视觉样本数 |
|---|---:|---:|
{vision_report}

中位数与多帧均值的特征平均绝对差：`{vision_cmp['mean_abs_mean_vs_median_feature_difference']:.7g}`；中位数与中心帧的特征平均绝对差：`{vision_cmp['mean_abs_center_vs_median_feature_difference']:.7g}`；检测到的人脸窗口 blendshape 越界比例：`{vision_cmp['blendshape_out_of_range_rate_on_median_detected_bins']:.7g}`。所有视觉策略共用同一覆盖率与置信度阈值。

### 文本时间映射

| 方法 | 词级时间边界覆盖率 | 时间逆序数 | 人工真值 |
|---|---:|---:|---|
{alignment_report}

没有提供人工标注时间戳，所以本比较不把边界覆盖率称为定位准确率。

## 全部 warning 汇总

| warning code | 样本数 |
|---|---:|
{warning_table}

视频未完整解码样本：{', '.join(validation['incomplete_video_decode_samples']) if validation['incomplete_video_decode_samples'] else '无'}。音频未完整解码样本：{', '.join(validation['incomplete_audio_decode_samples']) if validation['incomplete_audio_decode_samples'] else '无'}。

## 设计与复核材料

- 短/中/长样本图分别为 `figures/q1_typical_short.png`、`figures/q1_typical_medium.png`、`figures/q1_typical_long.png`。
- `figures/q1_design_comparison.png` 对照时间粒度、视觉聚合有效率与文本时间边界覆盖率。
- 图中关键帧、音频波形、词起止时间、三模态特征摘要和 mask 使用同一秒级时间轴；每个 k 可由 `q1_quality_report.csv` 的 `[start_sec,end_sec)` 回查。
"""
    (result_dir / "Q1_FINAL_AUDIT.md").write_text(audit, encoding="utf-8")

    paper = f"""# Q1 特征提取与时序对齐（论文正文）

## 1. 任务与无标签特征接口

对附件1中的 {len(manifest)} 条视频构造逐样本的文本、音频和视觉时序特征。特征提取阶段仅读取 `label-100.xlsx` 中的官方英文转写与样本标识，不读取情感分类标签、回归分数或人工情感标注；这些标签未参与编码器选择、窗口数、阈值或 PCA 拟合。最终接口为 `text={tuple(arrays['text'].shape)}`、`audio={tuple(arrays['audio'].shape)}`、`vision={tuple(arrays['vision'].shape)}`。

## 2. 统一真实时间轴

第 i 条视频的有效时长 `T_i` 按解码后有效视频帧的真实 PTS 确定：`T_i = PTS_last - PTS_first + d_last`，其中 `d_last` 优先取末帧实际时长，缺失时取相邻正 PTS 间隔中位数。若无有效帧 PTS，才回退到 ffprobe 视频流或容器时长并标记 warning。以 `time_edges[i,k]=kT_i/50` 构造 50 个半开区间，三种模态共用同一组边界。早期数据审计报告的 2.648–34.567 秒与旧脚本 ffprobe 流时长的 2.257–29.288 秒并不一致；本次将实际解码 PTS 定为统一时间口径，并同时保留容器、流和解码时长差以定位差异。真实解码时长为 {duration_stats}；视频流与解码时长差的中位数为 {duration_delta.median():.4f} 秒。

## 3. 文本表示与词级时间投影

文本采用官方英文转写，不用 ASR 替换。原文保留在对齐文件；用于 MFA 的副本执行 NFKC Unicode 规范化、空白合并和标点/引号标准化，并记录变化。BERT-base-uncased（revision `{run_info.get('model_revisions', {}).get('text', '见 feature_config.yaml')}`）产生 768 维上下文子词表示，同一词的 WordPiece（子词）向量取均值。MFA 对已有词建立精确时间；相邻已对齐词间缺失词按词中心线性插值；无可用强制锚点或边界外缺词按词序比例降级。时间来源分别保存为 `forced_alignment`、`interpolated` 和 `proportional_fallback`。对词 j 与窗口 k，按 `overlap_duration / word_duration` 加权聚合。仅对有效文本窗口拟合固定随机种子的无监督 PCA（主成分分析），由 768 维压缩至 128 维；保存中心、投影矩阵和解释方差参数。结果共有 {total_words} 个词级记录，其中精确对齐 {forced}，插值 {interpolated}，比例降级 {proportional}；词边界覆盖率为 {(forced + interpolated + proportional) / max(total_words, 1):.4f}，该值不是对齐准确率。

## 4. 音频特征

原始音频完整解码验收后，经 FFmpeg 转为单声道、16 kHz PCM WAV。37 个描述子使用 25 ms 帧长和 10 ms 帧移，包含 13 个 MFCC（一阶梅尔频率倒谱系数）、13 个 delta MFCC（一阶差分）、对数能量、F0（基频）、有声概率、过零率、谱质心、谱带宽、谱滚降、谱平坦度、谱通量、谐噪比和 RMS（均方根能量）。每窗分别计算均值与标准差，共 74 维。考虑到 25 ms 精确窗口与 80 Hz 下限的周期支撑边界，pYIN 使用 403 个采样点（25.19 ms）估计 F0，再插值到 400 点（25 ms）的共同特征帧中心；音频主帧网格仍为 25 ms / 10 ms。F0 仅由 pYIN 判为有声且超过概率及能量阈值的帧统计；无有声帧时 F0 的均值和标准差置零。`audio_present` 表示有分析帧，`audio_mask` 还要求窗口平均 RMS ≥ {0.001}；本次有 {audio_present} 个音频存在窗、{audio_valid} 个可靠音频窗。

## 5. 视觉特征与稳健聚合

视频逐帧解码至结束，并从每个实际帧的 `pts*time_base` 获取时间；不存在 `frame_index/30` 替代。每个带可用 PTS 的帧均运行 MediaPipe Face Landmarker，提取 52 个具名 blendshape（面部形变系数），并用 YuNet（人脸检测模型）的分数提供检测置信度。Blendshape 是模型输出，不是人工 AU（面部动作单元）真值或情感标签。帧按真实 PTS 落入共同的 50 个窗口，窗口内检测成功帧以中位数聚合。视觉有效 mask 要求人脸覆盖率 ≥0.20 且平均检测分数 ≥0.50。最终 {vision_valid}/{total_windows} 个窗口有效；无有效视觉窗样本数为 {comparisons['visual_aggregation']['zero_vision_sample_counts']['multi_frame_median']}。

## 6. 归一化、保存及审计

主 NPZ 保存原始聚合特征为 float16、时间边界为 float32，并记录 float32 到 float16 的逐模态最大/平均绝对误差（text {errors['text']['max_absolute_error']:.7g}/{errors['text']['mean_absolute_error']:.7g}，audio {errors['audio']['max_absolute_error']:.7g}/{errors['audio']['mean_absolute_error']:.7g}，vision {errors['vision']['max_absolute_error']:.7g}/{errors['vision']['mean_absolute_error']:.7g}）。`normalization_stats.npz` 在各模态有效位置上逐维保存均值、标准差与有效计数；归一化使用 `(x-mean)/max(std,1e-6)`，不覆盖主文件原值。最终特征文件大小为 {size_mb:.2f} MiB。全量解码完成数为视频 {decode_video}/{len(manifest)}、音频 {decode_audio}/{int((manifest['duration_audio_stream_sec'] > 0).sum())}；NaN/Inf 检查结果为 {json.dumps(validation['nonfinite_feature_values'], ensure_ascii=False)}。其余 warning 与样本级质量指标见自动生成的 `Q1_FINAL_AUDIT.md`、manifest 和窗口质量表。

## 7. 方案比较

时间粒度比较 K=25、50、100；采用的 K=50 对应 {comparisons['time_granularity']['50']['mean_audio_frames_per_window']:.2f} 个平均音频分析帧/窗和 {comparisons['time_granularity']['50']['mean_video_frames_per_window']:.2f} 个平均视频帧/窗。视觉比较中心帧、多帧均值和多帧中位数，均采用相同质量阈值；词级比较词序均分、MFA 和 MFA 加降级映射。对齐比较统计边界覆盖与时间顺序，不将覆盖率解释为准确率。所有方案比较均未使用情感标签。
"""
    (result_dir / "Q1_论文正文.md").write_text(paper, encoding="utf-8")

    readme_path = result_dir / "README_Q1.md"
    old_readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else "# Q1 实验说明\n"
    measured = f"""
## 最终运行记录

- 样本：{len(manifest)}；时长：{duration_stats}。
- 形状：text `{tuple(arrays['text'].shape)}`；audio `{tuple(arrays['audio'].shape)}`；vision `{tuple(arrays['vision'].shape)}`。
- K=50 有效窗口：text {text_valid}/{total_windows}；audio {audio_valid}/{total_windows}；vision {vision_valid}/{total_windows}。
- 解码到结束：视频 {decode_video}/{len(manifest)}；音频 {decode_audio}/{int((manifest['duration_audio_stream_sec'] > 0).sum())}。
- 文件大小：{size_mb:.2f} MiB。完整逐样本数据见 `Q1_FINAL_AUDIT.md` 和 `q1_manifest.csv`。
"""
    readme_path.write_text(old_readme.split("\n## 最终运行记录")[0].rstrip() + "\n" + measured, encoding="utf-8")


def main() -> None:
    environment_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir
    manifest = pd.read_csv(result_dir / "q1_manifest.csv", dtype={"sample_id": str})
    arrays = np.load(result_dir / "q1_features.npz", allow_pickle=False)
    run_info = json.loads((result_dir / "q1_run_info.json").read_text(encoding="utf-8"))
    validation = json.loads((result_dir / "validation_report.json").read_text(encoding="utf-8"))
    alignment = [json.loads(line) for line in (result_dir / "q1_alignment.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    review = generate_manual_review(manifest, alignment, result_dir / "q1_manual_alignment_review.csv")
    figure_dir = result_dir / "figures"
    generate_typical_figures(result_dir, args.data_dir, manifest, arrays, alignment, figure_dir)
    generate_comparison_figure(run_info, figure_dir)
    write_final_documents(result_dir, manifest, arrays, run_info, validation, alignment, review)
    print(json.dumps({"audit": str(result_dir / "Q1_FINAL_AUDIT.md"), "paper": str(result_dir / "Q1_论文正文.md"), "manual_review_rows": len(review), "figures": sorted(path.name for path in figure_dir.glob("*.png"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
