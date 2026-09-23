#!/usr/bin/env python3
"""Extract and align Q1 text/audio/vision features for the 100 Attachment 1 clips.

The workbook's answer label and annotation columns are intentionally not used.
All feature arrays share clip-relative 0.5-second windows and explicit modality masks.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ROOT = Path(
    os.environ.get(
        "Q1_DATA_DIR",
        REPO_ROOT
        / "data"
        / "E题数据"
        / "附件1-数据集原始多模态样本"
        / "MOSEI数据集部分原始视频-100条",
    )
)
DEFAULT_BASE = Path(os.environ.get("Q1_BASE_DIR", SCRIPT_DIR))
MODEL_DIR = Path(os.environ.get("Q1_MODEL_DIR", REPO_ROOT / "models"))
DEFAULT_ACOUSTIC = MODEL_DIR / "english_us_arpa_acoustic.zip"
DEFAULT_DICTIONARY = MODEL_DIR / "english_us_arpa_dictionary.dict"
DEFAULT_FACE_MODEL = MODEL_DIR / "face_landmarker.task"
DEFAULT_TEXT_MODEL = MODEL_DIR / "bert_uncased_L-2-H-128_A-2"

SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms at 16 kHz
FFT_SIZE = 512
HOP_LENGTH = 160  # 10 ms at 16 kHz
TEXT_DIM_FALLBACK = 128
AUDIO_FRAME_NAMES = [
    *[f"mfcc_c{i:02d}" for i in range(13)],
    "rms",
    "zero_crossing_rate",
    "yin_f0_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "align", "extract", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BASE / "work")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE / "results")
    parser.add_argument("--acoustic-model", type=Path, default=DEFAULT_ACOUSTIC)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--g2p-model", type=str, default="english_us_arpa")
    parser.add_argument("--beam", type=int, default=100)
    parser.add_argument("--retry-beam", type=int, default=400)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--sample-id", type=str, default=None, help="Extract one prepared sample for a focused review")
    return parser.parse_args()


def setup_logging(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(work_dir / "run_q1.log", encoding="utf-8"),
        ],
        force=True,
    )


def run_command(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def clip_id_text(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def clean_transcript(raw_text: str) -> tuple[list[str], list[str]]:
    """Return MFA/tokenizer words plus removed bracketed editorial spans."""
    stripped: list[str] = []

    def remove_bracket(match: re.Match[str]) -> str:
        stripped.append(match.group(1).strip())
        return " "

    text = re.sub(r"\[([^\]]*)\]", remove_bracket, raw_text)
    text = text.replace("’", "'").replace("‘", "'").replace("–", " ").replace("—", " ")
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text.lower())
    return words, stripped


def probe_clip(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not on PATH; activate the math Conda environment")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels,start_time,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed: {path}")
    raw = json.loads(result.stdout)
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt_duration = float(raw.get("format", {}).get("duration") or 0.0)
    return {
        "duration_video_sec": float(video.get("duration") or fmt_duration),
        "duration_audio_sec": float(audio.get("duration") or 0.0),
        "video_start_sec": float(video.get("start_time") or 0.0),
        "audio_start_sec": float(audio.get("start_time") or 0.0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": video.get("avg_frame_rate") or video.get("r_frame_rate") or "",
        "audio_sample_rate_source": int(audio.get("sample_rate") or 0),
        "audio_channels_source": int(audio.get("channels") or 0),
        "has_video": bool(video),
        "has_audio": bool(audio),
        "duration_container_sec": fmt_duration,
    }


def load_rows(data_dir: Path) -> list[dict[str, Any]]:
    import pandas as pd

    workbook = data_dir / "label-100.xlsx"
    # Keep answer labels and annotations outside the feature-generation path.
    table = pd.read_excel(workbook, sheet_name="label", usecols=["video_id", "clip_id", "text"])
    required = {"video_id", "clip_id", "text"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"label sheet is missing required columns: {sorted(missing)}")
    if len(table) != 100:
        raise ValueError(f"Expected 100 Attachment 1 rows, found {len(table)}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in table.iterrows():
        video_id = str(row["video_id"]).strip()
        clip_id = clip_id_text(row["clip_id"])
        key = (video_id, clip_id)
        if key in seen:
            raise ValueError(f"Duplicate workbook key: {video_id}/{clip_id}")
        seen.add(key)
        raw_text = "" if row["text"] is None else str(row["text"])
        words, removed = clean_transcript(raw_text)
        source_path = data_dir / video_id / f"{clip_id}.mp4"
        row_id = f"{video_id}/{clip_id}"
        rows.append(
            {
                "sample_id": row_id,
                "video_id": video_id,
                "clip_id": clip_id,
                "source_relpath": source_path.relative_to(data_dir).as_posix(),
                "source_path": str(source_path),
                "raw_text": raw_text,
                "alignment_text": " ".join(words),
                "alignment_tokens": words,
                "removed_bracket_annotations": removed,
            }
        )
    return rows


def prepare_corpus(args: argparse.Namespace) -> list[dict[str, Any]]:
    corpus_dir = args.work_dir / "mfa_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(load_rows(args.data_dir), start=1):
        rec = dict(row)
        clip_path = Path(row["source_path"])
        rec["prepare_status"] = "ok"
        rec["error"] = ""
        try:
            if not clip_path.is_file():
                raise FileNotFoundError(f"Missing clip: {clip_path}")
            rec.update(probe_clip(clip_path))
            safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", rec["video_id"])
            safe_clip = re.sub(r"[^A-Za-z0-9_-]", "_", rec["clip_id"])
            speaker_dir = corpus_dir / safe_video
            speaker_dir.mkdir(parents=True, exist_ok=True)
            wav_path = speaker_dir / f"{safe_clip}.wav"
            text_path = speaker_dir / f"{safe_clip}.txt"
            rec["wav_path"] = str(wav_path)
            rec["textgrid_path"] = str(args.work_dir / "mfa_aligned" / safe_video / f"{safe_clip}.TextGrid")
            if rec["has_audio"] and rec["alignment_tokens"]:
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise RuntimeError("ffmpeg is not on PATH; activate the math Conda environment")
                cmd = [
                    ffmpeg,
                    "-nostdin",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(clip_path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ]
                result = run_command(cmd, timeout=180)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "FFmpeg audio conversion failed")
                text_path.write_text(rec["alignment_text"] + "\n", encoding="utf-8")
            else:
                rec["prepare_status"] = "no_audio_or_empty_transcript"
        except Exception as exc:  # preserve a row for every supplied sample
            rec["prepare_status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            logging.exception("Prepare failed for %s", rec["sample_id"])
        records.append(rec)
        logging.info("Prepared %d/100 %s (%s)", index, rec["sample_id"], rec["prepare_status"])

    save_jsonl(args.work_dir / "records.jsonl", records)
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align_corpus(args: argparse.Namespace) -> None:
    corpus = args.work_dir / "mfa_corpus"
    output = args.work_dir / "mfa_aligned"
    output.mkdir(parents=True, exist_ok=True)
    for model_path in (args.acoustic_model, args.dictionary):
        if not model_path.is_file():
            raise FileNotFoundError(f"MFA model file not found: {model_path}")
    mfa = shutil.which("mfa")
    if not mfa:
        raise RuntimeError("mfa is not on PATH; run this script with `conda run -n math`")

    g2p_dictionary = args.work_dir / "mfa_g2p_oov.dict"
    g2p_log = args.work_dir / "mfa_g2p.log"
    g2p_command = [
        mfa,
        "g2p",
        str(corpus),
        args.g2p_model,
        str(g2p_dictionary),
        "--dictionary_path",
        str(args.dictionary),
        "--num_jobs",
        str(args.num_jobs),
        "--overwrite",
    ]
    logging.info("Generating pronunciations for words missing from the base dictionary")
    g2p_result = run_command(g2p_command, timeout=7200)
    g2p_log.write_text(
        "COMMAND: " + " ".join(g2p_command) + "\n\nSTDOUT:\n" + g2p_result.stdout + "\nSTDERR:\n" + g2p_result.stderr,
        encoding="utf-8",
    )
    if g2p_result.returncode:
        raise RuntimeError(f"MFA G2P failed with exit code {g2p_result.returncode}; see {g2p_log}")

    augmented_dictionary = args.work_dir / "english_us_arpa_augmented.dict"
    with args.dictionary.open("rb") as base_file, augmented_dictionary.open("wb") as augmented_file:
        shutil.copyfileobj(base_file, augmented_file)
        if args.dictionary.stat().st_size:
            base_file.seek(-1, 2)
            if base_file.read(1) not in {b"\n", b"\r"}:
                augmented_file.write(b"\n")
        if g2p_dictionary.is_file():
            with g2p_dictionary.open("rb") as g2p_file:
                shutil.copyfileobj(g2p_file, augmented_file)
    command = [
        mfa,
        "align",
        str(corpus),
        str(augmented_dictionary),
        str(args.acoustic_model),
        str(output),
        "--clean",
        "--overwrite",
        "--num_jobs",
        str(args.num_jobs),
        "--beam",
        str(args.beam),
        "--retry_beam",
        str(args.retry_beam),
    ]
    logging.info("Running MFA on prepared corpus (%d jobs)", args.num_jobs)
    result = run_command(command, timeout=7200)
    (args.work_dir / "mfa_align.log").write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    status = {
        "return_code": result.returncode,
        "status": "ok" if result.returncode == 0 else "failed",
        "command": command,
        "log_path": str(args.work_dir / "mfa_align.log"),
        "g2p_model": args.g2p_model,
        "g2p_dictionary_path": str(g2p_dictionary),
        "augmented_dictionary_path": str(augmented_dictionary),
        "beam": args.beam,
        "retry_beam": args.retry_beam,
    }
    (args.work_dir / "mfa_align_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if result.returncode:
        logging.error("MFA returned %s; see %s", result.returncode, status["log_path"])
        raise RuntimeError(f"MFA alignment failed with exit code {result.returncode}; see {status['log_path']}")
    else:
        logging.info("MFA alignment finished; see %s", status["log_path"])


def parse_word_tier(textgrid_path: Path) -> list[tuple[float, float, str]]:
    if not textgrid_path.is_file():
        return []
    content = textgrid_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*item\s*\[\d+\]:", content)
    for block in blocks:
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match or name_match.group(1).strip().lower() != "words":
            continue
        entries: list[tuple[float, float, str]] = []
        pattern = re.compile(
            r'intervals\s*\[\d+\]:\s*\n\s*xmin\s*=\s*([^\n]+)\n\s*'
            r'xmax\s*=\s*([^\n]+)\n\s*text\s*=\s*"(.*)"'
        )
        for match in pattern.finditer(block):
            label = match.group(3).replace('""', '"').strip()
            if label and label.lower() not in {"sil", "sp", "spn"}:
                entries.append((float(match.group(1)), float(match.group(2)), label))
        return entries
    return []


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def map_aligned_words(tokens: list[str], intervals: list[tuple[float, float, str]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = [
        {"word": word, "start_sec": None, "end_sec": None, "aligned": False} for word in tokens
    ]
    cursor = 0
    for start, end, label in intervals:
        target = normalize_word(label)
        if not target:
            continue
        for token_index in range(cursor, len(tokens)):
            if normalize_word(tokens[token_index]) == target:
                mapped[token_index].update(
                    {"start_sec": float(start), "end_sec": float(end), "aligned": True, "mfa_label": label}
                )
                cursor = token_index + 1
                break
    return mapped


def embed_words(words: list[str], tokenizer: Any, model: Any, torch: Any, device: Any) -> tuple[np.ndarray, int]:
    hidden_size = int(model.config.hidden_size)
    embeddings = np.zeros((len(words), hidden_size), dtype=np.float32)
    if not words:
        return embeddings, hidden_size
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**model_inputs).last_hidden_state[0].detach().cpu().numpy()
    sums = np.zeros_like(embeddings)
    counts = np.zeros(len(words), dtype=np.int32)
    for subword_index, word_index in enumerate(word_ids):
        if word_index is not None and 0 <= word_index < len(words):
            sums[word_index] += output[subword_index]
            counts[word_index] += 1
    valid = counts > 0
    embeddings[valid] = sums[valid] / counts[valid, None]
    return embeddings, hidden_size


def aggregate_text(
    word_rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    n_bins: int,
    window: float,
    duration_video: float,
    alignment_offset: float,
    duration_audio: float,
) -> tuple[np.ndarray, np.ndarray]:
    dim = embeddings.shape[1] if embeddings.ndim == 2 else TEXT_DIM_FALLBACK
    sums = np.zeros((n_bins, dim), dtype=np.float64)
    weights = np.zeros(n_bins, dtype=np.float64)
    audio_start = alignment_offset
    audio_end = alignment_offset + duration_audio
    for word_index, row in enumerate(word_rows):
        if not row["aligned"] or word_index >= len(embeddings):
            continue
        start = max(0.0, float(row["start_sec"]) + alignment_offset, audio_start)
        end = min(duration_video, float(row["end_sec"]) + alignment_offset, audio_end)
        if end <= start:
            continue
        first = max(0, int(math.floor(start / window)))
        last = min(n_bins - 1, int(math.floor(max(start, end - 1e-9) / window)))
        for bin_index in range(first, last + 1):
            bin_start = bin_index * window
            bin_end = min(duration_video, bin_start + window)
            overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
            if overlap > 0:
                sums[bin_index] += embeddings[word_index] * overlap
                weights[bin_index] += overlap
    mask = weights > 0
    output = np.zeros((n_bins, dim), dtype=np.float32)
    output[mask] = (sums[mask] / weights[mask, None]).astype(np.float32)
    return output, mask


def extract_audio(
    wav_path: str | None,
    n_bins: int,
    window: float,
    duration_video: float,
    alignment_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    dim = len(AUDIO_FRAME_NAMES) * 2
    output = np.zeros((n_bins, dim), dtype=np.float32)
    mask = np.zeros(n_bins, dtype=bool)
    if not wav_path or not Path(wav_path).is_file() or n_bins == 0:
        return output, mask

    import librosa
    import soundfile as sf

    signal, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if sr != SAMPLE_RATE or len(signal) < FRAME_LENGTH:
        return output, mask
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13,
        n_fft=FFT_SIZE,
        win_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=False,
    )
    rms = librosa.feature.rms(y=signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False)
    zcr = librosa.feature.zero_crossing_rate(
        y=signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
    )
    f0 = librosa.yin(
        signal,
        fmin=70,
        fmax=400,
        sr=sr,
        frame_length=FFT_SIZE,
        hop_length=HOP_LENGTH,
        center=False,
    )
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sr, n_fft=FFT_SIZE, win_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=FFT_SIZE, win_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sr, n_fft=FFT_SIZE, win_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
    )
    flatness = librosa.feature.spectral_flatness(
        y=signal, n_fft=FFT_SIZE, win_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
    )
    feature_parts = [mfcc, rms, zcr, f0.reshape(1, -1), centroid, bandwidth, rolloff, flatness]
    shared_frame_count = min(part.shape[-1] for part in feature_parts)
    frame_features = np.vstack([part[..., :shared_frame_count] for part in feature_parts]).T.astype(np.float32)
    frame_features = np.nan_to_num(frame_features, nan=0.0, posinf=0.0, neginf=0.0)
    centers = (np.arange(frame_features.shape[0]) * HOP_LENGTH + FFT_SIZE / 2) / sr
    centers = centers + alignment_offset
    valid_centers = (centers >= 0.0) & (centers < duration_video)
    bin_indices = np.floor(np.maximum(centers, 0.0) / window).astype(int)
    for bin_index in range(n_bins):
        frames = frame_features[(bin_indices == bin_index) & valid_centers]
        if len(frames):
            output[bin_index, : len(AUDIO_FRAME_NAMES)] = frames.mean(axis=0)
            output[bin_index, len(AUDIO_FRAME_NAMES) :] = frames.std(axis=0)
            mask[bin_index] = True
    return output, mask


def extract_vision(
    clip_path: str,
    n_bins: int,
    duration_video: float,
    video_start_sec: float,
    window: float,
    landmarker: Any,
    mediapipe: Any,
    blendshape_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    values = np.zeros((n_bins, 52), dtype=np.float32)
    mask = np.zeros(n_bins, dtype=bool)
    frame_sampled = np.zeros(n_bins, dtype=bool)
    if n_bins == 0:
        return values, mask, frame_sampled, blendshape_names

    import av

    targets = np.minimum((np.arange(n_bins, dtype=np.float64) + 0.5) * window, max(0.0, duration_video - 1e-4))
    next_bin = 0
    try:
        with av.open(clip_path) as container:
            if not container.streams.video:
                return values, mask, frame_sampled, blendshape_names
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if next_bin >= n_bins:
                    break
                timestamp = frame.time
                if timestamp is None and frame.pts is not None:
                    timestamp = float(frame.pts * frame.time_base)
                if timestamp is not None:
                    timestamp = float(timestamp) - video_start_sec
                if timestamp is None or timestamp + 1e-7 < targets[next_bin]:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                frame_sampled[next_bin] = True
                if result.face_blendshapes:
                    categories = result.face_blendshapes[0]
                    names_now = [category.category_name for category in categories]
                    if blendshape_names is None:
                        blendshape_names = names_now
                    by_name = {category.category_name: float(category.score) for category in categories}
                    for j, name in enumerate(blendshape_names[:52]):
                        values[next_bin, j] = by_name.get(name, 0.0)
                    mask[next_bin] = True
                next_bin += 1
    except Exception:
        logging.exception("Video/face extraction failed for %s", clip_path)
    return values, mask, frame_sampled, blendshape_names


def package_versions() -> dict[str, str]:
    names = ["montreal-forced-aligner", "kalpy", "numpy", "pandas", "librosa", "soundfile", "torch", "transformers", "mediapipe", "av"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def external_tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        result = run_command([ffmpeg, "-version"], timeout=20)
        versions["ffmpeg"] = result.stdout.splitlines()[0] if result.stdout else "unknown"
    else:
        versions["ffmpeg"] = "not-found"
    return versions


def extract_all(args: argparse.Namespace) -> None:
    records = read_jsonl(args.work_dir / "records.jsonl")
    if len(records) != 100:
        raise ValueError(f"Expected 100 preparation records, found {len(records)}")
    if args.sample_id:
        records = [rec for rec in records if rec.get("sample_id") == args.sample_id]
        if not records:
            raise ValueError(f"Sample ID not found in prepared records: {args.sample_id}")
    for path in (args.face_model,):
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe face model not found: {path}")
    for path in (args.text_model,):
        if not path.is_dir():
            raise FileNotFoundError(f"Local text model directory not found: {path}")

    import mediapipe as mp
    import torch
    from transformers import AutoModel, AutoTokenizer

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(str(args.text_model), use_fast=True, local_files_only=True)
    text_model = AutoModel.from_pretrained(str(args.text_model), local_files_only=True).to(device)
    text_model.eval()
    text_dim = int(text_model.config.hidden_size)

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.face_model)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    window = float(args.window_sec)
    sample_outputs: list[dict[str, Any]] = []
    blendshape_names: list[str] | None = None
    text_dim = max(text_dim, 1)
    try:
        for index, rec in enumerate(records, start=1):
            duration_video = max(0.0, float(rec.get("duration_video_sec") or 0.0))
            duration_audio = max(0.0, float(rec.get("duration_audio_sec") or 0.0))
            n_bins = int(math.ceil(duration_video / window)) if duration_video > 0 else 0
            edges = np.zeros((n_bins, 2), dtype=np.float32)
            for b in range(n_bins):
                edges[b] = (b * window, min(duration_video, (b + 1) * window))

            tokens = rec.get("alignment_tokens", [])
            embeddings, _ = embed_words(tokens, tokenizer, text_model, torch, device)
            intervals = parse_word_tier(Path(rec.get("textgrid_path", "")))
            word_rows = map_aligned_words(tokens, intervals)
            alignment_offset = float(rec.get("audio_start_sec", 0.0)) - float(rec.get("video_start_sec", 0.0))
            text_values, text_mask = aggregate_text(
                word_rows,
                embeddings,
                n_bins,
                window,
                duration_video,
                alignment_offset,
                duration_audio,
            )
            audio_values, audio_mask = extract_audio(
                rec.get("wav_path"),
                n_bins,
                window,
                duration_video,
                alignment_offset,
            )
            vision_values, vision_mask, frame_sampled, blendshape_names = extract_vision(
                rec["source_path"],
                n_bins,
                duration_video,
                float(rec.get("video_start_sec", 0.0)),
                window,
                landmarker,
                mp,
                blendshape_names,
            )
            rec.update(
                {
                    "n_time_bins": n_bins,
                    "text_word_count": len(tokens),
                    "aligned_word_count": sum(bool(w["aligned"]) for w in word_rows),
                    "alignment_word_coverage": (
                        sum(bool(w["aligned"]) for w in word_rows) / len(tokens) if tokens else 0.0
                    ),
                    "alignment_status": "complete" if tokens and all(w["aligned"] for w in word_rows) else (
                        "partial" if any(w["aligned"] for w in word_rows) else "unavailable"
                    ),
                    "audio_valid_bins": int(audio_mask.sum()),
                    "vision_valid_bins": int(vision_mask.sum()),
                    "vision_frames_sampled": int(frame_sampled.sum()),
                    "text_dim": text_dim,
                    "audio_dim": int(audio_values.shape[1]),
                    "vision_dim": int(vision_values.shape[1]),
                    "window_sec": window,
                }
            )
            alignment_rows = []
            for word in word_rows:
                aligned = word["aligned"]
                start = end = None
                if aligned:
                    start = max(0.0, float(word["start_sec"]) + alignment_offset)
                    end = min(duration_video, float(word["end_sec"]) + alignment_offset)
                alignment_rows.append(
                    {
                        "sample_id": rec["sample_id"],
                        "word": word["word"],
                        "start_sec_video": start,
                        "end_sec_video": end,
                        "aligned": bool(aligned and end is not None and start is not None and end > start),
                        "method": "MFA forced word alignment" if aligned else "unavailable",
                    }
                )
            sample_outputs.append(
                {
                    "record": rec,
                    "edges": edges,
                    "text": text_values,
                    "text_mask": text_mask,
                    "audio": audio_values,
                    "audio_mask": audio_mask,
                    "vision": vision_values,
                    "vision_mask": vision_mask,
                    "time_mask": np.ones(n_bins, dtype=bool),
                    "alignment_rows": alignment_rows,
                }
            )
            logging.info(
                "Features %d/%d %s bins=%d words=%d/%d audio=%d vision=%d",
                index,
                len(records),
                rec["sample_id"],
                n_bins,
                rec["aligned_word_count"],
                rec["text_word_count"],
                rec["audio_valid_bins"],
                rec["vision_valid_bins"],
            )
    finally:
        landmarker.close()

    max_bins = max((len(item["edges"]) for item in sample_outputs), default=0)
    n = len(sample_outputs)
    arrays = {
        "sample_ids": np.asarray([x["record"]["sample_id"] for x in sample_outputs], dtype="U128"),
        "durations_video_sec": np.asarray([x["record"].get("duration_video_sec", 0.0) for x in sample_outputs], dtype=np.float32),
        "lengths": np.asarray([len(x["edges"]) for x in sample_outputs], dtype=np.int32),
        "window_sec": np.asarray(window, dtype=np.float32),
        "time_windows_sec": np.full((n, max_bins, 2), np.nan, dtype=np.float32),
        "time_mask": np.zeros((n, max_bins), dtype=bool),
        "text_features": np.zeros((n, max_bins, text_dim), dtype=np.float32),
        "text_mask": np.zeros((n, max_bins), dtype=bool),
        "audio_features": np.zeros((n, max_bins, len(AUDIO_FRAME_NAMES) * 2), dtype=np.float32),
        "audio_mask": np.zeros((n, max_bins), dtype=bool),
        "vision_features": np.zeros((n, max_bins, 52), dtype=np.float32),
        "vision_mask": np.zeros((n, max_bins), dtype=bool),
    }
    all_alignments: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    for i, item in enumerate(sample_outputs):
        length = len(item["edges"])
        arrays["time_windows_sec"][i, :length] = item["edges"]
        arrays["time_mask"][i, :length] = item["time_mask"]
        arrays["text_features"][i, :length] = item["text"]
        arrays["text_mask"][i, :length] = item["text_mask"]
        arrays["audio_features"][i, :length] = item["audio"]
        arrays["audio_mask"][i, :length] = item["audio_mask"]
        arrays["vision_features"][i, :length] = item["vision"]
        arrays["vision_mask"][i, :length] = item["vision_mask"]
        final_records.append(item["record"])
        all_alignments.extend(item["alignment_rows"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "q1_features.npz"
    np.savez_compressed(npz_path, **arrays)
    save_jsonl(args.output_dir / "q1_alignment.jsonl", all_alignments)
    manifest_path = args.output_dir / "q1_manifest.csv"
    columns = [
        "sample_id",
        "video_id",
        "clip_id",
        "source_relpath",
        "raw_text",
        "alignment_text",
        "prepare_status",
        "duration_video_sec",
        "duration_audio_sec",
        "video_start_sec",
        "audio_start_sec",
        "width",
        "height",
        "fps",
        "audio_sample_rate_source",
        "audio_channels_source",
        "n_time_bins",
        "window_sec",
        "text_word_count",
        "aligned_word_count",
        "alignment_word_coverage",
        "alignment_status",
        "audio_valid_bins",
        "vision_frames_sampled",
        "vision_valid_bins",
        "text_dim",
        "audio_dim",
        "vision_dim",
        "removed_bracket_annotations",
        "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in final_records:
            writer.writerow(
                {
                    **rec,
                    "removed_bracket_annotations": " | ".join(rec.get("removed_bracket_annotations", [])),
                }
            )

    metadata = {
        "sample_count": n,
        "window_sec": window,
        "max_time_bins": max_bins,
        "feature_shapes": {k: list(v.shape) for k, v in arrays.items() if k.endswith("features")},
        "feature_definitions": {
            "text": f"Contextual last-layer wordpiece mean embeddings from generic BERT, {text_dim} dimensions; aligned word vectors duration-weight averaged in each window.",
            "audio_frame_features": AUDIO_FRAME_NAMES,
            "audio": "20 frame descriptors (13 MFCC + RMS, zero-crossing rate, YIN F0, spectral centroid, bandwidth, rolloff, flatness); per-window mean and standard deviation = 40 dimensions.",
            "vision": blendshape_names or [],
            "vision_note": "MediaPipe Face Landmarker blendshape scores are facial-shape/action proxies, not emotion labels or AU ground truth.",
            "masks": "Separate time, text, audio, and detected-face masks; padded time windows contain NaN edges and zero feature vectors.",
        },
        "models": {
            "mfa_acoustic": str(args.acoustic_model),
            "mfa_dictionary_base": str(args.dictionary),
            "mfa_dictionary_augmented": str(args.work_dir / "english_us_arpa_augmented.dict"),
            "text_encoder": str(args.text_model),
            "face_landmarker": str(args.face_model),
        },
        "alignment_settings": {
            "beam": args.beam,
            "retry_beam": args.retry_beam,
            "g2p_model": args.g2p_model,
        },
        "device": str(device),
        "package_versions": package_versions(),
        "external_tool_versions": external_tool_versions(),
        "feature_archive_bytes": npz_path.stat().st_size,
    }
    (args.output_dir / "q1_run_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Wrote %s (%d bytes), %s, %s", npz_path, npz_path.stat().st_size, manifest_path, args.output_dir / "q1_alignment.jsonl")


def main() -> int:
    args = parse_args()
    setup_logging(args.work_dir)
    logging.info("Q1 stage=%s data=%s window=%.3fs", args.stage, args.data_dir, args.window_sec)
    if args.stage in {"prepare", "all"}:
        prepare_corpus(args)
    if args.stage in {"align", "all"}:
        align_corpus(args)
    if args.stage in {"extract", "all"}:
        extract_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
