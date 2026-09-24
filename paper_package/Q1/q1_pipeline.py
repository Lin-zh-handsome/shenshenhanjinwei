#!/usr/bin/env python3
"""Q1 frozen-spec feature extraction for the 100 Attachment 1 videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


K_FINAL = 50
K_COMPARISON = (25, 50, 100)
SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
FRAME_HOP = 160
FFT_SIZE = 512
N_MFCC = 13
PCA_DIM = 128
RANDOM_SEED = 20260923
FACE_COVERAGE_THRESHOLD = 0.20
FACE_CONFIDENCE_THRESHOLD = 0.50
AUDIO_RMS_THRESHOLD = 0.001
VOICED_PROBABILITY_THRESHOLD = 0.50
TEXT_MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
YUNET_MODEL_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get(
    "Q1_DATA_DIR",
    "data",
))
DEFAULT_MODEL_DIR = Path(os.environ.get("Q1_MODEL_DIR", "models"))
DEFAULT_BASE_DIR = Path(os.environ.get("Q1_BASE_DIR", "q1_run"))
DEFAULT_TEXT_MODEL = DEFAULT_MODEL_DIR / "q1_doc03/bert-base-uncased"
DEFAULT_FACE_MODEL = DEFAULT_MODEL_DIR / "face_landmarker.task"
DEFAULT_YUNET_MODEL = DEFAULT_MODEL_DIR / "q1_doc03/face_detection_yunet_2023mar.onnx"
DEFAULT_ACOUSTIC_MODEL = DEFAULT_MODEL_DIR / "english_us_arpa_acoustic.zip"
DEFAULT_DICTIONARY = DEFAULT_MODEL_DIR / "english_us_arpa_dictionary.dict"

TEXT_DIM = 768
TEXT_FEATURE_DIM = 128
AUDIO_FEATURE_NAMES = [
    *[f"mfcc_{i:02d}" for i in range(13)],
    *[f"delta_mfcc_{i:02d}" for i in range(13)],
    "log_energy", "f0_hz", "voicing_probability", "zero_crossing_rate",
    "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz",
    "spectral_flatness", "spectral_flux", "harmonic_to_noise_ratio_db", "rms_energy",
]
AUDIO_F0_INDEX = 27
AUDIO_VOICING_INDEX = 28
AUDIO_RMS_INDEX = 36
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "align", "extract"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BASE_DIR / "work")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_DIR / "results")
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--yunet-model", type=Path, default=DEFAULT_YUNET_MODEL)
    parser.add_argument("--acoustic-model", type=Path, default=DEFAULT_ACOUSTIC_MODEL)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--sample-id", help="One-sample extraction path for the requested smoke check")
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--beam", type=int, default=100)
    parser.add_argument("--retry-beam", type=int, default=400)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
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


def run_command(command: list[str], timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_transcript(raw_text: str) -> dict[str, Any]:
    """Create a reversible-audit alignment copy and character spans for each word."""
    text = unicodedata.normalize("NFKC", raw_text)
    changes: list[str] = []
    bracketed_text: list[str] = []
    if text != raw_text:
        changes.append("unicode_nfkc")

    def preserve_bracket_content(match: re.Match[str]) -> str:
        bracketed_text.append(match.group(1).strip())
        return f" {match.group(1)} "

    without_brackets = re.sub(r"\[([^\]]*)\]", preserve_bracket_content, text)
    if without_brackets != text:
        changes.append("bracket_delimiters_removed_but_content_preserved_in_alignment_copy")
    canonical = without_brackets.replace("’", "'").replace("‘", "'")
    canonical = canonical.replace("–", " ").replace("—", " ").replace("−", " ")
    canonical = canonical.replace("“", '"').replace("”", '"')
    if canonical != without_brackets:
        changes.append("quotes_and_dashes_canonicalized")
    tokens = [match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", canonical)]
    alignment_text = " ".join(tokens)
    if re.sub(r"\s+", " ", canonical).strip() != alignment_text:
        changes.append("punctuation_and_extra_whitespace_normalized")
    spans: list[dict[str, Any]] = []
    cursor = 0
    for token in tokens:
        start = alignment_text.find(token, cursor)
        end = start + len(token)
        spans.append({"word": token, "char_start": start, "char_end": end})
        cursor = end
    return {
        "alignment_text": alignment_text,
        "alignment_tokens": tokens,
        "word_spans": spans,
        "text_cleaning_changes": changes,
        "bracketed_text_preserved": bracketed_text,
    }


def clip_id_text(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def mfa_paths(work_dir: Path) -> tuple[Path, Path]:
    segments = [re.sub(r"[^A-Za-z0-9_-]+", "_", part) for part in work_dir.resolve().parts[-2:]]
    run_id_path = work_dir / "mfa_run_id.json"
    run_id = json.loads(run_id_path.read_text(encoding="utf-8")).get("run_id", "") if run_id_path.is_file() else ""
    run_name = "_".join(part for part in segments if part)
    if run_id:
        run_name += "_" + re.sub(r"[^A-Za-z0-9_-]+", "_", run_id)
    run_name = run_name or "q1_run"
    return work_dir / f"{run_name}_mfa_corpus", work_dir / f"{run_name}_mfa_aligned"


def probe_clip(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not on PATH")
    result = run_command([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels,start_time,duration,duration_ts,time_base",
        "-of", "json", str(path),
    ], timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    raw = json.loads(result.stdout)
    streams = raw.get("streams", [])
    video = next((row for row in streams if row.get("codec_type") == "video"), {})
    audio = next((row for row in streams if row.get("codec_type") == "audio"), {})
    def stream_duration(stream: dict[str, Any]) -> float:
        if stream.get("duration") not in (None, "N/A"):
            return float(stream["duration"])
        if stream.get("duration_ts") is not None and stream.get("time_base"):
            numerator, denominator = stream["time_base"].split("/")
            return float(stream["duration_ts"]) * int(numerator) / int(denominator)
        return 0.0
    container_duration = float(raw.get("format", {}).get("duration") or 0.0)
    return {
        "duration_container_sec": container_duration,
        "duration_video_stream_sec": stream_duration(video),
        "duration_audio_stream_sec": stream_duration(audio),
        "video_start_sec": float(video.get("start_time") or 0.0),
        "audio_start_sec": float(audio.get("start_time") or 0.0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps_nominal": video.get("avg_frame_rate") or video.get("r_frame_rate") or "",
        "fps_avg": video.get("avg_frame_rate") or "",
        "fps_r": video.get("r_frame_rate") or "",
        "original_audio_rate": int(audio.get("sample_rate") or 0),
        "original_audio_channels": int(audio.get("channels") or 0),
        "has_video_stream": bool(video),
        "has_audio_stream": bool(audio),
    }


def load_rows(data_dir: Path, sample_id: str | None = None) -> list[dict[str, Any]]:
    import pandas as pd
    workbook = data_dir / "label-100.xlsx"
    table = pd.read_excel(workbook, sheet_name="label", usecols=["video_id", "clip_id", "text"])
    if len(table) != 100:
        raise ValueError(f"Expected 100 workbook rows, found {len(table)}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, values in table.iterrows():
        video_id = str(values["video_id"]).strip()
        clip_id = clip_id_text(values["clip_id"])
        key = f"{video_id}/{clip_id}"
        if key in seen:
            raise ValueError(f"Duplicate sample ID: {key}")
        seen.add(key)
        raw_text = "" if values["text"] is None else str(values["text"])
        cleaned = normalize_transcript(raw_text)
        relpath = (Path(video_id) / f"{clip_id}.mp4").as_posix()
        row = {
            "sample_id": key,
            "video_id": video_id,
            "clip_id": clip_id,
            "source_relpath": relpath,
            "source_path": str(data_dir / video_id / f"{clip_id}.mp4"),
            "raw_text": raw_text,
            **cleaned,
        }
        rows.append(row)
    if sample_id:
        rows = [row for row in rows if row["sample_id"] == sample_id]
        if not rows:
            raise ValueError(f"Sample ID not found: {sample_id}")
    return rows


def prepare_corpus(args: argparse.Namespace) -> list[dict[str, Any]]:
    import soundfile as sf
    del sf
    corpus_dir, _ = mfa_paths(args.work_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    ffmpeg = shutil.which("ffmpeg")
    sources = load_rows(args.data_dir, args.sample_id)
    model_hashes = {
        "acoustic": sha256_file(args.acoustic_model),
        "dictionary": sha256_file(args.dictionary),
    }
    for index, source in enumerate(sources, start=1):
        rec = dict(source)
        rec["prepare_status"] = "ok"
        rec["warning_codes"] = []
        try:
            clip = Path(rec["source_path"])
            if not clip.is_file():
                raise FileNotFoundError("video_missing")
            rec.update(probe_clip(clip))
            rec["source_video_sha256"] = sha256_file(clip)
            safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", rec["video_id"])
            safe_clip = re.sub(r"[^A-Za-z0-9_-]", "_", rec["clip_id"])
            speaker_dir = corpus_dir / safe_video
            speaker_dir.mkdir(parents=True, exist_ok=True)
            wav_path = speaker_dir / f"{safe_clip}.wav"
            txt_path = speaker_dir / f"{safe_clip}.txt"
            rec["wav_path"] = str(wav_path)
            rec["text_path"] = str(txt_path)
            _, aligned_dir = mfa_paths(args.work_dir)
            rec["textgrid_path"] = str(aligned_dir / safe_video / f"{safe_clip}.TextGrid")
            txt_path.write_text(rec["alignment_text"] + "\n", encoding="utf-8")
            if not rec["has_audio_stream"]:
                raise RuntimeError("audio_stream_missing")
            if not ffmpeg:
                raise RuntimeError("ffmpeg_missing")
            command = [
                ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-xerror", "-i", str(clip), "-map", "0:a:0", "-vn", "-ac", "1",
                "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_path),
            ]
            converted = run_command(command, timeout=300)
            if converted.returncode:
                raise RuntimeError("ffmpeg_audio_decode_failed: " + converted.stderr[-500:])
            rec["wav_sha256"] = sha256_file(wav_path)
            rec["alignment_text_sha256"] = hashlib.sha256(rec["alignment_text"].encode("utf-8")).hexdigest()
            cache_input = {
                "sample_id": rec["sample_id"],
                "source_video_sha256": rec["source_video_sha256"],
                "wav_sha256": rec["wav_sha256"],
                "alignment_text_sha256": rec["alignment_text_sha256"],
                "mfa_model_hashes": model_hashes,
                "beam": args.beam,
                "retry_beam": args.retry_beam,
            }
            rec["mfa_cache_key"] = hashlib.sha256(
                json.dumps(cache_input, sort_keys=True).encode("utf-8")
            ).hexdigest()
        except Exception as exc:
            rec["prepare_status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["warning_codes"].append("prepare_failed")
            logging.exception("Preparation failed for %s", rec["sample_id"])
        records.append(rec)
        logging.info("Prepared %d/%d %s status=%s", index, len(sources), rec["sample_id"], rec["prepare_status"])
    save_jsonl(args.work_dir / "records.jsonl", records)
    return records


def align_corpus(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.work_dir / "records.jsonl")
    corpus, output = mfa_paths(args.work_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    mfa = shutil.which("mfa")
    status: dict[str, Any] = {
        "status": "failed", "return_code": None,
        "beam": args.beam, "retry_beam": args.retry_beam,
        "g2p_model": "english_us_arpa",
    }
    cache_entries: dict[str, str] = {}
    g2p_return_code: int | None = None
    try:
        if not mfa:
            raise RuntimeError("mfa_missing")
        g2p_path = args.work_dir / "mfa_g2p_oov.dict"
        g2p_command = [
            mfa, "g2p", str(corpus), "english_us_arpa", str(g2p_path),
            "--dictionary_path", str(args.dictionary), "--num_jobs", str(args.num_jobs),
            "--overwrite",
        ]
        g2p = run_command(g2p_command, timeout=7200)
        g2p_return_code = g2p.returncode
        (args.work_dir / "mfa_g2p.log").write_text(
            g2p.stdout + "\n" + g2p.stderr, encoding="utf-8"
        )
        if g2p.returncode:
            logging.warning("MFA G2P returned %s; base dictionary alignment will still be attempted", g2p.returncode)
        augmented = args.work_dir / "english_us_arpa_augmented.dict"
        with args.dictionary.open("rb") as source, augmented.open("wb") as target:
            shutil.copyfileobj(source, target)
            if args.dictionary.stat().st_size:
                source.seek(-1, 2)
                if source.read(1) not in {b"\n", b"\r"}:
                    target.write(b"\n")
            if g2p_path.is_file():
                with g2p_path.open("rb") as source:
                    shutil.copyfileobj(source, target)
        command = [
            mfa, "align", str(corpus), str(augmented), str(args.acoustic_model),
            str(output), "--clean", "--overwrite", "--num_jobs", str(args.num_jobs),
            "--beam", str(args.beam), "--retry_beam", str(args.retry_beam),
        ]
        aligned = run_command(command, timeout=7200)
        (args.work_dir / "mfa_align.log").write_text(
            aligned.stdout + "\n" + aligned.stderr, encoding="utf-8"
        )
        status.update({
            "return_code": aligned.returncode,
            "status": "ok" if aligned.returncode == 0 else "failed",
            "g2p_return_code": g2p_return_code,
            "textgrid_count": sum(Path(row["textgrid_path"]).is_file() for row in records),
        })
        if aligned.returncode:
            logging.error("MFA align failed with code %d; word-order fallback will retain all text", aligned.returncode)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("MFA stage failed; extraction will use word-order fallback")
    for row in records:
        textgrid = Path(row.get("textgrid_path", ""))
        if row.get("mfa_cache_key") and textgrid.is_file():
            cache_entries[row["sample_id"]] = row["mfa_cache_key"]
    save_json(args.work_dir / "alignment_cache.json", cache_entries)
    save_json(args.work_dir / "mfa_align_status.json", status)
    return status


def parse_word_tier(path: Path) -> list[tuple[float, float, str]]:
    if not path.is_file():
        return []
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*item\s*\[\d+\]:", content)
    for block in blocks:
        tier = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not tier or tier.group(1).strip().lower() != "words":
            continue
        pattern = re.compile(
            r'intervals\s*\[\d+\]:\s*\n\s*xmin\s*=\s*([^\n]+)\n\s*'
            r'xmax\s*=\s*([^\n]+)\n\s*text\s*=\s*"(.*)"'
        )
        output: list[tuple[float, float, str]] = []
        for match in pattern.finditer(block):
            word = match.group(3).replace('""', '"').strip()
            if word and word.lower() not in {"sil", "sp", "spn"}:
                output.append((float(match.group(1)), float(match.group(2)), word))
        return output
    return []


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def map_forced_words(tokens: list[str], spans: list[dict[str, Any]], intervals: list[tuple[float, float, str]]) -> list[dict[str, Any]]:
    rows = [
        {
            "word": word, "char_start": spans[i]["char_start"], "char_end": spans[i]["char_end"],
            "start_sec": None, "end_sec": None, "aligned": False,
            "timestamp_source": None, "confidence": None, "fallback_reason": None,
            "mfa_label": None,
        }
        for i, word in enumerate(tokens)
    ]
    cursor = 0
    for start, end, label in intervals:
        target = normalize_word(label)
        if not target or end <= start:
            continue
        for idx in range(cursor, len(tokens)):
            if normalize_word(tokens[idx]) == target:
                rows[idx].update({
                    "start_sec": float(start), "end_sec": float(end),
                    "aligned": True, "timestamp_source": "forced_alignment",
                    "mfa_label": label,
                })
                cursor = idx + 1
                break
    return rows


def add_word_time_fallback(
    words: list[dict[str, Any]], duration_used: float, audio_offset: float, audio_duration: float
) -> list[dict[str, Any]]:
    if not words or duration_used <= 0:
        return words
    lo = max(0.0, audio_offset)
    hi = min(duration_used, audio_offset + max(0.0, audio_duration))
    if hi <= lo:
        lo, hi = 0.0, duration_used
    forced = [i for i, row in enumerate(words) if row["aligned"]]
    def assign_proportional(indices: list[int], left_bound: float = lo, right_bound: float = hi) -> None:
        count = len(indices)
        span = max(right_bound - left_bound, 0.0)
        for order, i in enumerate(indices):
            start = left_bound + span * (order / max(count, 1))
            end = left_bound + span * ((order + 1) / max(count, 1))
            words[i].update({
                "start_sec": start, "end_sec": end, "aligned": False,
                "timestamp_source": "proportional_fallback", "confidence": None,
                "fallback_reason": "word-order proportional mapping within available neighboring anchor bounds",
            })
    if not forced:
        assign_proportional(list(range(len(words))))
        return words
    gaps: list[list[int]] = []
    cursor = 0
    for anchor in forced:
        if anchor > cursor:
            gaps.append(list(range(cursor, anchor)))
        cursor = anchor + 1
    if cursor < len(words):
        gaps.append(list(range(cursor, len(words))))
    for group in gaps:
        left = max((i for i in forced if i < group[0]), default=None)
        right = min((i for i in forced if i > group[-1]), default=None)
        if left is None and right is not None:
            assign_proportional(group, lo, float(words[right]["start_sec"]))
            continue
        if right is None and left is not None:
            assign_proportional(group, float(words[left]["end_sec"]), hi)
            continue
        if left is None and right is None:
            assign_proportional(group, lo, hi)
            continue
        left_center = (float(words[left]["start_sec"]) + float(words[left]["end_sec"])) / 2
        right_center = (float(words[right]["start_sec"]) + float(words[right]["end_sec"])) / 2
        if right_center <= left_center:
            assign_proportional(group)
            continue
        anchor_durations = [
            float(words[i]["end_sec"]) - float(words[i]["start_sec"])
            for i in forced
            if words[i]["end_sec"] > words[i]["start_sec"]
        ]
        typical = float(np.median(anchor_durations)) if anchor_durations else 0.08
        step = (right_center - left_center) / (len(group) + 1)
        word_duration = min(typical, max(0.002, step * 0.8))
        for order, i in enumerate(group, start=1):
            center = left_center + step * order
            start = max(lo, center - word_duration / 2)
            end = min(hi, center + word_duration / 2)
            if end <= start:
                assign_proportional([i])
            else:
                words[i].update({
                    "start_sec": start, "end_sec": end, "aligned": False,
                    "timestamp_source": "interpolated", "confidence": None,
                    "fallback_reason": "linear interpolation between adjacent aligned-word centers",
                })
    return words


def overlap_coverage(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals if min(end, b) > max(start, a))
    if not clipped:
        return 0.0
    total = 0.0
    left, right = clipped[0]
    for a, b in clipped[1:]:
        if a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    total += right - left
    return total / max(end - start, 1e-9)


def aggregate_text(
    words: list[dict[str, Any]], embeddings: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = len(edges) - 1
    dim = embeddings.shape[1]
    sums = np.zeros((bins, dim), dtype=np.float64)
    weights = np.zeros(bins, dtype=np.float64)
    timed_intervals: list[tuple[float, float]] = []
    for row in words:
        if row["start_sec"] is None or row["end_sec"] is None:
            row["overlapping_bins"] = []
            continue
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if end <= start:
            row["overlapping_bins"] = []
            continue
        timed_intervals.append((start, end))
        duration = end - start
        hits: list[int] = []
        for k in range(bins):
            overlap = max(0.0, min(end, float(edges[k + 1])) - max(start, float(edges[k])))
            if overlap > 0:
                weight = overlap / max(duration, 1e-9)
                sums[k] += embeddings[int(row["_word_index"])] * weight
                weights[k] += weight
                hits.append(k)
        row["overlapping_bins"] = hits
    mask = weights > 0
    output = np.zeros((bins, dim), dtype=np.float32)
    output[mask] = (sums[mask] / weights[mask, None]).astype(np.float32)
    coverage = np.asarray([
        overlap_coverage(timed_intervals, float(edges[k]), float(edges[k + 1]))
        for k in range(bins)
    ], dtype=np.float32)
    return output, mask, coverage


def embed_words(words: list[str], tokenizer: Any, model: Any, torch: Any, device: Any) -> np.ndarray:
    hidden = int(model.config.hidden_size)
    embeddings = np.zeros((len(words), hidden), dtype=np.float32)
    if not words:
        return embeddings
    sums = np.zeros_like(embeddings)
    counts = np.zeros(len(words), dtype=np.int32)
    full = tokenizer(words, is_split_into_words=True, add_special_tokens=True, truncation=False)
    if len(full["input_ids"]) <= int(model.config.max_position_embeddings):
        batches = [(0, words, tokenizer(
            words, is_split_into_words=True, add_special_tokens=True,
            truncation=False, return_tensors="pt",
        ))]
    else:
        batches = [
            (offset, words[offset:offset + 64], tokenizer(
                words[offset:offset + 64], is_split_into_words=True,
                add_special_tokens=True, truncation=True,
                max_length=int(model.config.max_position_embeddings), return_tensors="pt",
            ))
            for offset in range(0, len(words), 64)
        ]
    for offset, batch_words, encoded in batches:
        word_ids = encoded.word_ids(batch_index=0)
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[0].detach().cpu().numpy()
        for subword, local_word_index in enumerate(word_ids):
            if local_word_index is not None and local_word_index < len(batch_words):
                target = offset + local_word_index
                sums[target] += output[subword]
                counts[target] += 1
    valid = counts > 0
    embeddings[valid] = sums[valid] / counts[valid, None]
    return embeddings


def delta_features(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    if values.shape[0] <= 1:
        return output
    output[0] = values[1] - values[0]
    output[-1] = values[-1] - values[-2]
    if values.shape[0] > 2:
        output[1:-1] = (values[2:] - values[:-2]) / 2.0
    return output


def extract_audio_frames(wav_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return exactly 37 descriptors per 25-ms/10-ms frame and the voiced flags."""
    import librosa
    import soundfile as sf

    signal, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = np.asarray(signal, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected 16 kHz PCM, got {sample_rate}")
    if len(signal) < FRAME_LENGTH:
        return (
            np.zeros((0, 37), dtype=np.float32), np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool), {"wav_samples": len(signal), "pitch_status": "too_short"},
        )

    frames_view = np.lib.stride_tricks.sliding_window_view(signal, FRAME_LENGTH)[::FRAME_HOP]
    window = np.hanning(FRAME_LENGTH).astype(np.float32)
    frames = np.asarray(frames_view * window[None, :], dtype=np.float32)
    spectrum = np.fft.rfft(frames, n=FFT_SIZE, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = magnitude * magnitude

    mel_filter = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=40, fmin=0.0,
        fmax=SAMPLE_RATE / 2.0, norm="slaney", dtype=np.float32,
    )
    mel_power = mel_filter @ power.T
    log_mel = (10.0 * np.log10(np.maximum(mel_power, 1e-12))).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC, dct_type=2, norm="ortho").T
    delta = delta_features(mfcc)

    rms = np.sqrt(np.mean(np.square(frames_view), axis=1) + 1e-12).astype(np.float32)
    energy = np.log(np.sum(np.square(frames_view), axis=1) + 1e-12).astype(np.float32)
    zcr = np.mean(np.diff(np.signbit(frames_view), axis=1), axis=1).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE).astype(np.float32)
    power_sum = np.maximum(power.sum(axis=1), 1e-12)
    centroid = (power * frequencies[None, :]).sum(axis=1) / power_sum
    bandwidth = np.sqrt(
        (power * np.square(frequencies[None, :] - centroid[:, None])).sum(axis=1) / power_sum
    )
    cumulative = np.cumsum(power, axis=1)
    rolloff_index = np.argmax(cumulative >= (0.85 * power_sum[:, None]), axis=1)
    rolloff = frequencies[rolloff_index]
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=1)) / np.maximum(
        np.mean(power, axis=1), 1e-12
    )
    normalized_spectrum = magnitude / np.maximum(
        np.linalg.norm(magnitude, axis=1, keepdims=True), 1e-12
    )
    flux = np.zeros(len(frames_view), dtype=np.float32)
    if len(frames_view) > 1:
        flux[1:] = np.square(np.maximum(normalized_spectrum[1:] - normalized_spectrum[:-1], 0)).sum(axis=1)

    autocorrelation = np.fft.irfft(power, n=FFT_SIZE, axis=1)
    autocorrelation = autocorrelation / np.maximum(autocorrelation[:, :1], 1e-12)
    min_lag = max(1, int(SAMPLE_RATE / 400.0))
    max_lag = min(FFT_SIZE // 2, int(SAMPLE_RATE / 80.0))
    periodicity = np.clip(np.max(autocorrelation[:, min_lag:max_lag + 1], axis=1), 1e-5, 0.999)
    hnr = 10.0 * np.log10(periodicity / (1.0 - periodicity))

    pitch_status = "ok"
    try:
        pitch_frame_length = 403
        f0, voiced_flag, voiced_probability = librosa.pyin(
            signal, fmin=80.0, fmax=400.0, sr=SAMPLE_RATE,
            frame_length=pitch_frame_length, hop_length=FRAME_HOP,
            n_thresholds=25, resolution=0.1, center=False,
        )
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        voiced_probability = np.nan_to_num(
            np.asarray(voiced_probability, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        voiced_flag = np.asarray(voiced_flag, dtype=bool)
    except Exception as exc:
        f0 = np.zeros(len(frames_view), dtype=np.float32)
        voiced_probability = np.zeros(len(frames_view), dtype=np.float32)
        voiced_flag = np.zeros(len(frames_view), dtype=bool)
        pitch_status = f"pyin_failed:{type(exc).__name__}"
    frame_count = len(frames_view)
    pitch_centers = (np.arange(len(f0), dtype=np.float32) * FRAME_HOP + 201.5) / SAMPLE_RATE
    feature_centers = (np.arange(frame_count, dtype=np.float32) * FRAME_HOP + FRAME_LENGTH / 2.0) / SAMPLE_RATE
    if len(pitch_centers):
        f0 = np.interp(feature_centers, pitch_centers, f0, left=0.0, right=0.0).astype(np.float32)
        voiced_probability = np.interp(
            feature_centers, pitch_centers, voiced_probability, left=0.0, right=0.0
        ).astype(np.float32)
        nearest_pitch = np.clip(np.rint((feature_centers - pitch_centers[0]) * SAMPLE_RATE / FRAME_HOP).astype(int), 0, len(voiced_flag) - 1)
        voiced_flag = voiced_flag[nearest_pitch]
    else:
        f0 = np.zeros(frame_count, dtype=np.float32)
        voiced_probability = np.zeros(frame_count, dtype=np.float32)
        voiced_flag = np.zeros(frame_count, dtype=bool)
    voiced_probability = np.clip(voiced_probability, 0.0, 1.0)
    voiced = voiced_flag & (voiced_probability >= VOICED_PROBABILITY_THRESHOLD) & (rms >= AUDIO_RMS_THRESHOLD)
    f0[~voiced] = 0.0
    feature_frames = np.column_stack([
        mfcc, delta, energy, f0, voiced_probability, zcr, centroid, bandwidth,
        rolloff, flatness, flux, hnr, rms,
    ]).astype(np.float32)
    feature_frames = np.nan_to_num(feature_frames, nan=0.0, posinf=0.0, neginf=0.0)
    centers = feature_centers
    return feature_frames, centers, voiced, {
        "wav_samples": int(len(signal)),
        "wav_sample_rate": int(sample_rate),
        "analysis_frame_count": int(frame_count),
        "pitch_status": pitch_status,
    }


def assign_bins(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, times, side="right") - 1
    return np.clip(indices, 0, len(edges) - 2).astype(np.int32)


def aggregate_audio(
    frame_features: np.ndarray, frame_times: np.ndarray, voiced: np.ndarray,
    edges: np.ndarray, rms_threshold: float = AUDIO_RMS_THRESHOLD,
) -> dict[str, np.ndarray]:
    bins = len(edges) - 1
    values = np.zeros((bins, 74), dtype=np.float32)
    counts = np.zeros(bins, dtype=np.uint32)
    voiced_counts = np.zeros(bins, dtype=np.uint32)
    voiced_ratio = np.zeros(bins, dtype=np.float32)
    present = np.zeros(bins, dtype=np.uint8)
    reliable = np.zeros(bins, dtype=np.uint8)
    if not len(frame_features):
        return {
            "values": values, "counts": counts, "voiced_ratio": voiced_ratio,
            "voiced_counts": voiced_counts, "present": present, "reliable": reliable,
        }
    in_time = (frame_times >= edges[0]) & (frame_times < edges[-1] + 1e-7)
    bins_for_frame = assign_bins(frame_times[in_time], edges)
    used_features = frame_features[in_time]
    used_voiced = voiced[in_time]
    for k in range(bins):
        selected = bins_for_frame == k
        if not np.any(selected):
            continue
        rows = used_features[selected]
        voice = used_voiced[selected]
        counts[k] = len(rows)
        voiced_counts[k] = int(voice.sum())
        present[k] = 1
        means = rows.mean(axis=0)
        stds = rows.std(axis=0)
        voiced_f0 = rows[voice, AUDIO_F0_INDEX]
        if len(voiced_f0):
            means[AUDIO_F0_INDEX] = float(voiced_f0.mean())
            stds[AUDIO_F0_INDEX] = float(voiced_f0.std())
        else:
            means[AUDIO_F0_INDEX] = 0.0
            stds[AUDIO_F0_INDEX] = 0.0
        values[k, :37] = means
        values[k, 37:] = stds
        voiced_ratio[k] = float(voice.mean())
        reliable[k] = int(float(means[AUDIO_RMS_INDEX]) >= rms_threshold)
    return {
        "values": values, "counts": counts, "voiced_ratio": voiced_ratio,
        "voiced_counts": voiced_counts, "present": present, "reliable": reliable,
    }


def letterbox_face_image(bgr: np.ndarray, cv2: Any, size: int = 320) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    output[y:y + nh, x:x + nw] = resized
    return output


def create_face_landmarker(face_model: Path) -> Any:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def create_yunet_detector(yunet_model: Path) -> Any:
    import cv2
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("Installed OpenCV lacks FaceDetectorYN")
    return cv2.FaceDetectorYN.create(
        str(yunet_model), "", (320, 320), 0.0, 0.3, 5000
    )


def decode_video_and_extract_faces(
    clip_path: Path, landmarker: Any, mediapipe: Any, yunet: Any,
) -> dict[str, Any]:
    import av
    import cv2

    video_pts: list[float] = []
    face_features: list[np.ndarray] = []
    face_detected: list[bool] = []
    face_confidence: list[float] = []
    audio_pts: list[float] = []
    audio_end: list[float] = []
    warnings: list[str] = []
    negative_pts = 0
    nonmonotonic_pts = 0
    missing_video_pts = 0
    video_decode_complete = False
    audio_decode_complete = False
    decode_error = ""
    last_video_duration = 0.0
    decoded_video_count = 0
    decoded_audio_count = 0
    last_decoded_frame_has_pts = False
    try:
        with av.open(str(clip_path)) as container:
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if video_stream is None:
                raise RuntimeError("video_stream_missing")
            prev_pts: float | None = None
            streams = [video_stream] + ([audio_stream] if audio_stream is not None else [])
            for frame in container.decode(*streams):
                if isinstance(frame, av.video.frame.VideoFrame):
                    decoded_video_count += 1
                    timestamp = (
                        float(frame.pts * frame.time_base)
                        if frame.pts is not None and frame.time_base is not None else None
                    )
                    if timestamp is None:
                        last_decoded_frame_has_pts = False
                        missing_video_pts += 1
                        video_pts.append(float("nan"))
                    else:
                        last_decoded_frame_has_pts = True
                        if timestamp < -1e-7:
                            negative_pts += 1
                        if prev_pts is not None and timestamp < prev_pts - 1e-7:
                            nonmonotonic_pts += 1
                        prev_pts = timestamp
                        video_pts.append(timestamp)
                    last_video_duration = (
                        float(frame.duration * frame.time_base)
                        if frame.duration and frame.time_base else 0.0
                    )
                    bgr = frame.to_ndarray(format="bgr24")
                    small = letterbox_face_image(bgr, cv2, 320)
                    max_confidence = 0.0
                    try:
                        _, detections = yunet.detect(small)
                        if detections is not None and len(detections):
                            max_confidence = float(np.max(detections[:, 14]))
                    except Exception:
                        warnings.append("yunet_frame_error")
                    image = mediapipe.Image(
                        image_format=mediapipe.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    )
                    try:
                        result = landmarker.detect(image)
                        has_face = bool(result.face_blendshapes)
                        values = np.zeros(52, dtype=np.float32)
                        if has_face:
                            category_map = {
                                item.category_name: float(item.score)
                                for item in result.face_blendshapes[0]
                            }
                            values = np.asarray(
                                [category_map.get(name, 0.0) for name in BLENDSHAPE_NAMES],
                                dtype=np.float32,
                            )
                            if max_confidence <= 0:
                                warnings.append("yunet_score_missing_for_mediapipe_face")
                        face_features.append(values)
                        face_detected.append(has_face)
                        face_confidence.append(max_confidence if has_face else 0.0)
                    except Exception:
                        face_features.append(np.zeros(52, dtype=np.float32))
                        face_detected.append(False)
                        face_confidence.append(0.0)
                        warnings.append("mediapipe_frame_error")
                elif isinstance(frame, av.audio.frame.AudioFrame):
                    decoded_audio_count += 1
                    if frame.pts is None or frame.time_base is None:
                        continue
                    start = float(frame.pts * frame.time_base)
                    rate = float(frame.sample_rate or getattr(audio_stream, "rate", 0) or 0)
                    length = (frame.samples / rate) if rate > 0 else 0.0
                    audio_pts.append(start)
                    audio_end.append(start + length)
            video_decode_complete = decoded_video_count > 0
            audio_decode_complete = audio_stream is not None and decoded_audio_count > 0
    except Exception as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
        warnings.append("av_decode_incomplete")
    if missing_video_pts:
        warnings.append("video_pts_missing")
    if negative_pts:
        warnings.append("negative_video_pts")
    if nonmonotonic_pts:
        warnings.append("video_pts_nonmonotonic")
    pts = np.asarray(video_pts, dtype=np.float64)
    valid_pts = pts[np.isfinite(pts)]
    if len(valid_pts):
        first_video_pts = float(valid_pts[0])
        last_video_pts = float(valid_pts[-1])
        relative_pts = pts - first_video_pts
        differences = np.diff(valid_pts)
        positive_differences = differences[differences > 0]
        step = last_video_duration if last_decoded_frame_has_pts and last_video_duration > 0 else (
            float(np.median(positive_differences)) if len(positive_differences) else 0.0
        )
        duration_used = max(0.0, last_video_pts - first_video_pts + step)
    else:
        first_video_pts = None
        last_video_pts = None
        relative_pts = np.full(len(pts), np.nan, dtype=np.float64)
        duration_used = 0.0
    if len(audio_pts):
        first_audio_pts = float(audio_pts[0])
        last_audio_pts = float(audio_pts[-1])
        audio_duration_actual = max(audio_end) - first_audio_pts
    else:
        first_audio_pts = None
        last_audio_pts = None
        audio_duration_actual = 0.0
    audio_offset = (
        first_audio_pts - first_video_pts
        if first_audio_pts is not None and first_video_pts is not None else 0.0
    )
    face_matrix = np.asarray(face_features, dtype=np.float32).reshape((-1, 52))
    if len(face_matrix) != len(relative_pts):
        warnings.append("video_pts_feature_count_mismatch")
        common = min(len(face_matrix), len(relative_pts))
        face_matrix = face_matrix[:common]
        relative_pts = relative_pts[:common]
        face_detected = face_detected[:common]
        face_confidence = face_confidence[:common]
    return {
        "video_pts_absolute": pts,
        "video_pts_relative": relative_pts,
        "face_features": face_matrix,
        "face_detected": np.asarray(face_detected, dtype=bool),
        "face_confidence": np.asarray(face_confidence, dtype=np.float32),
        "decoded_video_frame_count": decoded_video_count,
        "decoded_audio_frame_count": decoded_audio_count,
        "first_decoded_video_pts_sec": first_video_pts,
        "last_decoded_video_pts_sec": last_video_pts,
        "first_decoded_audio_pts_sec": first_audio_pts,
        "last_decoded_audio_pts_sec": last_audio_pts,
        "duration_video_decoded_sec": duration_used,
        "duration_audio_decoded_sec": audio_duration_actual,
        "audio_offset_video_sec": audio_offset,
        "video_decode_complete": bool(video_decode_complete),
        "audio_decode_complete": bool(audio_decode_complete),
        "negative_video_pts_count": negative_pts,
        "nonmonotonic_video_pts_count": nonmonotonic_pts,
        "missing_video_pts_count": missing_video_pts,
        "decode_error": decode_error,
        "warnings": sorted(set(warnings)),
    }


def aggregate_visual(
    frame_times: np.ndarray,
    frame_features: np.ndarray,
    face_detected: np.ndarray,
    confidence: np.ndarray,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    bins = len(edges) - 1
    median_values = np.zeros((bins, 52), dtype=np.float32)
    mean_values = np.zeros_like(median_values)
    center_values = np.zeros_like(median_values)
    frame_count = np.zeros(bins, dtype=np.uint32)
    detected_count = np.zeros(bins, dtype=np.uint32)
    coverage = np.zeros(bins, dtype=np.float32)
    mean_confidence = np.zeros(bins, dtype=np.float32)
    median_mask = np.zeros(bins, dtype=np.uint8)
    mean_mask = np.zeros(bins, dtype=np.uint8)
    center_mask = np.zeros(bins, dtype=np.uint8)
    if not len(frame_times):
        return {
            "median": median_values, "mean": mean_values, "center": center_values,
            "frame_count": frame_count, "detected_count": detected_count,
            "coverage": coverage, "mean_confidence": mean_confidence,
            "median_mask": median_mask, "mean_mask": mean_mask,
            "center_mask": center_mask,
        }
    bins_for_frame = assign_bins(frame_times, edges)
    for k in range(bins):
        in_bin = bins_for_frame == k
        indices = np.flatnonzero(in_bin)
        frame_count[k] = len(indices)
        if not len(indices):
            continue
        detected_indices = indices[face_detected[indices]]
        detected_count[k] = len(detected_indices)
        coverage[k] = len(detected_indices) / len(indices)
        if len(detected_indices):
            scores = confidence[detected_indices]
            mean_confidence[k] = float(np.mean(scores))
            observations = frame_features[detected_indices]
            median_values[k] = np.median(observations, axis=0)
            mean_values[k] = np.mean(observations, axis=0)
            eligible = coverage[k] >= FACE_COVERAGE_THRESHOLD and (
                mean_confidence[k] >= FACE_CONFIDENCE_THRESHOLD
            )
            median_mask[k] = mean_mask[k] = int(eligible)
        center_time = (float(edges[k]) + float(edges[k + 1])) / 2.0
        closest = int(indices[np.argmin(np.abs(frame_times[indices] - center_time))])
        center_values[k] = frame_features[closest]
        center_mask[k] = int(
            face_detected[closest] and confidence[closest] >= FACE_CONFIDENCE_THRESHOLD
        )
    return {
        "median": median_values, "mean": mean_values, "center": center_values,
        "frame_count": frame_count, "detected_count": detected_count,
        "coverage": coverage, "mean_confidence": mean_confidence,
        "median_mask": median_mask, "mean_mask": mean_mask,
        "center_mask": center_mask,
    }


def make_edges(duration: float, count: int) -> np.ndarray:
    if not np.isfinite(duration) or duration <= 0:
        duration = 1e-3
    result = np.linspace(0.0, float(duration), count + 1, dtype=np.float64)
    result[0] = 0.0
    result[-1] = float(duration)
    return result


def source_word_rows(
    rec: dict[str, Any],
    textgrid: Path,
    duration_used: float,
    audio_offset: float,
    audio_duration: float,
    expected_cache: dict[str, str],
) -> list[dict[str, Any]]:
    tokens = rec.get("alignment_tokens", [])
    spans = rec.get("word_spans", [])
    cache_valid = (
        expected_cache.get(rec["sample_id"]) == rec.get("mfa_cache_key")
        and bool(rec.get("mfa_cache_key"))
        and textgrid.is_file()
    )
    intervals = parse_word_tier(textgrid) if cache_valid else []
    rows = map_forced_words(tokens, spans, intervals)
    for index, row in enumerate(rows):
        row["_word_index"] = index
        if row["aligned"]:
            start = float(row["start_sec"]) + audio_offset
            end = float(row["end_sec"]) + audio_offset
            start, end = max(0.0, start), min(duration_used, end)
            if end <= start:
                row.update({
                    "start_sec": None, "end_sec": None, "aligned": False,
                    "timestamp_source": None, "fallback_reason": "forced time outside decoded media bounds",
                })
            else:
                row["start_sec"], row["end_sec"] = start, end
    return add_word_time_fallback(rows, duration_used, audio_offset, audio_duration)


def process_sample(
    rec: dict[str, Any],
    args: argparse.Namespace,
    alignment_cache: dict[str, str],
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    landmarker: Any,
    mediapipe: Any,
    yunet: Any,
) -> dict[str, Any]:
    source_path = Path(rec["source_path"])
    decoded = decode_video_and_extract_faces(source_path, landmarker, mediapipe, yunet)
    duration_used = float(decoded["duration_video_decoded_sec"])
    duration_fallback_source = ""
    if duration_used <= 0:
        duration_used = float(rec.get("duration_video_stream_sec") or rec.get("duration_container_sec") or 0.0)
        duration_fallback_source = "ffprobe_stream_or_container_duration"
        decoded["warnings"].append("duration_used_fallback_from_stream_metadata")
    if duration_used <= 0:
        duration_used = 1e-3
        duration_fallback_source = "minimum_shape_fallback"
        decoded["warnings"].append("no_decoded_duration")
    audio_duration = float(decoded["duration_audio_decoded_sec"] or rec.get("duration_audio_stream_sec") or 0.0)
    rows = source_word_rows(
        rec, Path(rec.get("textgrid_path", "")), duration_used,
        float(decoded["audio_offset_video_sec"]), audio_duration, alignment_cache,
    )
    for index, row in enumerate(rows):
        row["_word_index"] = index
    tokens = rec.get("alignment_tokens", [])
    embeddings = embed_words(tokens, tokenizer, model, torch, device)
    if embeddings.shape[1] != int(model.config.hidden_size):
        raise ValueError("Text model hidden size mismatch")
    if len(tokens) and np.linalg.norm(embeddings, axis=1).sum() == 0:
        decoded["warnings"].append("text_encoder_returned_empty_word_vectors")

    audio_detail: dict[str, Any]
    try:
        frame_audio, frame_audio_centers, voiced, audio_detail = extract_audio_frames(rec["wav_path"])
    except Exception as exc:
        frame_audio = np.zeros((0, 37), dtype=np.float32)
        frame_audio_centers = np.zeros(0, dtype=np.float32)
        voiced = np.zeros(0, dtype=bool)
        audio_detail = {"pitch_status": f"audio_feature_error:{type(exc).__name__}"}
        decoded["warnings"].append("audio_feature_extraction_failed")
    frame_audio_centers = frame_audio_centers + float(decoded["audio_offset_video_sec"])
    variants: dict[int, dict[str, Any]] = {}
    timestamped_video = np.isfinite(decoded["video_pts_relative"])
    for k_count in K_COMPARISON:
        edges = make_edges(duration_used, k_count)
        word_copy = [dict(row) for row in rows]
        text_high, text_mask, text_coverage = aggregate_text(word_copy, embeddings, edges)
        audio = aggregate_audio(frame_audio, frame_audio_centers, voiced, edges)
        visual = aggregate_visual(
            decoded["video_pts_relative"][timestamped_video],
            decoded["face_features"][timestamped_video],
            decoded["face_detected"][timestamped_video],
            decoded["face_confidence"][timestamped_video], edges,
        )
        variants[k_count] = {
            "edges": edges, "text_high": text_high, "text_mask": text_mask,
            "text_coverage": text_coverage, "audio": audio, "visual": visual,
            "word_rows": word_copy,
        }
    main = variants[K_FINAL]
    alignment_rows: list[dict[str, Any]] = []
    for row in main["word_rows"]:
        alignment_rows.append({
            "sample_id": rec["sample_id"],
            "raw_text": rec["raw_text"],
            "normalized_alignment_text": rec["alignment_text"],
            "word": row["word"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "timestamp_source": row["timestamp_source"],
            "confidence": row.get("confidence"),
            "aligned": bool(row["aligned"]),
            "overlapping_bins": row.get("overlapping_bins", []),
            "fallback_reason": row.get("fallback_reason"),
            "alignment_cache_valid": bool(
                alignment_cache.get(rec["sample_id"]) == rec.get("mfa_cache_key")
            ),
        })
    forced_count = sum(row["timestamp_source"] == "forced_alignment" for row in alignment_rows)
    interpolated_count = sum(row["timestamp_source"] == "interpolated" for row in alignment_rows)
    fallback_count = sum(row["timestamp_source"] == "proportional_fallback" for row in alignment_rows)
    warning_list = list(rec.get("warning_codes", [])) + decoded["warnings"]
    if rows and forced_count == 0:
        warning_list.append("mfa_word_alignment_unavailable")
    if interpolated_count:
        warning_list.append("text_words_interpolated")
    if fallback_count:
        warning_list.append("text_words_proportional_fallback")
    if int(main["audio"]["reliable"].sum()) == 0:
        warning_list.append("no_reliable_audio_windows")
    if int(main["visual"]["median_mask"].sum()) == 0:
        warning_list.append("no_reliable_vision_windows")
    if str(audio_detail.get("pitch_status", "")).startswith("pyin_failed"):
        warning_list.append(audio_detail["pitch_status"])
    warnings = sorted(set(warning_list))
    video_stream = float(rec.get("duration_video_stream_sec") or 0.0)
    audio_stream = float(rec.get("duration_audio_stream_sec") or 0.0)
    video_count_all = int(decoded["decoded_video_frame_count"])
    face_coverage_global = (
        float(decoded["face_detected"].sum()) / video_count_all if video_count_all else 0.0
    )
    if face_coverage_global < FACE_COVERAGE_THRESHOLD:
        warnings = sorted(set(warnings + ["low_global_face_coverage"]))
    if not decoded["video_decode_complete"]:
        warnings = sorted(set(warnings + ["video_decode_incomplete"]))
    if rec.get("has_audio_stream") and not decoded["audio_decode_complete"]:
        warnings = sorted(set(warnings + ["audio_decode_incomplete"]))
    error_text = str(rec.get("error", "") or decoded["decode_error"])
    for private_root in (str(args.data_dir), str(args.work_dir), str(args.output_dir)):
        error_text = error_text.replace(private_root, "[local_path]")
    error_text = error_text.replace(str(source_path), "[source_video]")
    manifest = {
        "sample_id": rec["sample_id"],
        "video_id": rec["video_id"],
        "clip_id": rec["clip_id"],
        "relative_video_path": rec["source_relpath"],
        "duration_used_sec": duration_used,
        "duration_used_source": duration_fallback_source or "decoded_video_pts_span",
        "duration_container_sec": float(rec.get("duration_container_sec") or 0.0),
        "duration_video_stream_sec": video_stream,
        "duration_audio_stream_sec": audio_stream,
        "duration_video_decoded_sec": float(decoded["duration_video_decoded_sec"]),
        "duration_audio_decoded_sec": float(decoded["duration_audio_decoded_sec"]),
        "duration_container_minus_used_sec": float(rec.get("duration_container_sec") or 0.0) - duration_used,
        "duration_video_stream_minus_used_sec": video_stream - duration_used,
        "duration_audio_video_decoded_difference_sec": float(decoded["duration_audio_decoded_sec"]) - float(decoded["duration_video_decoded_sec"]),
        "video_start_sec": float(rec.get("video_start_sec") or 0.0),
        "audio_start_sec": float(rec.get("audio_start_sec") or 0.0),
        "first_video_pts_sec": decoded["first_decoded_video_pts_sec"],
        "last_video_pts_sec": decoded["last_decoded_video_pts_sec"],
        "first_audio_pts_sec": decoded["first_decoded_audio_pts_sec"],
        "last_audio_pts_sec": decoded["last_decoded_audio_pts_sec"],
        "decoded_frame_count": video_count_all,
        "decoded_audio_frame_count": int(decoded["decoded_audio_frame_count"]),
        "video_decode_complete": bool(decoded["video_decode_complete"]),
        "audio_decode_complete": bool(decoded["audio_decode_complete"]),
        "negative_video_pts_count": int(decoded["negative_video_pts_count"]),
        "nonmonotonic_video_pts_count": int(decoded["nonmonotonic_video_pts_count"]),
        "missing_video_pts_count": int(decoded["missing_video_pts_count"]),
        "fps_nominal": rec.get("fps_nominal", ""),
        "fps": rec.get("fps_avg", "") or rec.get("fps_r", ""),
        "fps_decoded": video_count_all / duration_used if duration_used > 0 else 0.0,
        "width": int(rec.get("width") or 0),
        "height": int(rec.get("height") or 0),
        "original_audio_rate": int(rec.get("original_audio_rate") or 0),
        "original_audio_channels": int(rec.get("original_audio_channels") or 0),
        "text_word_count": len(rows),
        "forced_aligned_word_count": forced_count,
        "interpolated_word_count": interpolated_count,
        "proportional_fallback_word_count": fallback_count,
        "text_word_boundary_coverage": (
            (forced_count + interpolated_count + fallback_count) / len(rows) if rows else 0.0
        ),
        "text_valid_bins": int(main["text_mask"].sum()),
        "audio_present_bins": int(main["audio"]["present"].sum()),
        "audio_reliable_bins": int(main["audio"]["reliable"].sum()),
        "audio_valid_bins": int(main["audio"]["reliable"].sum()),
        "voiced_ratio_global": (
            float(main["audio"]["voiced_counts"].sum() / main["audio"]["counts"].sum())
            if main["audio"]["counts"].sum() else 0.0
        ),
        "vision_valid_bins": int(main["visual"]["median_mask"].sum()),
        "face_coverage_global": face_coverage_global,
        "mean_face_confidence_global": float(
            decoded["face_confidence"][decoded["face_detected"]].mean()
        ) if decoded["face_detected"].any() else 0.0,
        "text_shape": f"1x{K_FINAL}x{TEXT_FEATURE_DIM}",
        "audio_shape": f"1x{K_FINAL}x74",
        "vision_shape": f"1x{K_FINAL}x52",
        "prepare_status": rec.get("prepare_status", "unknown"),
        "source_video_sha256": rec.get("source_video_sha256", ""),
        "wav_sha256": rec.get("wav_sha256", ""),
        "alignment_text_sha256": rec.get("alignment_text_sha256", ""),
        "mfa_cache_key": rec.get("mfa_cache_key", ""),
        "warning_codes": "|".join(warnings),
        "error": error_text,
        "status": "ok" if rec.get("prepare_status") == "ok" and decoded["video_decode_complete"] and (not rec.get("has_audio_stream") or decoded["audio_decode_complete"]) and not warnings else "degraded",
        "text_cleaning_changes": "|".join(rec.get("text_cleaning_changes", [])),
    }
    window_rows: list[dict[str, Any]] = []
    for k in range(K_FINAL):
        window_rows.append({
            "sample_id": rec["sample_id"],
            "k": k,
            "start_sec": float(main["edges"][k]),
            "end_sec": float(main["edges"][k + 1]),
            "text_coverage": float(main["text_coverage"][k]),
            "text_mask": int(main["text_mask"][k]),
            "audio_frame_count": int(main["audio"]["counts"][k]),
            "audio_present": int(main["audio"]["present"][k]),
            "audio_reliable": int(main["audio"]["reliable"][k]),
            "voiced_ratio": float(main["audio"]["voiced_ratio"][k]),
            "video_frame_count": int(main["visual"]["frame_count"][k]),
            "face_detected_count": int(main["visual"]["detected_count"][k]),
            "face_coverage": float(main["visual"]["coverage"][k]),
            "mean_face_confidence": float(main["visual"]["mean_confidence"][k]),
            "vision_mask": int(main["visual"]["median_mask"][k]),
        })
    return {
        "record": rec,
        "manifest": manifest,
        "variants": variants,
        "alignment_rows": alignment_rows,
        "window_rows": window_rows,
        "processing_log": {
            "sample_id": rec["sample_id"],
            "status": manifest["status"],
            "warnings": warnings,
            "video_decode_complete": manifest["video_decode_complete"],
            "audio_decode_complete": manifest["audio_decode_complete"],
            "decoded_video_frame_count": video_count_all,
            "decoded_audio_frame_count": manifest["decoded_audio_frame_count"],
            "decode_error": decoded["decode_error"],
            "audio_feature_details": audio_detail,
        },
    }


def model_metadata(args: argparse.Namespace) -> dict[str, Any]:
    files = [
        (args.text_model / "model.safetensors", "https://huggingface.co/google-bert/bert-base-uncased", TEXT_MODEL_REVISION),
        (args.text_model / "tokenizer.json", "https://huggingface.co/google-bert/bert-base-uncased", TEXT_MODEL_REVISION),
        (args.text_model / "vocab.txt", "https://huggingface.co/google-bert/bert-base-uncased", TEXT_MODEL_REVISION),
        (args.face_model, "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker", "face_landmarker.task model asset"),
        (args.yunet_model, "https://huggingface.co/opencv/face_detection_yunet", YUNET_MODEL_REVISION),
        (args.acoustic_model, "Montreal Forced Aligner english_us_arpa acoustic model", "local pinned model asset"),
        (args.dictionary, "Montreal Forced Aligner english_us_arpa dictionary", "local pinned dictionary asset"),
        (Path.home() / "Documents" / "MFA" / "pretrained_models" / "g2p" / "english_us_arpa.zip", "Montreal Forced Aligner pretrained G2P model", "local pinned model asset"),
    ]
    result = []
    for path, source, revision in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({
            "file": path.name, "source": source, "revision": revision,
            "sha256": sha256_file(path),
        })
    return {
        "files": result,
        "text_encoder": {
            "name": "BERT base uncased",
            "source": "google-bert/bert-base-uncased",
            "revision": TEXT_MODEL_REVISION,
            "tokenizer": "BertTokenizerFast, pinned with the same model revision",
            "fine_tuned": False,
            "important_parameters": {"hidden_size": TEXT_DIM, "contextual_layer": "last hidden state", "subword_pooling": "mean within each official transcript word"},
        },
        "visual_feature_model": {
            "name": "MediaPipe Face Landmarker",
            "source": "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
            "revision": "local model asset; SHA256 recorded in files list",
            "fine_tuned": False,
            "important_parameters": {"running_mode": "IMAGE", "num_faces": 1, "blendshapes": True},
        },
        "visual_confidence_model": {
            "name": "OpenCV Zoo YuNet face detection",
            "source": "https://huggingface.co/opencv/face_detection_yunet",
            "revision": YUNET_MODEL_REVISION,
            "fine_tuned": False,
            "important_parameters": {"input": "aspect-preserving 320x320 letterbox", "confidence_column": 14},
        },
        "forced_alignment": {
            "name": "Montreal Forced Aligner English US ARPA",
            "source": "local MFA acoustic and dictionary model assets",
            "revision": "acoustic, dictionary and G2P model SHA256 values recorded in files list",
            "fine_tuned": False,
            "important_parameters": {"beam": args.beam, "retry_beam": args.retry_beam, "g2p": "english_us_arpa"},
        },
    }


def software_metadata() -> dict[str, Any]:
    names = [
        "numpy", "pandas", "scikit-learn", "torch", "transformers",
        "tokenizers", "librosa", "soundfile", "av", "mediapipe", "opencv-python",
        "montreal-forced-aligner", "kalpy", "matplotlib", "Pillow", "openpyxl", "PyYAML",
    ]
    versions = {name: package_version(name) for name in names}
    try:
        import cv2
        versions["opencv_runtime"] = cv2.__version__
    except Exception:
        pass
    try:
        import av
        versions["pyav_runtime"] = av.__version__
    except Exception:
        pass
    for command, flag in (("ffmpeg", "-version"), ("ffprobe", "-version"), ("mfa", "--help")):
        path = shutil.which(command)
        if path:
            result = run_command([path, flag], timeout=30)
            versions[f"{command}_first_line"] = (
                (result.stdout or result.stderr).splitlines()[0][:250]
                if (result.stdout or result.stderr) else "available"
            )
        else:
            versions[f"{command}_first_line"] = "not-found"
    versions["python"] = sys.version.split()[0]
    return versions


def feature_configuration(args: argparse.Namespace, models: dict[str, Any], software: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_axis": {
            "K": K_FINAL, "comparison_K": list(K_COMPARISON),
            "definition": "time_edges[i,k] = k * duration_used_sec[i] / K",
            "duration_used": "last decoded video PTS minus first decoded video PTS plus final decoded frame duration; if unavailable, ffprobe video-stream then container duration, warning recorded",
            "video_pts": "frame.pts * frame.time_base; never frame_index / fps",
        },
        "text": {
            "source": "label-100.xlsx label.text; no ASR",
            "normalization": "Unicode NFKC; preserve raw text; record bracket annotations, punctuation and whitespace changes",
            "encoder": "google-bert/bert-base-uncased",
            "revision": TEXT_MODEL_REVISION,
            "tokenizer": "BertTokenizerFast from the pinned model revision",
            "hidden_size": TEXT_DIM,
            "word_pooling": "mean of subword last-layer contextual states",
            "window_weight": "overlap_duration / word_duration; normalize within each nonempty window",
            "missing_word_policy": "MFA forced alignment; internal gaps linear interpolation; edge/no-anchor fallback proportional to word order",
            "pca": {
                "components": PCA_DIM, "fit_positions": "valid K=50 Q1 text windows only",
                "random_seed": args.random_seed, "solver": "randomized",
                "labels_used": False,
            },
        },
        "audio": {
            "source": "first audio stream; FFmpeg PCM S16LE mono 16000 Hz",
            "sample_rate": SAMPLE_RATE, "frame_length_samples": FRAME_LENGTH,
            "frame_length_ms": 25, "frame_hop_samples": FRAME_HOP,
            "frame_hop_ms": 10, "fft_size": FFT_SIZE, "window": "Hann",
            "features_37": AUDIO_FEATURE_NAMES,
            "mfcc": {"n_mfcc": 13, "mel_bands": 40, "mel_norm": "slaney", "dct_type": 2},
            "delta_mfcc": "central first difference at 10-ms frame spacing; one-sided at boundaries",
            "f0": {
                "method": "librosa.pyin", "frame_length_samples": 403, "fmin_hz": 80.0, "fmax_hz": 400,
                "voiced_probability_threshold": VOICED_PROBABILITY_THRESHOLD,
                "statistics": "mean/std only over voiced frames; 0 if none",
                "alignment": "403-sample (25.19-ms) pitch support is interpolated onto the common 400-sample (25-ms) descriptor frame centers",
            },
            "voicing_probability": "pYIN voiced probability in [0,1]",
            "log_energy": "natural log of frame sum of squares plus 1e-12",
            "spectral_flux": "sum of squared positive differences between consecutive L2-normalized magnitude spectra",
            "hnr": "10*log10(r/(1-r)), r is maximum normalized autocorrelation in 70-400 Hz lag range",
            "window_statistics": "mean and standard deviation of 37 frame descriptors",
            "audio_reliable_rms_threshold": AUDIO_RMS_THRESHOLD,
            "audio_present_vs_reliable": "present means one or more analysis frames; reliable means mean RMS >= configured threshold",
        },
        "vision": {
            "feature_tool": "MediaPipe Face Landmarker",
            "feature_dimension": 52,
            "feature_names": BLENDSHAPE_NAMES,
            "semantics": "model blendshape scores; not AU ground truth and not emotion labels",
            "detection_confidence_source": "OpenCV YuNet FaceDetectorYN score, column 14; for frames with a MediaPipe blendshape face",
            "confidence_model_revision": YUNET_MODEL_REVISION,
            "yunet_input": "aspect-preserving letterbox to 320x320",
            "confidence_threshold": FACE_CONFIDENCE_THRESHOLD,
            "face_coverage_threshold": FACE_COVERAGE_THRESHOLD,
            "frame_sampling": "every successfully decoded video frame with a real PTS",
            "window_aggregation": "median over detected frames; mean and nearest-center frame also computed for comparison",
            "window_mask": "face_coverage >= 0.20 and mean YuNet score >= 0.50",
        },
        "storage": {
            "features": "float16 raw aggregated values",
            "time_edges_and_masks": "float32 edges; uint8 masks and counts",
            "normalization_stats": "float32 mean/std/valid_count over valid K=50 positions",
            "text_pca_parameters": "float32 mean and projection matrix",
        },
        "alignment": {
            "tool": "Montreal Forced Aligner",
            "beam": args.beam, "retry_beam": args.retry_beam,
            "g2p_model": "english_us_arpa",
            "reuse_policy": "alignment cache accepted only when sample_id and input/model/config SHA256 key match; full run regenerates MFA alignment",
            "confidence": "TextGrid does not provide word confidence; recorded as null",
        },
        "models": models,
        "software": software,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def fit_text_pca(high_text: np.ndarray, text_mask: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from sklearn.decomposition import PCA

    positions = high_text[text_mask.astype(bool)]
    if positions.shape[0] < PCA_DIM:
        raise ValueError(f"Need at least {PCA_DIM} valid text positions for PCA, got {positions.shape[0]}")
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=seed)
    pca.fit(positions.astype(np.float32))
    transformed = np.zeros((*high_text.shape[:2], PCA_DIM), dtype=np.float32)
    transformed[text_mask.astype(bool)] = pca.transform(positions.astype(np.float32)).astype(np.float32)
    parameters = {
        "text_pca_mean": np.asarray(pca.mean_, dtype=np.float32),
        "text_pca_components": np.asarray(pca.components_, dtype=np.float32),
        "text_pca_explained_variance": np.asarray(pca.explained_variance_, dtype=np.float32),
        "text_pca_explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
        "text_pca_fit_position_count": np.asarray([positions.shape[0]], dtype=np.int64),
    }
    return transformed, parameters


def normalization_parameters(features: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = features.shape[-1]
    means = np.zeros(dims, dtype=np.float32)
    stds = np.zeros(dims, dtype=np.float32)
    counts = np.zeros(dims, dtype=np.int64)
    selected = features[masks.astype(bool)]
    if len(selected):
        means = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        stds = selected.std(axis=0, dtype=np.float64).astype(np.float32)
        counts.fill(len(selected))
    return means, stds, counts


def npz_size_bytes(arrays: dict[str, np.ndarray]) -> int:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.tell()


def compare_variants(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    for k_count in K_COMPARISON:
        text_values = np.stack([row["variants"][k_count]["text_high"] for row in results])
        audio_values = np.stack([row["variants"][k_count]["audio"]["values"] for row in results])
        vision_values = np.stack([row["variants"][k_count]["visual"]["median"] for row in results])
        text_mask = np.stack([row["variants"][k_count]["text_mask"] for row in results]).astype(bool)
        audio = [row["variants"][k_count]["audio"] for row in results]
        visual = [row["variants"][k_count]["visual"] for row in results]
        visual_masks = np.stack([item["median_mask"] for item in visual]).astype(bool)
        arrays = {
            "text_high": text_values.astype(np.float16),
            "audio": audio_values.astype(np.float16),
            "vision": vision_values.astype(np.float16),
            "text_mask": text_mask.astype(np.uint8),
            "audio_mask": np.stack([item["reliable"] for item in audio]).astype(np.uint8),
            "vision_mask": visual_masks.astype(np.uint8),
        }
        total_audio = sum(int(item["counts"].sum()) for item in audio)
        total_video = sum(int(item["frame_count"].sum()) for item in visual)
        by_k[str(k_count)] = {
            "K": k_count,
            "mean_audio_frames_per_window": total_audio / max(len(results) * k_count, 1),
            "mean_video_frames_per_window": total_video / max(len(results) * k_count, 1),
            "empty_text_window_ratio": float(1.0 - text_mask.mean()) if text_mask.size else 1.0,
            "vision_valid_window_ratio": float(visual_masks.mean()) if visual_masks.size else 0.0,
            "compressed_variant_npz_bytes_pre_pca": npz_size_bytes(arrays),
        }

    all_median = np.concatenate([row["variants"][K_FINAL]["visual"]["median"].reshape(-1, 52) for row in results])
    all_mean = np.concatenate([row["variants"][K_FINAL]["visual"]["mean"].reshape(-1, 52) for row in results])
    all_center = np.concatenate([row["variants"][K_FINAL]["visual"]["center"].reshape(-1, 52) for row in results])
    vision_median_mask = np.concatenate([row["variants"][K_FINAL]["visual"]["median_mask"] for row in results]).astype(bool)
    vision_mean_mask = np.concatenate([row["variants"][K_FINAL]["visual"]["mean_mask"] for row in results]).astype(bool)
    vision_center_mask = np.concatenate([row["variants"][K_FINAL]["visual"]["center_mask"] for row in results]).astype(bool)
    detected_values = np.concatenate([
        row["variants"][K_FINAL]["visual"]["median"][row["variants"][K_FINAL]["visual"]["detected_count"] > 0]
        for row in results
    ], axis=0) if any(np.any(row["variants"][K_FINAL]["visual"]["detected_count"] > 0) for row in results) else np.zeros((0, 52), dtype=np.float32)
    vision_comparison = {
        "aggregation_mask_rates": {
            "center_frame": float(vision_center_mask.mean()),
            "multi_frame_mean": float(vision_mean_mask.mean()),
            "multi_frame_median": float(vision_median_mask.mean()),
        },
        "zero_vision_sample_counts": {
            "center_frame": int(sum(not row["variants"][K_FINAL]["visual"]["center_mask"].any() for row in results)),
            "multi_frame_mean": int(sum(not row["variants"][K_FINAL]["visual"]["mean_mask"].any() for row in results)),
            "multi_frame_median": int(sum(not row["variants"][K_FINAL]["visual"]["median_mask"].any() for row in results)),
        },
        "mean_abs_mean_vs_median_feature_difference": float(np.mean(np.abs(all_mean - all_median))),
        "mean_abs_center_vs_median_feature_difference": float(np.mean(np.abs(all_center - all_median))),
        "blendshape_out_of_range_rate_on_median_detected_bins": (
            float(np.mean((detected_values < -1e-6) | (detected_values > 1.000001)))
            if len(detected_values) else 0.0
        ),
        "anomaly_definition": "non-finite or outside [0,1] among detected-frame median blendshape scores",
    }

    alignment_totals = {name: {"timed_words": 0, "total_words": 0, "order_violations": 0} for name in ("uniform_word_order", "mfa_only", "mfa_plus_fallback")}
    for result in results:
        rows = result["alignment_rows"]
        n = len(rows)
        duration = float(result["manifest"]["duration_used_sec"])
        for name in alignment_totals:
            alignment_totals[name]["total_words"] += n
        uniform = [(duration * i / n, duration * (i + 1) / n) for i in range(n)] if n else []
        mfa = [(row["start_sec"], row["end_sec"]) for row in rows if row["timestamp_source"] == "forced_alignment"]
        fallback = [(row["start_sec"], row["end_sec"]) for row in rows if row["start_sec"] is not None and row["end_sec"] is not None]
        for name, intervals in (("uniform_word_order", uniform), ("mfa_only", mfa), ("mfa_plus_fallback", fallback)):
            alignment_totals[name]["timed_words"] += len(intervals)
            alignment_totals[name]["order_violations"] += sum(
                intervals[i][0] > intervals[i + 1][0] + 1e-7 for i in range(max(0, len(intervals) - 1))
            )
    for item in alignment_totals.values():
        item["word_boundary_coverage"] = item["timed_words"] / item["total_words"] if item["total_words"] else 0.0
        item["manual_accuracy_available"] = False
        item["manual_accuracy_note"] = "No ground-truth word timestamps were supplied; coverage is not accuracy."
    return {
        "time_granularity": by_k,
        "visual_aggregation": vision_comparison,
        "text_time_alignment": alignment_totals,
    }


def save_final_arrays(results: list[dict[str, Any]], output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([row["manifest"]["sample_id"] for row in results], dtype=np.str_)
    durations = np.asarray([row["manifest"]["duration_used_sec"] for row in results], dtype=np.float32)
    time_edges = np.stack([row["variants"][K_FINAL]["edges"] for row in results]).astype(np.float32)
    text_high = np.stack([row["variants"][K_FINAL]["text_high"] for row in results]).astype(np.float32)
    text_mask = np.stack([row["variants"][K_FINAL]["text_mask"] for row in results]).astype(np.uint8)
    audio32 = np.stack([row["variants"][K_FINAL]["audio"]["values"] for row in results]).astype(np.float32)
    audio_mask = np.stack([row["variants"][K_FINAL]["audio"]["reliable"] for row in results]).astype(np.uint8)
    vision32 = np.stack([row["variants"][K_FINAL]["visual"]["median"] for row in results]).astype(np.float32)
    vision_mask = np.stack([row["variants"][K_FINAL]["visual"]["median_mask"] for row in results]).astype(np.uint8)
    text32, pca_params = fit_text_pca(text_high, text_mask, seed)
    text16 = text32.astype(np.float16)
    audio16 = audio32.astype(np.float16)
    vision16 = vision32.astype(np.float16)
    quality_keys = ["text_coverage", "audio_frame_count", "audio_present", "voiced_ratio", "video_frame_count", "face_detected_count", "face_coverage", "mean_face_confidence"]
    quality: dict[str, np.ndarray] = {}
    for key in quality_keys:
        quality[key] = np.asarray([
            [window[key] for window in row["window_rows"]] for row in results
        ], dtype=np.float32 if key in {"text_coverage", "voiced_ratio", "face_coverage", "mean_face_confidence"} else np.uint32)
    valid_mask = np.maximum(np.maximum(text_mask, audio_mask), vision_mask).astype(np.uint8)
    feature_arrays: dict[str, np.ndarray] = {
        "ids": ids, "text": text16, "audio": audio16, "vision": vision16,
        "valid_mask": valid_mask, "time_edges": time_edges, "durations": durations,
        "text_mask": text_mask, "audio_mask": audio_mask, "vision_mask": vision_mask,
        "text_feature_names": np.asarray([f"text_pca_{i:03d}" for i in range(PCA_DIM)], dtype=np.str_),
        "audio_feature_names": np.asarray([f"{name}_{stat}" for stat in ("mean", "std") for name in AUDIO_FEATURE_NAMES], dtype=np.str_),
        "vision_feature_names": np.asarray(BLENDSHAPE_NAMES, dtype=np.str_),
        **quality,
        **pca_params,
    }
    np.savez_compressed(output_dir / "q1_features.npz", **feature_arrays)
    stats: dict[str, np.ndarray] = {}
    for modality, features, mask in (("text", text32, text_mask), ("audio", audio32, audio_mask), ("vision", vision32, vision_mask)):
        mean, std, count = normalization_parameters(features, mask)
        stats[f"{modality}_mean"] = mean.astype(np.float32)
        stats[f"{modality}_std"] = std.astype(np.float32)
        stats[f"{modality}_valid_count"] = count.astype(np.float32)
    np.savez_compressed(output_dir / "normalization_stats.npz", **stats)
    errors = {}
    for name, source, reduced in (("text", text32, text16), ("audio", audio32, audio16), ("vision", vision32, vision16)):
        delta = np.abs(source.astype(np.float64) - reduced.astype(np.float64))
        errors[name] = {"max_absolute_error": float(delta.max(initial=0.0)), "mean_absolute_error": float(delta.mean())}
    return {
        "shapes": {"text": list(text16.shape), "audio": list(audio16.shape), "vision": list(vision16.shape), "time_edges": list(time_edges.shape)},
        "dtypes": {"text": str(text16.dtype), "audio": str(audio16.dtype), "vision": str(vision16.dtype), "time_edges": str(time_edges.dtype)},
        "float16_conversion_error": errors,
        "compressed_npz_bytes": int((output_dir / "q1_features.npz").stat().st_size),
        "pca_fit_positions": int(pca_params["text_pca_fit_position_count"][0]),
    }


def extract_all(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    records = read_jsonl(args.work_dir / "records.jsonl")
    alignment_cache_path = args.work_dir / "alignment_cache.json"
    alignment_cache = json.loads(alignment_cache_path.read_text(encoding="utf-8")) if alignment_cache_path.is_file() else {}
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.text_model), use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(args.text_model), local_files_only=True).to(device).eval()
    mediapipe, landmarker = create_face_landmarker(args.face_model)
    yunet = create_yunet_detector(args.yunet_model)
    results: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(records, start=1):
            if record.get("prepare_status") != "ok":
                logging.warning("Continuing with degraded retained sample %s: %s", record["sample_id"], record.get("error"))
            item = process_sample(record, args, alignment_cache, model, tokenizer, torch, device, landmarker, mediapipe, yunet)
            results.append(item)
            logging.info("Extracted %d/%d %s shape=(50,768)/(50,74)/(50,52) video_frames=%d vision_valid=%d status=%s", index, len(records), record["sample_id"], item["manifest"]["decoded_frame_count"], item["manifest"]["vision_valid_bins"], item["processing_log"]["status"])
    finally:
        landmarker.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mfa_status_path = args.work_dir / "mfa_align_status.json"
    mfa_status = json.loads(mfa_status_path.read_text(encoding="utf-8")) if mfa_status_path.is_file() else {"status": "not_run"}
    save_json(args.output_dir / "mfa_align_status.json", mfa_status)
    arrays_info: dict[str, Any] | None = None
    if not args.sample_id:
        arrays_info = save_final_arrays(results, args.output_dir, args.random_seed)
    else:
        smoke_item = results[0]
        smoke_variant = smoke_item["variants"][K_FINAL]
        smoke_checks = {
            "text_shape_is_50_by_768_before_pca": smoke_variant["text_high"].shape == (50, TEXT_DIM),
            "audio_shape_is_50_by_74": smoke_variant["audio"]["values"].shape == (50, 74),
            "vision_shape_is_50_by_52": smoke_variant["visual"]["median"].shape == (50, 52),
            "exactly_51_edges_start_at_zero_and_end_at_duration": (
                len(smoke_variant["edges"]) == 51
                and smoke_variant["edges"][0] == 0
                and abs(float(smoke_variant["edges"][-1]) - smoke_item["manifest"]["duration_used_sec"]) < 1e-8
            ),
            "time_edges_strictly_increasing": bool(np.all(np.diff(smoke_variant["edges"]) > 0)),
            "audio_mask_never_set_when_no_audio_frame": bool(np.all(smoke_variant["audio"]["reliable"] <= smoke_variant["audio"]["present"])),
            "empty_text_windows_have_zero_features": bool(np.all(smoke_variant["text_high"][~smoke_variant["text_mask"]] == 0)),
            "vision_mask_requires_coverage_and_confidence": bool(np.all(
                smoke_variant["visual"]["median_mask"]
                <= ((smoke_variant["visual"]["coverage"] >= FACE_COVERAGE_THRESHOLD)
                    & (smoke_variant["visual"]["mean_confidence"] >= FACE_CONFIDENCE_THRESHOLD))
            )),
            "real_video_pts_recorded_when_frames_decoded": (
                smoke_item["manifest"]["decoded_frame_count"] == 0
                or (smoke_item["manifest"]["first_video_pts_sec"] is not None and smoke_item["manifest"]["last_video_pts_sec"] is not None)
            ),
        }
        assert all(smoke_checks.values()), f"Single-sample Q1 checks failed: {smoke_checks}"
        save_json(args.output_dir / "single_sample_smoke.json", {
            "sample_id": smoke_item["manifest"]["sample_id"],
            "raw_text_shape": list(smoke_item["variants"][K_FINAL]["text_high"].shape),
            "audio_shape": list(smoke_item["variants"][K_FINAL]["audio"]["values"].shape),
            "vision_shape": list(smoke_item["variants"][K_FINAL]["visual"]["median"].shape),
            "time_edges": smoke_item["variants"][K_FINAL]["edges"].tolist(),
            "manifest": smoke_item["manifest"],
            "processing_log": smoke_item["processing_log"],
            "mfa_alignment": mfa_status,
            "quality_windows": smoke_item["window_rows"],
            "assertions": smoke_checks,
        })
    manifest_rows = [row["manifest"] for row in results]
    alignment_rows = [entry for row in results for entry in row["alignment_rows"]]
    quality_rows = [entry for row in results for entry in row["window_rows"]]
    log_rows = [row["processing_log"] for row in results]
    write_csv(args.output_dir / "q1_manifest.csv", manifest_rows)
    save_jsonl(args.output_dir / "q1_alignment.jsonl", alignment_rows)
    write_csv(args.output_dir / "q1_quality_report.csv", quality_rows)
    save_jsonl(args.output_dir / "processing_log.jsonl", log_rows)
    if args.sample_id:
        return {"sample_count": 1, "smoke_only": True, "arrays": arrays_info}

    software = software_metadata()
    models = model_metadata(args)
    configuration = feature_configuration(args, models, software)
    try:
        import yaml
        (args.output_dir / "feature_config.yaml").write_text(yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except ImportError:
        save_json(args.output_dir / "feature_config.yaml", configuration)
    save_json(args.output_dir / "software_versions.json", software)
    comparison = compare_variants(results)
    run_info = {
        "sample_count": len(results),
        "sample_ids": [row["manifest"]["sample_id"] for row in results],
        "run_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "random_seed": args.random_seed,
        "device": str(device),
        "model_revisions": {"text": TEXT_MODEL_REVISION, "yunet": YUNET_MODEL_REVISION},
        "mfa_alignment": mfa_status,
        "arrays": arrays_info,
        "comparison": comparison,
        "warning_summary": {
            code: sum(code in row["manifest"]["warning_codes"].split("|") for row in results)
            for code in sorted({code for row in results for code in row["manifest"]["warning_codes"].split("|") if code})
        },
        "unresolved_decode_samples": [row["manifest"]["sample_id"] for row in results if not row["manifest"]["video_decode_complete"] or (row["manifest"]["duration_audio_stream_sec"] > 0 and not row["manifest"]["audio_decode_complete"])],
        "low_vision_coverage_samples": [row["manifest"]["sample_id"] for row in results if row["manifest"]["vision_valid_bins"] == 0 or row["manifest"]["face_coverage_global"] < FACE_COVERAGE_THRESHOLD],
        "near_silence_samples": [row["manifest"]["sample_id"] for row in results if row["manifest"]["audio_reliable_bins"] == 0],
        "duration_differences_sec": {
            "container_minus_decoded_video": [row["manifest"]["duration_container_minus_used_sec"] for row in results],
            "video_stream_minus_decoded_video": [row["manifest"]["duration_video_stream_minus_used_sec"] for row in results],
            "audio_decoded_minus_video_decoded": [row["manifest"]["duration_audio_video_decoded_difference_sec"] for row in results],
        },
    }
    save_json(args.output_dir / "q1_run_info.json", run_info)
    validator = Path(__file__).with_name("validate_q1.py")
    checked = run_command([
        sys.executable, str(validator), "--data-dir", str(args.data_dir),
        "--result-dir", str(args.output_dir),
    ], timeout=600)
    if checked.stdout:
        logging.info("Automatic Q1 validation:\n%s", checked.stdout.strip())
    if checked.returncode:
        raise RuntimeError("Automatic Q1 validation failed:\n" + checked.stdout + "\n" + checked.stderr)
    return run_info


def main() -> None:
    environment_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    args = parse_args()
    setup_logging(args.work_dir)
    if args.stage in {"all", "prepare"}:
        import datetime
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        save_json(args.work_dir / "mfa_run_id.json", {"run_id": run_id})
        prepare_corpus(args)
    if args.stage in {"all", "align"}:
        align_status = align_corpus(args)
        logging.info("MFA alignment stage: %s", align_status)
    if args.stage in {"all", "extract"}:
        info = extract_all(args)
        logging.info("Q1 extraction completed: %s", json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
