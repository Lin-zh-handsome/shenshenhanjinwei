#!/usr/bin/env python3
"""Automatic acceptance checks for the Q1 frozen-spec outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_rows = pd.read_excel(args.data_dir / "label-100.xlsx", sheet_name="label", usecols=["video_id", "clip_id"])
    expected_ids = {
        f"{str(row.video_id).strip()}/{int(row.clip_id) if float(row.clip_id).is_integer() else str(row.clip_id).strip()}"
        for row in expected_rows.itertuples(index=False)
    }
    manifest = pd.read_csv(args.result_dir / "q1_manifest.csv", dtype={"sample_id": str})
    arrays = np.load(args.result_dir / "q1_features.npz", allow_pickle=False)
    ids = arrays["ids"].astype(str).tolist()
    manifest_ids = manifest["sample_id"].astype(str).tolist()

    assert len(expected_ids) == 100, f"Expected 100 Excel IDs, got {len(expected_ids)}"
    assert len(manifest) == 100, f"Expected 100 manifest rows, got {len(manifest)}"
    assert manifest["sample_id"].nunique() == 100, "Manifest sample IDs are not unique"
    assert len(ids) == 100, f"Expected 100 NPZ samples, got {len(ids)}"
    assert len(set(ids)) == 100, "NPZ sample IDs are not unique"
    assert set(ids) == expected_ids, "NPZ sample IDs differ from Excel"
    assert set(manifest_ids) == expected_ids, "Manifest sample IDs differ from Excel"
    assert set(ids) == set(manifest_ids), "Manifest and NPZ IDs differ"
    missing_video_files = [
        sample_id for sample_id, relative in zip(manifest_ids, manifest["relative_video_path"].astype(str))
        if not (args.data_dir / relative).is_file()
    ]
    assert not missing_video_files, f"Source videos missing: {missing_video_files}"

    assert arrays["text"].shape == (100, 50, 128), f"Unexpected text shape {arrays['text'].shape}"
    assert arrays["audio"].shape == (100, 50, 74), f"Unexpected audio shape {arrays['audio'].shape}"
    assert arrays["vision"].shape == (100, 50, 52), f"Unexpected vision shape {arrays['vision'].shape}"
    assert arrays["time_edges"].shape == (100, 51), f"Unexpected time_edges shape {arrays['time_edges'].shape}"
    assert arrays["durations"].shape == (100,), f"Unexpected durations shape {arrays['durations'].shape}"
    assert np.all(arrays["time_edges"][:, 0] == 0), "time_edges do not start at zero"
    assert np.all(np.diff(arrays["time_edges"], axis=1) > 0), "time_edges are not strictly increasing"
    assert np.allclose(arrays["time_edges"][:, -1], arrays["durations"], rtol=1e-6, atol=1e-5), "Last time edge differs from duration"
    assert arrays["text"].shape[1] == arrays["audio"].shape[1] == arrays["vision"].shape[1] == 50
    for name in ("text", "audio", "vision"):
        assert np.isfinite(arrays[name]).all(), f"{name} contains NaN or Inf"
    assert np.isfinite(arrays["time_edges"]).all() and np.isfinite(arrays["durations"]).all()
    assert arrays["text"].dtype == np.float16 and arrays["audio"].dtype == np.float16 and arrays["vision"].dtype == np.float16

    text_mask = arrays["text_mask"].astype(bool)
    audio_mask = arrays["audio_mask"].astype(bool)
    vision_mask = arrays["vision_mask"].astype(bool)
    for name, raw_mask, mask in (("text", arrays["text_mask"], text_mask), ("audio", arrays["audio_mask"], audio_mask), ("vision", arrays["vision_mask"], vision_mask)):
        assert mask.shape == (100, 50), f"{name} mask has wrong shape"
        assert np.isin(raw_mask, [0, 1]).all(), f"{name} mask is not binary"
    assert np.array_equal(arrays["valid_mask"], np.maximum(np.maximum(text_mask, audio_mask), vision_mask).astype(np.uint8))
    assert np.all(arrays["audio_mask"] <= arrays["audio_present"]), "Reliable audio mask exceeds audio-present mask"
    assert np.all(arrays["vision_mask"] <= (arrays["face_detected_count"] > 0)), "Vision mask set without detected face frames"
    assert np.all(arrays["text"][~text_mask] == 0), "Empty text windows contain nonzero projected feature values"

    quality = pd.read_csv(args.result_dir / "q1_quality_report.csv")
    assert len(quality) == 100 * 50, f"Expected 5000 quality windows, got {len(quality)}"
    assert quality.groupby("sample_id")["k"].nunique().eq(50).all(), "At least one sample lacks 50 windows"
    for sample_id, group in quality.groupby("sample_id", sort=False):
        group = group.sort_values("k")
        edges = arrays["time_edges"][ids.index(str(sample_id))]
        assert np.allclose(group["start_sec"].to_numpy(), edges[:-1], atol=1e-5)
        assert np.allclose(group["end_sec"].to_numpy(), edges[1:], atol=1e-5)
        assert (group["end_sec"].to_numpy() > group["start_sec"].to_numpy()).all()

    alignment = [json.loads(line) for line in (args.result_dir / "q1_alignment.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_index = manifest.set_index("sample_id")
    allowed_sources = {"forced_alignment", "interpolated", "proportional_fallback"}
    order_errors = []
    bounds_errors = []
    for sample_id in ids:
        rows = [item for item in alignment if item["sample_id"] == sample_id]
        previous_start = -float("inf")
        duration = float(manifest_index.loc[sample_id, "duration_used_sec"])
        for item in rows:
            start, end = item.get("start_sec"), item.get("end_sec")
            if start is not None and end is not None:
                if start < -1e-6 or end > duration + 1e-5 or end < start:
                    bounds_errors.append((sample_id, item["word"], start, end, duration))
                if start < previous_start - 1e-5:
                    order_errors.append((sample_id, item["word"], start, previous_start))
                previous_start = start
            if item.get("timestamp_source") is not None:
                assert item["timestamp_source"] in allowed_sources
            assert item.get("char_start") is not None and item.get("char_end") is not None
    assert not bounds_errors, f"Word times out of bounds: {bounds_errors[:5]}"
    assert not order_errors, f"Word timestamps reverse order: {order_errors[:5]}"

    incomplete_video = manifest.loc[~manifest["video_decode_complete"].astype(bool), "sample_id"].astype(str).tolist()
    incomplete_audio = manifest.loc[(manifest["duration_audio_stream_sec"] > 0) & ~manifest["audio_decode_complete"].astype(bool), "sample_id"].astype(str).tolist()
    assert not incomplete_video, f"Video decode did not reach EOF for samples: {incomplete_video}"
    assert not incomplete_audio, f"Audio decode did not reach EOF for samples: {incomplete_audio}"
    result = {
        "video_count": 100,
        "manifest_row_count": len(manifest),
        "unique_sample_id_count": len(set(ids)),
        "npz_sample_count": len(ids),
        "shapes": {name: list(arrays[name].shape) for name in ("text", "audio", "vision", "time_edges")},
        "nonfinite_feature_values": {name: int((~np.isfinite(arrays[name])).sum()) for name in ("text", "audio", "vision")},
        "alignment_word_rows": len(alignment),
        "alignment_order_errors": len(order_errors),
        "alignment_bounds_errors": len(bounds_errors),
        "incomplete_video_decode_samples": incomplete_video,
        "incomplete_audio_decode_samples": incomplete_audio,
        "low_vision_coverage_samples": manifest.loc[(manifest["vision_valid_bins"] == 0) | (manifest["face_coverage_global"] < 0.20), "sample_id"].astype(str).tolist(),
        "near_silence_samples": manifest.loc[manifest["audio_reliable_bins"] == 0, "sample_id"].astype(str).tolist(),
        "padding_used_on_time_axis": False,
        "status": "PASS" if not incomplete_video and not incomplete_audio else "PASS_WITH_RECORDED_DECODE_WARNINGS",
    }
    (args.result_dir / "validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
