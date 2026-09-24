import math
import re
from pathlib import Path

import cv2


TIME_METHOD = "uniform_valid_grid_to_video_duration"
TEXT_METHOD = "proportional_raw_text_partition"
MAPPING_NOTE = "aligned_grid_exact; second/text projection approximate"


def regex_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video cannot be opened: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded_count += 1
    cap.release()
    if not math.isfinite(fps) or fps <= 0 or decoded_count <= 0:
        raise ValueError(f"Invalid video FPS or no decodable frames: {video_path}: {fps}/{decoded_count}")
    warning = "" if reported_count == decoded_count else f"reported_frames={reported_count};decoded_frames={decoded_count}"
    return {"fps": fps, "frame_count": decoded_count, "reported_frame_count": reported_count,
            "duration": decoded_count / fps, "warning": warning}


def project_span(span, valid_len, raw_text, duration):
    s, e = span["grid_start"], span["grid_end_exclusive"]
    start = max(0.0, min(duration, duration * s / valid_len))
    end = max(0.0, min(duration, duration * e / valid_len))
    words = regex_tokenize(raw_text)
    a = min(len(words), math.floor(s / valid_len * len(words)))
    b = min(len(words), math.ceil(e / valid_len * len(words)))
    return {"start_sec_approx": start, "end_sec_approx": end,
            "text_fragment_approx": " ".join(words[a:b]) if words else "",
            "text_word_start_approx": a if words else "",
            "text_word_end_approx": b if words else "",
            "mapping_method": TIME_METHOD, "text_mapping_method": TEXT_METHOD}


def save_keyframe(video_path, span, metadata, output_path):
    midpoint = 0.5 * (span["start_sec_approx"] + span["end_sec_approx"])
    frame_index = min(metadata["frame_count"] - 1, max(0, round(midpoint * metadata["fps"])))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    fallback = False
    if not ok:
        fallback = True
        cap = cv2.VideoCapture(str(video_path))
        frame = None
        for _ in range(frame_index + 1):
            ok, current = cap.read()
            if not ok:
                break
            frame = current
        cap.release()
        if frame is None:
            raise ValueError(f"Cannot decode keyframe {video_path} frame={frame_index}")
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, round(height * 640 / width)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise ValueError(f"Cannot save keyframe: {output_path}")
    return str(output_path), fallback
