import argparse
from pathlib import Path

from utils.io import load_config, save_json


def inventory(aligned_dir, unaligned_dir):
    aligned_dir = Path(aligned_dir)
    unaligned_dir = Path(unaligned_dir)
    aligned = {p.stem for p in aligned_dir.glob("*.pkl")}
    unaligned = {p.stem for p in unaligned_dir.glob("*.pkl")}
    aligned_videos = {p.stem for p in (aligned_dir / "videos").glob("*.mp4")}
    unaligned_videos = {p.stem for p in (unaligned_dir / "videos").glob("*.mp4")}
    return {
        "aligned_feature_count": len(aligned),
        "unaligned_feature_count": len(unaligned),
        "video_count": len(aligned_videos) + len(unaligned_videos),
        "paired_aligned_count": len(aligned & aligned_videos),
        "paired_unaligned_count": len(unaligned & unaligned_videos),
        "unique_video_id_count": len(aligned_videos | unaligned_videos),
        "aligned_video_count": len(aligned_videos),
        "unaligned_video_count": len(unaligned_videos),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    aligned_dir = Path(cfg["paths"]["attachment4_aligned_dir"])
    result = inventory(aligned_dir, aligned_dir.parent / "未对齐版本")
    save_json(args.output, result)
    print(result, flush=True)
