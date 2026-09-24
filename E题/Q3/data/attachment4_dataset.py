import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.valid_mask import infer_aligned_valid_mask


class Attachment4AlignedDataset(Dataset):
    def __init__(self, feature_dir, video_dir):
        self.feature_dir = Path(feature_dir)
        self.video_dir = Path(video_dir)
        self.files = sorted(self.feature_dir.glob("*.pkl"), key=lambda p: [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.stem)])
        if not self.files:
            raise FileNotFoundError(self.feature_dir)
        for p in self.files:
            if not (self.video_dir / f"{p.stem}.mp4").is_file():
                raise FileNotFoundError(self.video_dir / f"{p.stem}.mp4")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with p.open("rb") as f:
            d = pickle.load(f, encoding="latin1")
        x = {k: np.asarray(d[k], dtype=np.float32) for k in ("text", "audio", "vision")}
        for k, width in (("text", 768), ("audio", 74), ("vision", 35)):
            if x[k].shape != (50, width) or not np.isfinite(x[k]).all():
                raise ValueError(f"{p}: {k} shape or NaN/Inf")
        b = np.asarray(d["text_bert"], dtype=np.int64)
        if b.shape != (3, 50):
            raise ValueError(f"{p}: text_bert shape {b.shape}")
        return {
            "sample_id": str(d.get("id") or p.stem), "raw_text": str(d.get("raw_text", "")),
            **{k: torch.from_numpy(v) for k, v in x.items()},
            "text_bert": torch.from_numpy(b),
            "valid_mask": torch.from_numpy(infer_aligned_valid_mask(b, x["text"], x["audio"], x["vision"])),
            "video_path": str(self.video_dir / f"{p.stem}.mp4"),
        }
