import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from data.valid_mask import infer_aligned_valid_mask


class AlignedMoseiDataset(Dataset):
    def __init__(self, pkl_path: str, split: str):
        if split not in ("train", "valid", "test"):
            raise ValueError(split)
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f, encoding="latin1")[split]
        self.data = dict(self.data)
        for name, width in (("text", 768), ("audio", 74), ("vision", 35)):
            x = np.asarray(self.data[name], dtype=np.float32)
            if x.ndim != 3 or x.shape[1:] != (50, width):
                raise ValueError(f"{split}.{name} shape {x.shape}")
            bad = np.argwhere(~np.isfinite(x))
            if len(bad):
                raise ValueError(f"{split}.{name} NaN/Inf at {bad[0].tolist()}")
            self.data[name] = x
        b = np.asarray(self.data["text_bert"], dtype=np.int64)
        if b.shape != (len(self), 3, 50):
            raise ValueError(f"{split}.text_bert shape {b.shape}")
        self.data["text_bert"] = b
        cls = np.asarray(self.data["classification_labels"])
        reg = np.asarray(self.data["regression_labels"])
        if not np.isfinite(cls).all() or not np.isin(cls, (0, 1, 2)).all():
            raise ValueError(f"{split} classification label range")
        if not np.isfinite(reg).all() or (np.abs(reg) > 3).any():
            raise ValueError(f"{split} regression label range")
        self.cls = cls.astype(np.int64)
        self.reg = reg.astype(np.float32)
        self.masks = np.stack([infer_aligned_valid_mask(b[i], *(self.data[k][i] for k in ("text", "audio", "vision"))) for i in range(len(self))])

    def __len__(self):
        return len(self.data["text"])

    def __getitem__(self, idx):
        d = self.data
        return {
            "id": str(d["id"][idx]), "raw_text": str(d["raw_text"][idx]),
            "text": torch.from_numpy(d["text"][idx]),
            "audio": torch.from_numpy(d["audio"][idx]),
            "vision": torch.from_numpy(d["vision"][idx]),
            "text_bert": torch.from_numpy(d["text_bert"][idx]),
            "valid_mask": torch.from_numpy(self.masks[idx]),
            "y_cls": torch.tensor(self.cls[idx], dtype=torch.long),
            "y_reg": torch.tensor(self.reg[idx], dtype=torch.float32),
        }
