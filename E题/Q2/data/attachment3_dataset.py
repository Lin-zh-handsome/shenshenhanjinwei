from pathlib import Path
import pickle
import re
import numpy as np
import torch
from torch.utils.data import Dataset


def natural_key(path):
    return [int(v) if v.isdigit() else v for v in re.split(r'(\d+)', path.stem)]


class Attachment3AlignedDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(Path(directory).glob('*.pkl'), key=natural_key)
        if not self.files:
            raise FileNotFoundError(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, 'rb') as f:
            x = pickle.load(f)
        if 'test' in x and isinstance(x['test'], dict):
            x = x['test']
        out = {}
        for key, shape in [('text_bert', (3, 50)), ('audio', (50, 74)), ('vision', (50, 35))]:
            a = np.asarray(x[key])
            if a.shape == (1, *shape):
                a = a[0]
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'{path}:{key} shape={a.shape}')
            if key == 'text_bert':
                if np.max(np.abs(a - np.rint(a))) > 1e-3:
                    raise ValueError(f'{path}: fractional token values')
                a = np.rint(a).astype(np.int64)
            else:
                a = a.astype(np.float32)
            out[key] = torch.from_numpy(a.copy())
        out['sample_id'] = path.stem
        return out
