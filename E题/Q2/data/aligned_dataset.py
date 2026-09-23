import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def load_aligned(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for split in ('train', 'valid', 'test'):
        x = data[split]
        n = len(x['id'])
        for key, tail in {'text_bert': (3, 50), 'text': (50, 768), 'audio': (50, 74), 'vision': (50, 35)}.items():
            arr = np.asarray(x[key])
            if arr.shape != (n, *tail) or not np.isfinite(arr).all():
                raise ValueError(f'{split}.{key}: invalid shape or nonfinite values: {arr.shape}')
        for key in ('classification_labels', 'regression_labels'):
            if np.asarray(x[key]).size != n or not np.isfinite(x[key]).all():
                raise ValueError(f'{split}.{key}: invalid labels')
        cls = np.asarray(x['classification_labels']).reshape(-1)
        reg = np.asarray(x['regression_labels']).reshape(-1)
        if not np.isin(cls, [0, 1, 2]).all() or (np.abs(reg) > 3.0001).any():
            raise ValueError(f'{split}: label range')
    return data


class AlignedMoseiDataset(Dataset):
    def __init__(self, pkl_path=None, split='train', source=None):
        self.split = split
        x = source[split] if source is not None else load_aligned(pkl_path)[split]
        self.ids = [str(v) for v in x['id']]
        self.text_bert = np.asarray(x['text_bert'], dtype=np.int64)
        self.text_teacher = np.asarray(x['text'], dtype=np.float32)
        self.audio = np.asarray(x['audio'], dtype=np.float32)
        self.vision = np.asarray(x['vision'], dtype=np.float32)
        self.y_cls = np.asarray(x['classification_labels'], dtype=np.int64).reshape(-1)
        self.y_reg = np.asarray(x['regression_labels'], dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {'id': self.ids[idx], 'text_bert': torch.from_numpy(self.text_bert[idx]),
                'text_teacher': torch.from_numpy(self.text_teacher[idx]),
                'audio': torch.from_numpy(self.audio[idx]), 'vision': torch.from_numpy(self.vision[idx]),
                'y_cls': torch.tensor(self.y_cls[idx]), 'y_reg': torch.tensor(self.y_reg[idx])}
