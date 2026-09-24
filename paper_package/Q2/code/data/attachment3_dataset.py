from pathlib import Path
import pickle
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from data.aligned_dataset import integer_text_bert
from data.mask_utils import base_masks, infer_missing_candidates, sequence_valid_from_signals


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
                a = integer_text_bert(a)
            else:
                a = a.astype(np.float32)
            out[key] = torch.from_numpy(a.copy())
        out['sample_id'] = path.stem
        text_bert = out['text_bert'].unsqueeze(0)
        audio = out['audio'].unsqueeze(0)
        vision = out['vision'].unsqueeze(0)
        valid = sequence_valid_from_signals(text_bert, audio, vision)
        candidates = infer_missing_candidates(text_bert, audio, vision, valid)
        observed = {key: valid & ~candidates[modality] for modality, key in
                    (('T', 'text'), ('A', 'audio'), ('V', 'vision'))}
        masks = base_masks(text_bert, audio, vision, observed=observed)
        out.update({key: value.squeeze(0) for key, value in masks.items()})
        return out
