import torch
from torch import nn


class SpanGeometryEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, d_model))

    def forward(self, missing_mask, valid_mask):
        b, t = missing_mask.shape
        features = torch.zeros(b, t, 5, device=missing_mask.device)
        for i in range(b):
            length = int(valid_mask[i].sum())
            starts = torch.nonzero(missing_mask[i] & ~torch.cat((torch.zeros(1, dtype=torch.bool, device=missing_mask.device), missing_mask[i, :-1]))).flatten()
            for start_tensor in starts:
                start = int(start_tensor)
                end = start + 1
                while end < t and bool(missing_mask[i, end]):
                    end += 1
                span = end - start
                pos = torch.arange(start, end, device=missing_mask.device)
                features[i, start:end, 0] = 1
                features[i, start:end, 1] = span / max(length, 1)
                features[i, start:end, 2] = (pos - start + 1) / max(length, 1)
                features[i, start:end, 3] = (end - pos) / max(length, 1)
                features[i, start:end, 4] = pos / max(length - 1, 1)
        encoded = self.net(features)
        return encoded * missing_mask[..., None]
