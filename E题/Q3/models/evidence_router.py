import torch
import torch.nn as nn


def weighted_pool(h, weight):
    weight = weight.to(h.dtype)
    return (h * weight.unsqueeze(-1)).sum(dim=1) / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)


def masked_mean(h, valid_mask):
    return weighted_pool(h, valid_mask)


class EvidenceRouter(nn.Module):
    def __init__(self, d_model=128, dropout=0.15):
        super().__init__()
        self.route = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3))
        self.fuse = nn.Sequential(nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, reps, route=True):
        joined = torch.cat(reps, dim=-1)
        if route:
            alpha = self.route(joined).softmax(dim=-1)
        else:
            alpha = joined.new_full((joined.shape[0], 3), 1 / 3)
        fused = torch.cat([alpha[:, i:i+1] * r for i, r in enumerate(reps)], dim=-1)
        return self.fuse(fused), alpha
