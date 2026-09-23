import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityFusion(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) for _ in range(3)])
        self.beta = nn.Parameter(torch.ones(3))
        self.fuse = nn.Sequential(nn.Linear(3*d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, h_t, h_a, h_v, g_t, g_a, g_v, miss_t, miss_a, miss_v, valid_mask):
        h = (h_t, h_a, h_v)
        g = (g_t, g_a, g_v)
        masks = (miss_t, miss_a, miss_v)
        logits = torch.cat([self.gates[i](torch.cat((h[i], g[i]), -1)) - F.softplus(self.beta[i])*masks[i][..., None].float() for i in range(3)], -1)
        reliability = logits.softmax(-1)
        fused = self.fuse(torch.cat([reliability[..., i:i+1] * h[i] for i in range(3)], -1))
        return fused * valid_mask[..., None], reliability
