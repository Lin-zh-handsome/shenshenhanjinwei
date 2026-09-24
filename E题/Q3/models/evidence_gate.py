import torch
import torch.nn as nn


class ContextEvidenceGate(nn.Module):
    def __init__(self, d_model=128, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, h_m, context, valid_mask, temperature=1.0):
        logits = self.net(torch.cat((h_m, context), dim=-1)).squeeze(-1)
        return torch.sigmoid(logits / temperature) * valid_mask.to(logits.dtype)
