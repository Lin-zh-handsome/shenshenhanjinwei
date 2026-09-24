import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.cls = nn.Linear(d_model, 3)
        self.reg = nn.Linear(d_model, 1)

    def forward(self, z):
        logits = self.cls(z)
        return logits, logits.softmax(dim=-1), 3.0 * torch.tanh(self.reg(z).squeeze(-1))
