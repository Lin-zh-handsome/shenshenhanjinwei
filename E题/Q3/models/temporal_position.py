import torch
import torch.nn as nn


class TemporalPositionEmbedding(nn.Module):
    def __init__(self, max_len=50, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(positions).unsqueeze(0)
