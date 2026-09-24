"""Text-query residual fusion for the clean Q2 validation ablation."""

import math

import torch
from torch import nn


class TextCenteredResidualFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.audio_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.vision_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        def gate():
            return nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model, d_model), nn.Sigmoid())

        self.audio_gate = gate()
        self.vision_gate = gate()
        init_alpha = math.log(0.1 / 0.9)
        self.audio_alpha_logit = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.vision_alpha_logit = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, text, audio, vision, valid_mask, use_audio=True, use_vision=True):
        padding_mask = ~valid_mask
        fused = text
        if use_audio:
            audio_ctx, _ = self.audio_cross(text, audio, audio, key_padding_mask=padding_mask,
                                            need_weights=False)
            audio_gate = self.audio_gate(torch.cat([text, audio_ctx], dim=-1))
            fused = fused + torch.sigmoid(self.audio_alpha_logit) * audio_gate * audio_ctx
        if use_vision:
            vision_ctx, _ = self.vision_cross(text, vision, vision, key_padding_mask=padding_mask,
                                              need_weights=False)
            vision_gate = self.vision_gate(torch.cat([text, vision_ctx], dim=-1))
            fused = fused + torch.sigmoid(self.vision_alpha_logit) * vision_gate * vision_ctx
        fused = self.norm1(fused)
        return self.norm2(fused + self.ffn(fused))
