import torch
import torch.nn as nn

from models.temporal_position import TemporalPositionEmbedding


class ModalityTemporalEncoder(nn.Module):
    def __init__(self, input_dim, cfg, modality=None):
        super().__init__()
        d = cfg["d_model"]
        self.adapter = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d), nn.GELU(), nn.Dropout(cfg["dropout"]))
        self.position = TemporalPositionEmbedding(cfg.get("max_len", 50), d) if cfg.get("use_position", False) else None
        if self.position is not None:
            self.modality_embedding = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.modality_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg["n_heads"], dim_feedforward=cfg["ff_dim"], dropout=cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["temporal_layers"], enable_nested_tensor=False)

    def forward(self, x, valid_mask):
        projected = self.adapter(x)
        if self.position is not None:
            projected = self.position(projected) + self.modality_embedding
        return self.encoder(projected, src_key_padding_mask=~valid_mask)
