import torch.nn as nn


class ModalityTemporalEncoder(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.adapter = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d), nn.GELU(), nn.Dropout(cfg["dropout"]))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg["n_heads"], dim_feedforward=cfg["ff_dim"], dropout=cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["temporal_layers"], enable_nested_tensor=False)

    def forward(self, x, valid_mask):
        return self.encoder(self.adapter(x), src_key_padding_mask=~valid_mask)
