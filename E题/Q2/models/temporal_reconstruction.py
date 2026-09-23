import torch
from torch import nn


class TemporalReconstructionEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, num_layers=2, ff_dim=256, dropout=0.15):
        super().__init__()
        self.missing_token = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.recon_head = nn.Linear(d_model, d_model)

    def forward(self, clean_proj, missing_mask, valid_mask, geometry_embed):
        h = torch.where(missing_mask[..., None], self.missing_token[None, None, :], clean_proj) + geometry_embed
        hidden = self.encoder(h, src_key_padding_mask=~valid_mask)
        return hidden, self.recon_head(hidden)
