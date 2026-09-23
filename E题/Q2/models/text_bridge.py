import torch
from torch import nn


class TextFeatureBridge(nn.Module):
    def __init__(self, vocab_size=30522, d_model=128, max_len=50, n_heads=4,
                 num_layers=2, ff_dim=256, dropout=0.15):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
        self.token_type = nn.Embedding(2, d_model)
        self.missing_token = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.teacher_head = nn.Linear(d_model, 768)

    def forward(self, text_bert, shared_valid_mask, text_missing_mask=None):
        b, _, t = text_bert.shape
        ids = text_bert[:, 0].clamp(0, self.token.num_embeddings - 1)
        typ = text_bert[:, 2].clamp(0, 1)
        h = self.token(ids) + self.position(torch.arange(t, device=ids.device))[None] + self.token_type(typ)
        if text_missing_mask is not None:
            h = torch.where(text_missing_mask[..., None], self.missing_token[None, None, :], h)
        h = self.encoder(h, src_key_padding_mask=~shared_valid_mask)
        return h, self.teacher_head(h)
