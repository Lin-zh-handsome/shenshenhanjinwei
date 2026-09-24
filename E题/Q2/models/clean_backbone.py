"""Configurable clean backbone preserving the 5790875 Run A architecture by default."""

import torch
from torch import nn

from models.text_centered_fusion import TextCenteredResidualFusion


class CleanMultimodalBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['d_model']
        drop = cfg['dropout']
        self.use_position = cfg['use_position']
        self.add_position_before_fused_encoder = cfg.get('add_position_before_fused_encoder', True)
        self.fusion_mode = cfg.get('fusion_mode', 'concat')
        self.pool_mode = cfg.get('pool_mode', 'attention')
        self.head_mode = cfg.get('head_mode', 'linear')
        self.enabled_modalities = set(cfg.get('enabled_modalities', ['text', 'audio', 'vision']))
        if 'text' not in self.enabled_modalities:
            raise ValueError('Clean backbone requires text as the anchor modality')
        if self.fusion_mode not in {'concat', 'text_centered'}:
            raise ValueError(f'Unknown fusion_mode: {self.fusion_mode}')
        if self.pool_mode not in {'attention', 'attention_max'}:
            raise ValueError(f'Unknown pool_mode: {self.pool_mode}')
        if self.head_mode not in {'linear', 'mlp'}:
            raise ValueError(f'Unknown head_mode: {self.head_mode}')

        self.text = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.audio = nn.Sequential(nn.LayerNorm(74), nn.Linear(74, dim))
        self.vision = nn.Sequential(nn.LayerNorm(35), nn.Linear(35, dim))
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.position = nn.Embedding(50, dim) if self.use_position else None

        def temporal_encoder():
            layer = nn.TransformerEncoderLayer(dim, cfg['heads'], 2 * dim, drop,
                                               batch_first=True, activation='gelu')
            return nn.TransformerEncoder(layer, cfg['temporal_layers'], enable_nested_tensor=False)

        self.temporal = nn.ModuleList([temporal_encoder() for _ in range(3)])
        if self.fusion_mode == 'concat':
            self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        else:
            self.text_centered_fusion = TextCenteredResidualFusion(dim, cfg['heads'], drop)
        self.fused_encoder = temporal_encoder()
        self.pool = nn.Linear(dim, 1)
        if self.pool_mode == 'attention_max':
            self.pool_merge = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        if self.head_mode == 'linear':
            self.cls_head = nn.Linear(dim, 3)
            self.reg_head = nn.Linear(dim, 1)
        else:
            self.cls_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.20),
                                          nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.10),
                                          nn.Linear(dim // 2, 3))
            self.reg_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.10),
                                          nn.Linear(dim // 2, 1))

    def forward(self, batch):
        valid = batch['text_bert'][:, 1, :] > 0
        if not bool(valid.any(1).all()):
            raise ValueError('An Oracle Text sample has no valid token positions')
        x = [self.text(batch['text_teacher']), self.audio(batch['audio']),
             self.vision(batch['vision'])]
        pos = None
        if self.position is not None:
            pos = self.position(torch.arange(valid.shape[1], device=valid.device))[None]
        encoded = []
        for index, (features, encoder) in enumerate(zip(x, self.temporal)):
            modality = ('text', 'audio', 'vision')[index]
            if modality not in self.enabled_modalities:
                encoded.append(torch.zeros_like(x[0]))
                continue
            features = features + self.modality_embedding[index]
            if pos is not None:
                features = features + pos
            encoded.append(encoder(features, src_key_padding_mask=~valid))
        if self.fusion_mode == 'concat':
            fused = self.fusion(torch.cat(encoded, dim=-1))
        else:
            fused = self.text_centered_fusion(encoded[0], encoded[1], encoded[2], valid,
                                               use_audio='audio' in self.enabled_modalities,
                                               use_vision='vision' in self.enabled_modalities)
        if pos is not None and self.add_position_before_fused_encoder:
            fused = fused + pos
        fused = self.fused_encoder(fused, src_key_padding_mask=~valid)
        attention = self.pool(fused).squeeze(-1).masked_fill(~valid, -1e4).softmax(1)
        pooled = (attention[..., None] * fused).sum(1)
        if self.pool_mode == 'attention_max':
            masked = fused.masked_fill(~valid[..., None], torch.finfo(fused.dtype).min)
            max_pooled = masked.max(dim=1).values
            pooled = self.pool_merge(torch.cat([pooled, max_pooled], dim=-1))
        return self.cls_head(pooled), 3 * torch.tanh(self.reg_head(pooled).squeeze(-1))
