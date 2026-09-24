"""Clean sentiment backbone used for controlled score-push experiments."""

import torch
from torch import nn


class FeatureProjection(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                         nn.GELU(), nn.Dropout(dropout))


def temporal_encoder(dim, heads, feedforward, dropout, layers):
    layer = nn.TransformerEncoderLayer(dim, heads, feedforward, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TextCenteredFusion(nn.Module):
    def __init__(self, dim, heads, dropout, use_audio, use_vision):
        super().__init__()
        self.use_audio = use_audio
        self.use_vision = use_vision
        if use_audio:
            self.audio_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.audio_gate = nn.Linear(2 * dim, dim)
            self.alpha_audio = nn.Parameter(torch.tensor(0.1))
        if use_vision:
            self.vision_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.vision_gate = nn.Linear(2 * dim, dim)
            self.alpha_vision = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(2 * dim, dim), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, text, audio, vision, padding):
        fused = text
        if self.use_audio:
            context, _ = self.audio_attn(text, audio, audio, key_padding_mask=padding,
                                         need_weights=False)
            gate = torch.sigmoid(self.audio_gate(torch.cat((text, context), dim=-1)))
            fused = fused + self.alpha_audio * gate * context
        if self.use_vision:
            context, _ = self.vision_attn(text, vision, vision, key_padding_mask=padding,
                                          need_weights=False)
            gate = torch.sigmoid(self.vision_gate(torch.cat((text, context), dim=-1)))
            fused = fused + self.alpha_vision * gate * context
        fused = self.norm(fused)
        return self.ff_norm(fused + self.ff(fused))


class StrongCleanBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['d_model']
        heads = cfg['heads']
        dropout = cfg['dropout']
        self.cfg = cfg
        self.text_mode = cfg.get('text_mode', 'teacher_feature')
        self.modalities = cfg.get('modalities', 'TAV')
        self.fusion_mode = cfg.get('fusion_mode', 'concat')
        self.pooling_mode = cfg.get('pooling', 'attention')
        self.text_projection = FeatureProjection(768, dim, dropout)
        if self.text_mode == 'pretrained_bert':
            from transformers import BertModel
            model_path = cfg.get('bert_model_path')
            if not model_path:
                raise ValueError('pretrained_bert requires bert_model_path; random fallback is forbidden')
            self.bert = BertModel.from_pretrained(model_path, add_pooling_layer=False)
            for param in self.bert.parameters():
                param.requires_grad = False
            unfreeze = int(cfg.get('unfreeze_last_n_layers', 4))
            for layer in self.bert.encoder.layer[-unfreeze:] if unfreeze > 0 else []:
                for param in layer.parameters():
                    param.requires_grad = True
        elif self.text_mode != 'teacher_feature':
            raise ValueError(f'Unknown text mode: {self.text_mode}')
        self.audio_projection = FeatureProjection(74, dim, dropout) if 'A' in self.modalities else None
        self.vision_projection = FeatureProjection(35, dim, dropout) if 'V' in self.modalities else None
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.position = nn.Embedding(50, dim) if cfg.get('use_position', True) else None
        if self.position is not None:
            nn.init.normal_(self.position.weight, std=0.02)
        self.text_temporal = temporal_encoder(dim, heads, cfg['feedforward'], dropout,
                                              cfg['temporal_layers'])
        self.audio_temporal = temporal_encoder(dim, heads, cfg['feedforward'], dropout,
                                               cfg['temporal_layers']) if 'A' in self.modalities else None
        self.vision_temporal = temporal_encoder(dim, heads, cfg['feedforward'], dropout,
                                                cfg['temporal_layers']) if 'V' in self.modalities else None
        if self.fusion_mode == 'concat':
            count = len(self.modalities)
            self.fusion = nn.Sequential(nn.Linear(count * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        elif self.fusion_mode == 'text_centered':
            self.fusion = TextCenteredFusion(dim, heads, dropout, 'A' in self.modalities,
                                            'V' in self.modalities)
        else:
            raise ValueError(f'Unknown fusion mode: {self.fusion_mode}')
        self.fused_encoder = temporal_encoder(dim, heads, cfg['feedforward'], dropout,
                                              cfg['fusion_layers'])
        self.attention_pool = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.pool_projection = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        self.classifier = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2),
                                        nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.1),
                                        nn.Linear(dim // 2, 3))
        self.regressor = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.1),
                                       nn.Linear(dim // 2, 1))

    def forward(self, batch):
        valid = batch['text_bert'][:, 1, :] > 0
        padding = ~valid
        if self.text_mode == 'teacher_feature':
            text_input = batch['text_teacher']
        else:
            tokens = batch['text_bert']
            text_input = self.bert(input_ids=tokens[:, 0], attention_mask=tokens[:, 1],
                                   token_type_ids=tokens[:, 2]).last_hidden_state
        text_projected = self.text_projection(text_input)
        pos = self.position(torch.arange(valid.shape[1], device=valid.device))[None] if self.position is not None else 0
        text = self.text_temporal(text_projected + self.modality_embedding[0] + pos,
                                  src_key_padding_mask=padding)
        audio = None
        vision = None
        if self.audio_projection is not None:
            audio = self.audio_temporal(self.audio_projection(batch['audio']) +
                                        self.modality_embedding[1] + pos,
                                        src_key_padding_mask=padding)
        if self.vision_projection is not None:
            vision = self.vision_temporal(self.vision_projection(batch['vision']) +
                                          self.modality_embedding[2] + pos,
                                          src_key_padding_mask=padding)
        if self.fusion_mode == 'concat':
            features = {'T': text, 'A': audio, 'V': vision}
            fused = self.fusion(torch.cat([features[m] for m in self.modalities], dim=-1))
        else:
            fused = self.fusion(text, audio, vision, padding)
        fused = self.fused_encoder(fused + pos, src_key_padding_mask=padding)
        if self.pooling_mode == 'mean':
            pooled = (fused * valid[..., None]).sum(1) / valid.sum(1).clamp(min=1)[:, None]
        else:
            score = self.attention_pool(fused).squeeze(-1).masked_fill(padding, -1e4)
            attention = score.softmax(1)
            pooled = (attention[..., None] * fused).sum(1)
            if self.pooling_mode == 'attention_max':
                max_pooled = fused.masked_fill(padding[..., None], -1e4).max(1).values
                pooled = self.pool_projection(torch.cat((pooled, max_pooled), dim=-1))
            elif self.pooling_mode != 'attention':
                raise ValueError(f'Unknown pooling mode: {self.pooling_mode}')
        return {'logits': self.classifier(pooled),
                'regression': 3 * torch.tanh(self.regressor(pooled).squeeze(-1)),
                'text_projected': text_projected, 'valid_mask': valid}
