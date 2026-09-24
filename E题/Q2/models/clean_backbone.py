"""Shared clean and missing-aware multimodal backbone for Q2."""

import torch
from torch import nn

from models.bert_text_encoder import BertTextEncoder
from models.hierarchical_sentiment_head import HierarchicalSentimentHead
from models.text_centered_fusion import TextCenteredResidualFusion


class CleanMultimodalBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['d_model']
        drop = cfg['dropout']
        self.use_position = cfg.get('use_position', False)
        self.modalities = cfg.get('modalities', 'TAV')
        self.text_mode = cfg.get('text_mode', 'teacher_feature')
        self.fusion_mode = cfg.get('fusion_mode', 'concat')
        self.head_mode = cfg.get('head_mode', 'flat')
        self.regression_head_mode = cfg.get('regression_head_mode', 'linear')
        self.add_position_before_fused_encoder = cfg.get('add_position_before_fused_encoder', True)
        if self.text_mode == 'pretrained_bert':
            bert_cfg = cfg.get('bert', {})
            model_path = bert_cfg.get('local_path', cfg.get('bert_model_path', bert_cfg.get('model_name')))
            if not model_path:
                raise ValueError('pretrained_bert requires a BERT model path')
            self.bert = BertTextEncoder.from_pretrained(model_path, add_pooling_layer=False)
            mode = bert_cfg.get('mode', 'partial' if cfg.get('unfreeze_last_n_layers', 0) else 'frozen')
            n = bert_cfg.get('unfreeze_last_n', cfg.get('unfreeze_last_n_layers', 0))
            self.bert.configure_trainability(mode, n)
            self.bert_mode = mode
        elif self.text_mode != 'teacher_feature':
            raise ValueError(f'Unknown text_mode: {self.text_mode}')
        self.text = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.audio = nn.Sequential(nn.LayerNorm(74), nn.Linear(74, dim))
        self.vision = nn.Sequential(nn.LayerNorm(35), nn.Linear(35, dim))
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.position = nn.Embedding(50, dim) if self.use_position else None
        if self.position is not None:
            nn.init.normal_(self.position.weight, std=0.02)

        def temporal_encoder():
            layer = nn.TransformerEncoderLayer(dim, cfg['heads'], 2 * dim, drop,
                                               batch_first=True, activation='gelu')
            return nn.TransformerEncoder(layer, cfg['temporal_layers'], enable_nested_tensor=False)

        self.temporal = nn.ModuleList([temporal_encoder() for _ in range(3)])
        if self.fusion_mode == 'concat':
            self.fusion = nn.Sequential(nn.Linear(len(self.modalities) * dim, dim),
                                        nn.GELU(), nn.LayerNorm(dim))
        elif self.fusion_mode == 'text_centered':
            self.text_centered_fusion = TextCenteredResidualFusion(dim, cfg['heads'], drop)
        else:
            raise ValueError(f'Unknown fusion_mode: {self.fusion_mode}')
        self.fused_encoder = temporal_encoder()
        self.pool = nn.Linear(dim, 1)
        if self.head_mode == 'flat':
            self.cls_head = nn.Linear(dim, 3)
            if self.regression_head_mode == 'linear':
                self.reg_head = nn.Linear(dim, 1)
            elif self.regression_head_mode == 'mlp':
                self.reg_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                                              nn.Dropout(drop), nn.Linear(dim // 2, 1))
            else:
                raise ValueError(f'Unknown regression_head_mode: {self.regression_head_mode}')
        elif self.head_mode == 'hierarchical':
            self.hierarchical_head = HierarchicalSentimentHead(dim, drop)
        else:
            raise ValueError(f'Unknown head_mode: {self.head_mode}')

    def train(self, mode=True):
        super().train(mode)
        if (self.text_mode == 'pretrained_bert'
                and not any(parameter.requires_grad for parameter in self.bert.parameters())):
            self.bert.eval()
        return self

    def valid_mask(self, batch):
        valid = batch.get('sequence_valid_mask')
        if valid is None:
            valid = batch['text_bert'][:, 1, :] > 0
        if not bool(valid.any(dim=1).all()):
            raise ValueError('A sample has no valid temporal positions')
        return valid

    def project_raw(self, batch):
        if self.text_mode == 'pretrained_bert':
            text_features = self.bert(batch['text_bert'])
        else:
            text_features = batch['text_teacher']
        return {'T': self.text(text_features), 'A': self.audio(batch['audio']),
                'V': self.vision(batch['vision'])}

    def encode_projected(self, projected, valid):
        position = None
        if self.position is not None:
            position = self.position(torch.arange(valid.shape[1], device=valid.device))[None]
        encoded = {}
        for index, modality in enumerate('TAV'):
            if modality not in self.modalities:
                continue
            features = projected[modality] + self.modality_embedding[index]
            if position is not None:
                features = features + position
            encoded[modality] = self.temporal[index](features, src_key_padding_mask=~valid)
        return {'valid': valid, 'projected': projected, 'encoded': encoded,
                'position': position, 'text_projected': projected['T']}

    def extract_modalities(self, batch, valid_mask=None):
        valid = self.valid_mask(batch) if valid_mask is None else valid_mask
        return self.encode_projected(self.project_raw(batch), valid)

    def fuse_encoded(self, encoded, valid, position=None, correction=None):
        if self.fusion_mode == 'concat':
            fused = self.fusion(torch.cat([encoded[modality] for modality in self.modalities], dim=-1))
        else:
            fused = self.text_centered_fusion(encoded['T'], encoded['A'], encoded['V'], valid)
        if correction is not None:
            fused = fused + correction
        if position is not None and self.add_position_before_fused_encoder:
            fused = fused + position
        fused = self.fused_encoder(fused, src_key_padding_mask=~valid)
        attention = self.pool(fused).squeeze(-1).masked_fill(~valid, -1e4).softmax(dim=1)
        pooled = (attention[..., None] * fused).sum(dim=1)
        return pooled, attention

    def predict_pooled(self, pooled):
        if self.head_mode == 'hierarchical':
            output = self.hierarchical_head(pooled)
            output['cls_logits'] = output['cls_prob'].clamp_min(1e-8).log()
            return output
        logits = self.cls_head(pooled)
        return {'cls_logits': logits, 'cls_prob': logits.softmax(dim=-1),
                'reg_pred': 3 * torch.tanh(self.reg_head(pooled).squeeze(-1))}

    def forward(self, batch, return_text=False, return_outputs=False):
        extracted = self.extract_modalities(batch)
        pooled, attention = self.fuse_encoded(extracted['encoded'], extracted['valid'],
                                               extracted['position'])
        output = self.predict_pooled(pooled)
        output.update({'attention': attention, 'projected': extracted['projected'],
                       'encoded': extracted['encoded'], 'pooled': pooled})
        if return_outputs:
            return output
        pair = (output['cls_logits'], output['reg_pred'])
        return (*pair, extracted['text_projected']) if return_text else pair
