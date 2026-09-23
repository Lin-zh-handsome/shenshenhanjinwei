import torch
from torch import nn
from .text_bridge import TextFeatureBridge
from .span_geometry import SpanGeometryEncoder
from .temporal_reconstruction import TemporalReconstructionEncoder
from .reliability_fusion import ReliabilityFusion


class SRFMSA(nn.Module):
    def __init__(self, cfg, variant=5):
        super().__init__()
        self.variant = variant
        d = cfg['d_model']
        common = (d, cfg['n_heads'], cfg['temporal_layers'], cfg['ff_dim'], cfg['dropout'])
        self.text_bridge = TextFeatureBridge(cfg['vocab_size'], d, cfg['max_len'], cfg['n_heads'], cfg['text_layers'], cfg['ff_dim'], cfg['dropout'])
        self.audio_adapter = nn.Sequential(nn.LayerNorm(74), nn.Linear(74, d), nn.GELU(), nn.Dropout(cfg['dropout']))
        self.vision_adapter = nn.Sequential(nn.LayerNorm(35), nn.Linear(35, d), nn.GELU(), nn.Dropout(cfg['dropout']))
        self.geometry = nn.ModuleDict({m: SpanGeometryEncoder(d) for m in 'TAV'})
        self.reconstruct = nn.ModuleDict({m: TemporalReconstructionEncoder(*common) for m in 'TAV'})
        self.reliability = ReliabilityFusion(d)
        self.basic_fuse = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(d, cfg['n_heads'], cfg['ff_dim'], cfg['dropout'], batch_first=True, activation='gelu')
        self.fused_encoder = nn.TransformerEncoder(layer, cfg['fusion_layers'], enable_nested_tensor=False)
        self.pool = nn.Linear(d, 1)
        self.cls_head = nn.Linear(d, 3)
        self.reg_head = nn.Linear(d, 1)

    def forward(self, batch, valid_mask, missing):
        text_hidden, teacher_recon = self.text_bridge(batch['text_bert'], valid_mask, missing['T'])
        proj = {'T': text_hidden, 'A': self.audio_adapter(batch['audio']), 'V': self.vision_adapter(batch['vision'])}
        geometry = {m: self.geometry[m](missing[m], valid_mask) if self.variant >= 2 else torch.zeros_like(proj[m]) for m in 'TAV'}
        hidden, recon = {}, {}
        for m in 'TAV':
            if self.variant >= 3:
                hidden[m], recon[m] = self.reconstruct[m](proj[m], missing[m], valid_mask, geometry[m])
            else:
                hidden[m] = torch.where(missing[m][..., None], torch.zeros_like(proj[m]), proj[m]) + geometry[m]
        if self.variant >= 4:
            fused, reliability = self.reliability(*(hidden[m] for m in 'TAV'), *(geometry[m] for m in 'TAV'),
                                                  *(missing[m] for m in 'TAV'), valid_mask)
        else:
            fused = self.basic_fuse(torch.cat([hidden[m] for m in 'TAV'], -1))
            reliability = torch.full((*valid_mask.shape, 3), 1/3, device=valid_mask.device)
        fused = self.fused_encoder(fused, src_key_padding_mask=~valid_mask)
        score = self.pool(fused).squeeze(-1).masked_fill(~valid_mask, -1e4)
        weights = score.softmax(1)
        pooled = (weights[..., None] * fused).sum(1)
        logits = self.cls_head(pooled)
        return {'cls_logits': logits, 'cls_prob': logits.softmax(-1), 'reg_pred': 3*torch.tanh(self.reg_head(pooled).squeeze(-1)),
                'teacher_recon': teacher_recon, 'proj': proj, 'recon': recon,
                'reliability': reliability, 'attention': weights}
