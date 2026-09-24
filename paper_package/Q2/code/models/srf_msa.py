"""Formal Q2 model: one BERT clean backbone plus optional local-missing modules."""

import torch
from torch import nn
from torch.nn import functional as F

from .clean_backbone import CleanMultimodalBackbone
from .span_geometry import SpanGeometryEncoder
from .temporal_reconstruction import TemporalReconstructionEncoder


class IndependentReliabilityResidual(nn.Module):
    def __init__(self, dim, initial_scale=0.1):
        super().__init__()
        self.gates = nn.ModuleDict({modality: nn.Linear(2 * dim + 1, 1) for modality in 'TAV'})
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, encoded, geometry, missing):
        corrections = []
        gates = {}
        for modality in 'TAV':
            features = torch.cat((encoded[modality], geometry[modality],
                                  missing[modality][..., None].float()), dim=-1)
            gate = torch.sigmoid(self.gates[modality](features))
            gates[modality] = gate.squeeze(-1)
            corrections.append(gate * encoded[modality])
        return self.scale * torch.stack(corrections, dim=0).mean(dim=0), gates


class SRFMSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = CleanMultimodalBackbone(cfg)
        self.use_span_geometry = bool(cfg.get('use_span_geometry', False))
        self.use_reconstruction = bool(cfg.get('use_reconstruction', False))
        self.use_reliability = bool(cfg.get('use_reliability', False))
        dim = cfg['d_model']
        self.mask_tokens = nn.ParameterDict({m: nn.Parameter(torch.zeros(dim)) for m in 'TAV'})
        if self.use_span_geometry:
            self.geometry = nn.ModuleDict({m: SpanGeometryEncoder(dim) for m in 'TAV'})
        if self.use_reconstruction:
            self.reconstruct = nn.ModuleDict({m: TemporalReconstructionEncoder(
                dim, cfg['heads'], cfg.get('reconstruction_layers', 1), 2 * dim,
                cfg['dropout']) for m in 'TAV'})
        if self.use_reliability:
            self.reliability = IndependentReliabilityResidual(dim)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, batch, clean_batch=None):
        valid = self.backbone.valid_mask(batch)
        masks = {m: batch[f'{key}_missing_mask'].bool() & valid
                 for m, key in (('T', 'text'), ('A', 'audio'), ('V', 'vision'))}
        projected = self.backbone.project_raw(batch)
        geometry = {m: (self.geometry[m](masks[m], valid) if self.use_span_geometry
                        else torch.zeros_like(projected[m])) for m in 'TAV'}
        encoded_input = {}
        reconstruction = {}
        for modality in 'TAV':
            masked = torch.where(masks[modality][..., None],
                                 self.mask_tokens[modality][None, None, :], projected[modality])
            if self.use_reconstruction and masks[modality].any():
                _, reconstructed = self.reconstruct[modality](masked, masks[modality], valid,
                                                               geometry[modality])
                reconstruction[modality] = reconstructed
                encoded_input[modality] = torch.where(masks[modality][..., None],
                                                       reconstructed, masked)
            else:
                encoded_input[modality] = masked + geometry[modality]
        extracted = self.backbone.encode_projected(encoded_input, valid)
        correction = None
        gates = None
        if self.use_reliability:
            correction, gates = self.reliability(extracted['encoded'], geometry, masks)
        pooled, attention = self.backbone.fuse_encoded(extracted['encoded'], valid,
                                                       extracted['position'], correction)
        output = self.backbone.predict_pooled(pooled)
        output.update({'attention': attention, 'pooled': pooled, 'missing_masks': masks,
                       'reliability': gates})
        reconstruction_loss = torch.zeros((), device=valid.device)
        if self.use_reconstruction and clean_batch is not None:
            synthetic = batch['synthetic_missing_mask'].bool()
            with torch.no_grad():
                target = self.backbone.project_raw(clean_batch)
            terms = []
            for index, modality in enumerate('TAV'):
                supervised = synthetic[..., index] & valid
                if supervised.any():
                    terms.append(F.l1_loss(reconstruction[modality][supervised],
                                           target[modality][supervised]))
            if terms:
                reconstruction_loss = torch.stack(terms).mean()
        output['reconstruction_loss'] = reconstruction_loss
        return output
