import torch
import torch.nn as nn

from models.modality_encoder import ModalityTemporalEncoder
from models.evidence_gate import ContextEvidenceGate
from models.evidence_router import EvidenceRouter, masked_mean, weighted_pool
from models.prediction_heads import PredictionHeads


class FERMSA(nn.Module):
    def __init__(self, cfg, variant="A5"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        m = cfg["model"]
        d = m["d_model"]
        self.encoders = nn.ModuleDict({k: ModalityTemporalEncoder(w, m) for k, w in (("text", 768), ("audio", 74), ("vision", 35))})
        self.context = nn.Sequential(nn.Linear(3 * d, d), nn.LayerNorm(d))
        self.gates = nn.ModuleDict({k: ContextEvidenceGate(d, m["gate_hidden"]) for k in self.encoders})
        self.router = EvidenceRouter(d, m["dropout"])
        self.heads = PredictionHeads(d)

    def _predict(self, reps, route):
        z, alpha = self.router(reps, route=route)
        logits, prob, reg = self.heads(z)
        return logits, prob, reg, alpha

    def forward(self, text, audio, vision, valid_mask, temperature=1.0, return_explain=False):
        inputs = {"text": text, "audio": audio, "vision": vision}
        h = {k: self.encoders[k](v, valid_mask) for k, v in inputs.items()}
        context = self.context(torch.cat(tuple(h.values()), dim=-1))
        use_gate = self.variant != "A0"
        gates = {k: self.gates[k](v, context, valid_mask, temperature) if use_gate else valid_mask.to(v.dtype) for k, v in h.items()}
        eps = self.cfg["model"]["residual_context"]
        reps = [weighted_pool(h[k], ((1 - eps) * gates[k] + eps) * valid_mask) if use_gate else masked_mean(h[k], valid_mask) for k in h]
        logits, prob, reg, alpha = self._predict(reps, route=self.variant in ("A3", "A4", "A5"))
        result = {"cls_logits": logits, "cls_prob": prob, "reg_pred": reg, "router_alpha": alpha,
                  **{f"gate_{k}": v for k, v in gates.items()}}
        if self.training and self.variant in ("A4", "A5"):
            full_reps = [masked_mean(h[k], valid_mask) for k in h]
            fl, fp, fr, _ = self._predict(full_reps, route=True)
            result.update(full_cls_logits=fl, full_cls_prob=fp, full_reg_pred=fr)
        if self.training and self.variant == "A5":
            comp_reps = [weighted_pool(h[k], (1 - gates[k]) * valid_mask) for k in h]
            _, cp, cr, _ = self._predict(comp_reps, route=True)
            result.update(comp_cls_prob=cp, comp_reg_pred=cr)
        return result
