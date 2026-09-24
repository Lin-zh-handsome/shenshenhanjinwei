import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplainableMultitaskLoss(nn.Module):
    def __init__(self, cfg, class_weights, variant="A5"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, out, batch, epoch=0):
        c = self.cfg["loss"]
        y_cls, y_reg, valid = batch["y_cls"], batch["y_reg"], batch["valid_mask"]
        loss_cls = F.cross_entropy(out["cls_logits"], y_cls, weight=self.class_weights)
        loss_reg = F.smooth_l1_loss(out["reg_pred"], y_reg, beta=0.5)
        loss = c["cls"] * loss_cls + c["reg"] * loss_reg
        details = {"cls": loss_cls.detach(), "reg": loss_reg.detach()}
        if self.variant in ("A2", "A3", "A4", "A5"):
            gates = torch.stack([out[f"gate_{k}"] for k in ("text", "audio", "vision")], dim=1)
            v = valid[:, None, :].to(gates.dtype)
            denom = v.sum(dim=-1).clamp_min(1)
            budget = (((gates * v).sum(dim=-1) / denom - self.cfg["model"]["evidence_ratio"]) ** 2).mean()
            binary = ((gates * (1 - gates) * v).sum(dim=-1) / denom).mean()
            pair = v[:, :, 1:] * v[:, :, :-1]
            tv = ((gates[:, :, 1:] - gates[:, :, :-1]).abs() * pair).sum(dim=-1).div(pair.sum(dim=-1).clamp_min(1)).mean()
            loss = loss + c["evidence_budget"] * budget + c["evidence_binary"] * binary + c["evidence_tv"] * tv
            details.update(budget=budget.detach(), binary=binary.detach(), tv=tv.detach())
        if self.variant in ("A4", "A5"):
            full = F.cross_entropy(out["full_cls_logits"], y_cls, weight=self.class_weights) + F.smooth_l1_loss(out["full_reg_pred"], y_reg, beta=0.5)
            suff = F.kl_div(out["cls_prob"].clamp_min(1e-8).log(), out["full_cls_prob"].detach(), reduction="batchmean") + F.smooth_l1_loss(out["reg_pred"], out["full_reg_pred"].detach(), beta=0.5)
            warm = min(1.0, epoch / 3)
            loss = loss + c["full_aux"] * full + c["sufficiency"] * warm * suff
            details.update(full=full.detach(), suff=suff.detach())
        if self.variant == "A5":
            idx = y_cls[:, None]
            fp = out["full_cls_prob"].gather(1, idx).squeeze(1)
            cp = out["comp_cls_prob"].gather(1, idx).squeeze(1)
            comp_cls = F.relu(c["comp_cls_margin"] - (fp - cp)).mean()
            delta_reg = (out["comp_reg_pred"] - y_reg).abs() - (out["full_reg_pred"] - y_reg).abs()
            comp_reg = F.relu(c["comp_reg_margin"] - delta_reg).mean()
            comp = comp_cls + comp_reg
            agree = F.smooth_l1_loss(out["cls_prob"][:, 2] - out["cls_prob"][:, 0], out["reg_pred"] / 3, beta=0.5)
            loss = loss + c["comprehensiveness"] * min(1.0, epoch / 3) * comp + c["cls_reg_agreement"] * agree
            details.update(comp=comp.detach(), agree=agree.detach())
        details["total"] = loss.detach()
        return loss, details
