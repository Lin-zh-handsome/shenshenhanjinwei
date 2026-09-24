import torch

from explain.modality_ablation import MODALITIES
from utils.io import model_inputs


def _normalize(values):
    if values.numel() == 0:
        return values
    low, high = values.min(), values.max()
    return (values - low) / (high - low).clamp_min(1e-8)


@torch.no_grad()
def local_evidence_scores(model, batch, means, cfg, full=None):
    """Run the 36 candidate occlusions in one batched forward for one sample."""
    if batch["valid_mask"].shape[0] != 1:
        raise ValueError("local_evidence_scores expects one sample")
    full = full or model(**model_inputs(batch), temperature=model.inference_temperature)
    length = int(batch["valid_mask"][0].sum())
    candidates = []
    gate_vectors = {}
    for modality in MODALITIES:
        gate = full[f"gate_{modality}"][0].float()
        gate_vectors[modality] = gate
        top = torch.argsort(gate[:length], descending=True)[:cfg["explanation"]["local_candidate_topk"]].tolist()
        candidates.extend((modality, int(t)) for t in top)
    changed = {k: batch[k].expand(len(candidates), -1, -1).clone() for k in MODALITIES}
    window = cfg["explanation"]["occlusion_window"]
    half = window // 2
    for row, (modality, t) in enumerate(candidates):
        s, e = max(0, t - half), min(length, t + half + 1)
        changed[modality][row, s:e] = torch.as_tensor(means[modality], device=changed[modality].device, dtype=changed[modality].dtype)
    changed["valid_mask"] = batch["valid_mask"].expand(len(candidates), -1)
    occ = model(**changed, temperature=model.inference_temperature)
    c = int(full["cls_prob"][0].argmax())
    cls_signed = full["cls_prob"][0, c] - occ["cls_prob"][:, c]
    reg_signed = full["reg_pred"][0] - occ["reg_pred"]
    cls = cls_signed.abs()
    reg = reg_signed.abs() / 6
    impacts = cfg["explanation"]["modality_cls_weight"] * cls + cfg["explanation"]["modality_reg_weight"] * reg
    result = {}
    for modality in MODALITIES:
        scores = torch.zeros(50, device=impacts.device)
        occlusion = torch.zeros(50, device=impacts.device)
        classification_signed = torch.zeros(50, device=impacts.device)
        regression_signed = torch.zeros(50, device=impacts.device)
        rows = [(i, t) for i, (m, t) in enumerate(candidates) if m == modality]
        positions = torch.tensor([t for _, t in rows], device=impacts.device, dtype=torch.long)
        values = impacts[torch.tensor([i for i, _ in rows], device=impacts.device)]
        gate_norm = _normalize(gate_vectors[modality][positions])
        occ_norm = _normalize(values)
        scores[positions] = cfg["explanation"]["local_gate_weight"] * gate_norm + cfg["explanation"]["local_occlusion_weight"] * occ_norm
        occlusion[positions] = values
        rows_idx = torch.tensor([i for i, _ in rows], device=impacts.device)
        classification_signed[positions] = cls_signed[rows_idx]
        regression_signed[positions] = reg_signed[rows_idx]
        result[modality] = {"score": scores.cpu().numpy(), "gate": gate_vectors[modality].cpu().numpy(),
                            "occlusion": occlusion.cpu().numpy(),
                            "classification_signed_effect": classification_signed.cpu().numpy(),
                            "regression_signed_effect": regression_signed.cpu().numpy(),
                            "candidate_positions": positions.cpu().tolist()}
    return result
