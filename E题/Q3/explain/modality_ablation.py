import torch


MODALITIES = ("text", "audio", "vision")


def replace_modality(batch, modality, mean):
    result = {k: v for k, v in batch.items()}
    x = batch[modality].clone()
    baseline = torch.as_tensor(mean[modality], dtype=x.dtype, device=x.device)
    x = torch.where(batch["valid_mask"].unsqueeze(-1), baseline, x)
    result[modality] = x
    return result


@torch.no_grad()
def modality_importance(model, batch, means, cfg):
    from utils.io import model_inputs
    full = model(**model_inputs(batch), temperature=model.inference_temperature)
    pred = full["cls_prob"].argmax(dim=1)
    full_class_prob = full["cls_prob"].gather(1, pred[:, None]).squeeze(1)
    scores, signed_cls, signed_reg = [], [], []
    for modality in MODALITIES:
        removed = model(**model_inputs(replace_modality(batch, modality, means)), temperature=model.inference_temperature)
        cls_delta = full_class_prob - removed["cls_prob"].gather(1, pred[:, None]).squeeze(1)
        reg_delta = full["reg_pred"] - removed["reg_pred"]
        score = cfg["explanation"]["modality_cls_weight"] * cls_delta.abs() + cfg["explanation"]["modality_reg_weight"] * reg_delta.abs() / 6
        scores.append(score)
        signed_cls.append(cls_delta)
        signed_reg.append(reg_delta)
    raw = torch.stack(scores, dim=1)
    importance = (raw + 1e-8) / (raw + 1e-8).sum(dim=1, keepdim=True)
    return {"full": full, "importance": importance, "cls_prob_delta_signed": torch.stack(signed_cls, dim=1), "reg_delta_signed": torch.stack(signed_reg, dim=1)}
