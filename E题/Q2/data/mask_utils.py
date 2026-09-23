import warnings
import torch


def infer_shared_valid_mask(text_bert, audio, vision, eps=1e-8):
    observed = torch.stack((text_bert[:, 1] > 0, audio.abs().sum(-1) > eps,
                            vision.abs().sum(-1) > eps), dim=-1).any(-1)
    b, t = observed.shape
    positions = torch.arange(t, device=observed.device).expand(b, -1)
    last = torch.where(observed, positions, -1).max(1).values
    if (last < 0).any():
        warnings.warn(f'{(last < 0).sum().item()} all-zero samples; keeping first position')
    return positions <= last.clamp(min=0)[:, None]


def infer_missing_candidates(text_bert, audio, vision, valid_mask, eps=1e-8):
    return {'T': valid_mask & (text_bert[:, 1] <= 0),
            'A': valid_mask & (audio.abs().sum(-1) <= eps),
            'V': valid_mask & (vision.abs().sum(-1) <= eps)}


def combine_masks(a, b):
    return {m: a[m] | b[m] for m in 'TAV'}
