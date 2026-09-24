"""One definition of sequence, observed, and synthetic-missing masks for Q2."""

import torch

MODALITIES = {'T': 'text', 'A': 'audio', 'V': 'vision'}


def sequence_valid_from_signals(text_bert, audio, vision, eps=1e-8):
    """Find the last non-padding point using the union of three signals.

    Zero values are used only to locate the sequence end, never as missing labels.
    """
    signals = (text_bert[:, 1] > 0) | (audio.abs().sum(-1) > eps) | (vision.abs().sum(-1) > eps)
    positions = torch.arange(signals.shape[1], device=signals.device)[None].expand_as(signals)
    last = torch.where(signals, positions, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError('A sample has no valid temporal position')
    return positions <= last[:, None]


def base_masks(text_bert, audio, vision, observed=None):
    valid = sequence_valid_from_signals(text_bert, audio, vision)
    result = {'sequence_valid_mask': valid}
    for key in MODALITIES.values():
        value = valid if observed is None or key not in observed else observed[key].bool() & valid
        result[f'{key}_observed_mask'] = value
        result[f'{key}_missing_mask'] = valid & ~value
    result['synthetic_missing_mask'] = torch.zeros((*valid.shape, 3), dtype=torch.bool, device=valid.device)
    return result


def apply_synthetic_missing(batch, masks):
    """Apply exactly the supplied contiguous spans and carry their masks forward."""
    valid = batch['sequence_valid_mask'].bool()
    modified = dict(batch)
    synthetic = []
    for modality, key in MODALITIES.items():
        mask = masks[modality].bool() & valid & batch[f'{key}_observed_mask'].bool()
        synthetic.append(mask)
        observed = batch[f'{key}_observed_mask'].bool() & ~mask
        modified[f'{key}_observed_mask'] = observed
        modified[f'{key}_missing_mask'] = valid & ~observed
    modified['synthetic_missing_mask'] = torch.stack(synthetic, dim=-1)
    text_mask = synthetic[0]
    if text_mask.any():
        text_bert = batch['text_bert'].clone()
        text_bert[:, 0].masked_fill_(text_mask, 0)
        text_bert[:, 1].masked_fill_(text_mask, 0)
        text_bert[:, 2].masked_fill_(text_mask, 0)
        modified['text_bert'] = text_bert
    for mask, key in ((synthetic[1], 'audio'), (synthetic[2], 'vision')):
        if mask.any():
            modified[key] = batch[key].masked_fill(mask[..., None], 0)
    return modified


# Compatibility names for old experiment scripts. They are not synthetic labels.
infer_shared_valid_mask = sequence_valid_from_signals


def infer_missing_candidates(text_bert, audio, vision, valid_mask, eps=1e-8):
    return {'T': valid_mask & (text_bert[:, 1] <= 0),
            'A': valid_mask & (audio.abs().sum(-1) <= eps),
            'V': valid_mask & (vision.abs().sum(-1) <= eps)}


def combine_masks(a, b):
    return {modality: a[modality] | b[modality] for modality in 'TAV'}
