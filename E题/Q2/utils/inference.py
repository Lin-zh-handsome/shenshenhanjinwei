import numpy as np
import torch
from torch.utils.data import DataLoader
from data.mask_utils import infer_shared_valid_mask, infer_missing_candidates, combine_masks
from data.missing_simulator import deterministic_span_mask
from utils.metrics import metrics


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def fixed_missing(valid, offset, seed=101, rate=0.30):
    combos = ('T', 'A', 'V', 'TA', 'TV', 'AV')
    out = {m: torch.zeros_like(valid) for m in 'TAV'}
    for i in range(len(valid)):
        combo = combos[(offset+i) % len(combos)]
        one = deterministic_span_mask(valid[i:i+1], combo, rate=rate, position='random', seed=seed+offset+i)
        for m in 'TAV':
            out[m][i] = one[m][0]
    return out


@torch.no_grad()
def predict_dataset(model, dataset, device, batch_size=32, missing_fn=None):
    model.eval()
    rows = []
    offset = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for raw in loader:
        batch = to_device(raw, device)
        valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
        original = infer_missing_candidates(batch['text_bert'], batch['audio'], batch['vision'], valid)
        artificial = missing_fn(valid, offset) if missing_fn is not None else {m: torch.zeros_like(valid) for m in 'TAV'}
        missing = combine_masks(original, artificial)
        out = model(batch, valid, missing)
        probs = out['cls_prob'].float().cpu().numpy()
        reg = out['reg_pred'].float().cpu().numpy()
        rel = out['reliability'].float().cpu().numpy()
        vmask = valid.cpu().numpy()
        for i, sample_id in enumerate(raw['id']):
            row = {'id': str(sample_id), 'true_class': int(raw['y_cls'][i]),
                   'pred_class': int(np.argmax(probs[i])), 'p_neg': float(probs[i, 0]),
                   'p_neu': float(probs[i, 1]), 'p_pos': float(probs[i, 2]),
                   'true_reg': float(raw['y_reg'][i]), 'pred_reg': float(reg[i]),
                   'abs_reg_error': float(abs(float(raw['y_reg'][i]) - float(reg[i])))}
            for j, m in enumerate(('text', 'audio', 'vision')):
                row[f'mean_rel_{m}'] = float(rel[i, vmask[i], j].mean())
            rows.append(row)
        offset += len(raw['id'])
    score = metrics([r['true_class'] for r in rows], [r['pred_class'] for r in rows],
                    [r['true_reg'] for r in rows], [r['pred_reg'] for r in rows])
    return score, rows
