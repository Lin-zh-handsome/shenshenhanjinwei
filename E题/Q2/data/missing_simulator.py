import random
import torch


def _empty(valid_mask):
    return {m: torch.zeros_like(valid_mask) for m in 'TAV'}


class ContinuousSpanMasker:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = random.Random(seed)

    def sample(self, valid_mask):
        masks = _empty(valid_mask)
        meta = []
        combo_weights = self.cfg['modality_combo_weights']
        choices, weights = zip(*combo_weights.items())
        bins = list(self.cfg['length_bins'].values())
        for i, row in enumerate(valid_mask):
            length = int(row.sum())
            spans = []
            if self.rng.random() < self.cfg['probability'] and length > 1:
                combo = self.rng.choices(choices, weights)[0]
                shared = self.rng.random() < 0.30
                anchor = None
                for m in combo:
                    for j in range(self.rng.randint(1, self.cfg['max_spans_per_modality'])):
                        lo, hi, _ = self.rng.choices(bins, [v[2] for v in bins])[0]
                        span_len = min(self.rng.randint(lo, hi), max(1, length - 1))
                        if shared and anchor is not None and j == 0:
                            start = min(anchor, length - span_len)
                        else:
                            start = self.rng.randrange(length - span_len + 1)
                        if anchor is None:
                            anchor = start
                        end = start + span_len
                        masks[m][i, start:end] = True
                        spans.append({'modality': m, 'start': start, 'end_exclusive': end,
                                      'length': span_len, 'relative_start': start / length,
                                      'relative_length': span_len / length})
            meta.append({'spans': spans})
        masks['meta'] = meta
        return masks


def deterministic_span_mask(valid_mask, modality_combo, rate=0.30, position='random', seed=101, duration=None):
    masks = _empty(valid_mask)
    rng = random.Random(seed)
    for i, row in enumerate(valid_mask):
        length = int(row.sum())
        if length <= 1:
            continue
        n = min(max(1, round(length * rate)) if duration is None else duration, max(1, int(length * 0.8)))
        if position == 'early':
            start = 0
        elif position == 'middle':
            start = (length - n) // 2
        elif position == 'late':
            start = length - n
        elif position == 'random':
            start = rng.randrange(length - n + 1)
        else:
            raise ValueError(position)
        for m in modality_combo:
            masks[m][i, start:start+n] = True
    return masks
