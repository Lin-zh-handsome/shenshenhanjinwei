"""Contiguous local missing spans, with a fixed gradual training curriculum."""

import random
import torch


def curriculum(epoch):
    if epoch <= 3:
        return 0.0, 1, 0
    if epoch <= 7:
        return 0.25, 1, 8
    if epoch <= 12:
        return 0.40, 1, 15
    return 0.50, 2, 20


class CurriculumSpanMasker:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def sample(self, valid, epoch):
        probability, max_modalities, max_length = curriculum(epoch)
        masks = {m: torch.zeros_like(valid, dtype=torch.bool) for m in 'TAV'}
        for sample, row in enumerate(valid):
            length = int(row.sum())
            if length < 2 or self.rng.random() >= probability:
                continue
            count = self.rng.randint(1, max_modalities)
            for modality in self.rng.sample('TAV', count):
                span = self.rng.randint(1, min(max_length, length - 1))
                start = self.rng.randint(0, length - span)
                masks[modality][sample, start:start + span] = True
        return masks


def deterministic_spans(valid, combo='T', ratio=0.3, position='middle',
                        duration=None, seed=2026):
    rng = random.Random(seed)
    masks = {m: torch.zeros_like(valid, dtype=torch.bool) for m in 'TAV'}
    for sample, row in enumerate(valid):
        length = int(row.sum())
        if length < 2:
            continue
        span = min(length - 1, duration if duration is not None else max(1, round(length * ratio)))
        if position == 'early':
            start = 0
        elif position == 'middle':
            start = (length - span) // 2
        elif position == 'late':
            start = length - span
        elif position == 'random':
            start = rng.randint(0, length - span)
        else:
            raise ValueError(position)
        for modality in combo:
            masks[modality][sample, start:start + span] = True
    return masks
