import math

import numpy as np


def evidence_spans(modality, evidence, valid_len, cfg):
    if valid_len < 1:
        return []
    score = np.asarray(evidence["score"])[:valid_len]
    candidates = np.asarray(evidence["candidate_positions"], dtype=np.int64)
    count = min(len(candidates), max(1, math.ceil(cfg["explanation"]["evidence_ratio"] * valid_len)))
    selected = np.sort(candidates[np.argsort(score[candidates])[-count:]])
    groups = []
    current = [int(selected[0])]
    for t in selected[1:]:
        if int(t) == current[-1] + 1:
            current.append(int(t))
        else:
            groups.append(current)
            current = [int(t)]
    groups.append(current)
    spans = []
    for group in groups:
        start, end = group[0], group[-1] + 1
        length = end - start
        spans.append({
            "modality": modality, "grid_start": start, "grid_end_exclusive": end, "length": length,
            "mean_gate": float(np.mean(evidence["gate"][start:end])),
            "mean_occlusion": float(np.mean(evidence["occlusion"][start:end])),
            "evidence_score": float(np.mean(score[start:end]) * math.sqrt(length)),
        })
    spans.sort(key=lambda d: d["evidence_score"], reverse=True)
    spans = spans[:cfg["explanation"]["max_spans_per_modality"]]
    for rank, span in enumerate(spans, 1):
        span["rank"] = rank
    return spans
