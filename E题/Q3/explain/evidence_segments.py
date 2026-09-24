import math

import numpy as np


def evidence_spans(modality, evidence, valid_len, cfg, predicted_reg=None):
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
        if length < cfg["explanation"].get("min_span_len", 1):
            continue
        cls_signed = float(np.mean(evidence["classification_signed_effect"][start:end]))
        reg_signed = float(np.mean(evidence["regression_signed_effect"][start:end]))
        reg_aligned = predicted_reg is not None and abs(predicted_reg) >= 0.1 and reg_signed * predicted_reg > 0
        direction = "contradictory" if cls_signed < 0 else "supportive" if cls_signed > 0 and (reg_aligned or abs(reg_signed) < 0.01) else "mixed"
        spans.append({
            "modality": modality, "grid_start": start, "grid_end_exclusive": end, "length": length,
            "mean_gate": float(np.mean(evidence["gate"][start:end])),
            "mean_occlusion": float(np.mean(evidence["occlusion"][start:end])),
            "evidence_score": float(np.mean(score[start:end]) * math.sqrt(length)),
            "gate_score": float(np.mean(evidence["gate"][start:end])),
            "occlusion_magnitude": float(np.mean(evidence["occlusion"][start:end])),
            "classification_signed_effect": cls_signed,
            "regression_signed_effect": reg_signed,
            "evidence_direction": direction,
        })
    spans.sort(key=lambda d: d["evidence_score"], reverse=True)
    spans = spans[:cfg["explanation"]["max_spans_per_modality"]]
    for rank, span in enumerate(spans, 1):
        span["rank"] = rank
    return spans
