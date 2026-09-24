import warnings

import numpy as np


def infer_aligned_valid_mask(text_bert, text, audio, vision, eps=1e-8):
    """Exclude trailing padding while retaining internal zero rows."""
    t = np.asarray(text_bert)
    arrays = (np.asarray(text), np.asarray(audio), np.asarray(vision))
    candidates = [np.flatnonzero(t[1] > 0)]
    candidates += [np.flatnonzero(np.abs(x).sum(axis=-1) > eps) for x in arrays]
    ends = [int(x[-1]) for x in candidates if len(x)]
    if not ends:
        warnings.warn("All four validity signals are empty; retaining position 0", stacklevel=2)
    last = max(ends, default=0)
    mask = np.zeros(50, dtype=np.bool_)
    mask[: last + 1] = True
    return mask
