"""One batch forward/loss contract check; no optimizer step or model training."""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.aligned_dataset import load_aligned
from data.mask_utils import infer_shared_valid_mask
from baseline_lnln.common import (BaselineDataset, read_config, official_components,
                                  original_random_masks, make_inputs, training_labels, to_device)


def main():
    cfg = read_config(ROOT/'baseline_lnln'/'config.yaml')
    build_model, loss_class, _, official_cfg = official_components(cfg)
    source = load_aligned(cfg['paths']['aligned_pkl'])
    batch = next(iter(DataLoader(BaselineDataset(source, 'train'), batch_size=2, shuffle=False)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = to_device(batch, device)
    valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
    masks, rates = original_random_masks(valid, batch['text_bert'])
    model = build_model(official_cfg).to(device).eval()
    with torch.no_grad():
        result = model(*make_inputs(batch, masks))
        loss = loss_class(official_cfg)(result, training_labels(batch, rates))['loss']
    assert result['sentiment_preds'].shape == (2, 1)
    assert torch.isfinite(loss)
    print('LNLN official model + aligned adapter forward/loss contract passed', flush=True)


if __name__ == '__main__':
    main()
