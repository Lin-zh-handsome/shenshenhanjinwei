import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.attachment3_dataset import Attachment3AlignedDataset
from data.mask_utils import infer_shared_valid_mask, infer_missing_candidates
from evaluate import load_model
from utils.inference import to_device
from utils.io import load_config, output_dir, device_for


LABELS = ('Negative', 'Neutral', 'Positive')


@torch.no_grad()
def infer(cfg, checkpoint):
    device = device_for(cfg)
    dataset = Attachment3AlignedDataset(cfg['paths']['attachment3_dir'])
    model, _ = load_model(checkpoint, device)
    rows = []
    for raw in DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0):
        batch = to_device(raw, device)
        valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
        missing = infer_missing_candidates(batch['text_bert'], batch['audio'], batch['vision'], valid)
        out = model(batch, valid, missing)
        p = out['cls_prob'].float().cpu().numpy()
        y = out['reg_pred'].float().cpu().numpy()
        for i, sample_id in enumerate(raw['sample_id']):
            pred = int(np.argmax(p[i]))
            rows.append({'sample_id': sample_id, 'pred_class_id': pred, 'pred_label': LABELS[pred],
                         'prob_negative': float(p[i, 0]), 'prob_neutral': float(p[i, 1]),
                         'prob_positive': float(p[i, 2]), 'pred_intensity': float(np.clip(y[i], -3, 3))})
    if len(rows) != len(dataset):
        raise RuntimeError(f'inference rows {len(rows)} != files {len(dataset)}')
    path = output_dir(cfg)/'attachment3_predictions.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path}: {len(rows)} rows', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    parser.add_argument('--checkpoint', default='outputs/q2/best_joint.pt')
    args = parser.parse_args()
    infer(load_config(args.config), args.checkpoint)


if __name__ == '__main__':
    main()
