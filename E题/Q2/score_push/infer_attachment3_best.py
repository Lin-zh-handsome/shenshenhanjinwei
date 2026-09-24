"""Inference for the validation-selected Q2 BERT model on unlabeled Attachment 3."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.attachment3_dataset import Attachment3AlignedDataset
from diagnostics.oracle_text import OracleTextBaseline


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--compact-path')
    parser.add_argument('--bert-model-path', required=True)
    parser.add_argument('--attachment3-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.compact_path:
        from score_push.compact_best_model import load_compact
        model = load_compact(args.compact_path, args.bert_model_path, device)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = dict(checkpoint['config'])
        if cfg.get('text_mode') != 'pretrained_bert':
            raise ValueError('Checkpoint is not a deployable BERT model')
        cfg['bert_model_path'] = args.bert_model_path
        model = OracleTextBaseline(cfg).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    dataset = Attachment3AlignedDataset(args.attachment3_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    rows = []
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in raw.items()}
            logits, intensity = model(batch)
            probabilities = logits.softmax(dim=-1).float().cpu()
            intensity = intensity.float().cpu()
            for index, sample_id in enumerate(raw['sample_id']):
                rows.append({'sample_id': sample_id,
                             'pred_class': int(probabilities[index].argmax()),
                             'pred_label': ('Negative', 'Neutral', 'Positive')[
                                 int(probabilities[index].argmax())],
                             'p_negative': float(probabilities[index, 0]),
                             'p_neutral': float(probabilities[index, 1]),
                             'p_positive': float(probabilities[index, 2]),
                             'pred_intensity': float(intensity[index])})
    if len(rows) != len(dataset):
        raise ValueError('Prediction count mismatch')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} unlabeled predictions to {destination}')


if __name__ == '__main__':
    main()
