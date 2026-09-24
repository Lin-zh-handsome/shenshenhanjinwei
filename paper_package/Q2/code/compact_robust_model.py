"""Store only the selected missing-module delta beside the compact BERT backbone.

The public bert-base-uncased weights and the separately packaged compact
last-four-layer backbone are required to restore the complete model.
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from data.attachment3_dataset import Attachment3AlignedDataset
from diagnostics.oracle_text import to_device
from missing_train import evaluate
from models.srf_msa import SRFMSA
from score_push.compact_best_model import load_compact


def save_delta(checkpoint_path, delta_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['config']['run_name'] not in {
        'R1_Continuous_Missing', 'R2_Span_Geometry',
        'R3_Temporal_Reconstruction', 'R4_Reliability_Residual',
        'R5_Clean_Masked_Consistency'}:
        raise ValueError('Expected a selected BERT local-missing ablation checkpoint')
    cfg = dict(checkpoint['config']['model'])
    cfg['bert'] = dict(cfg['bert'])
    cfg['bert'].pop('local_path', None)
    delta = {key: value.cpu() for key, value in checkpoint['state_dict'].items()
             if not key.startswith('backbone.')}
    payload = {'format': 'q2_robust_delta_v1', 'model_config': cfg,
               'run_name': checkpoint['config']['run_name'],
               'source_epoch': int(checkpoint['epoch']), 'delta': delta}
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, delta_path)
    print(f'Delta parameters: {delta_path} ({delta_path.stat().st_size / 2**20:.2f} MiB)')


def load_model(base_compact_path, delta_path, bert_model_path, device):
    payload = torch.load(delta_path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'q2_robust_delta_v1':
        raise ValueError('Unknown Q2 robust delta format')
    cfg = dict(payload['model_config'])
    cfg['bert'] = dict(cfg['bert'])
    cfg['bert']['local_path'] = bert_model_path
    base = load_compact(base_compact_path, bert_model_path, torch.device('cpu'))
    model = SRFMSA(cfg)
    state = model.state_dict()
    state.update({f'backbone.{key}': value for key, value in base.state_dict().items()})
    state.update(payload['delta'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_raw_model(checkpoint_path, bert_model_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = dict(checkpoint['config']['model'])
    cfg['bert'] = dict(cfg['bert'])
    cfg['bert']['local_path'] = bert_model_path
    model = SRFMSA(cfg)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval()


def validate(model, aligned_pkl, report_path):
    with Path(aligned_pkl).open('rb') as handle:
        source = pickle.load(handle)
    dataset = AlignedMoseiDataset(split='valid', source=source)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    device = next(model.parameters()).device
    clean, _ = evaluate(model, loader, device)
    missing, _ = evaluate(model, loader, device, missing='mixed')
    report = {'split': 'valid', 'samples': len(dataset),
              'clean_valid': clean, 'missing_valid': missing, 'test_evaluated': False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def infer_attachment3(model, directory, output_path):
    dataset = Attachment3AlignedDataset(directory)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for raw in loader:
            batch = to_device(raw, device)
            result = model(batch)
            probabilities = result['cls_prob'].float().cpu()
            regression = result['reg_pred'].float().cpu()
            for index, sample_id in enumerate(raw['sample_id']):
                pred = int(probabilities[index].argmax())
                rows.append({'sample_id': sample_id, 'pred_class': pred,
                             'pred_label': ('Negative', 'Neutral', 'Positive')[pred],
                             'p_negative': float(probabilities[index, 0]),
                             'p_neutral': float(probabilities[index, 1]),
                             'p_positive': float(probabilities[index, 2]),
                             'pred_intensity': float(regression[index]),
                             'candidate_missing_text_steps': int(raw['text_missing_mask'][index].sum()),
                             'candidate_missing_audio_steps': int(raw['audio_missing_mask'][index].sum()),
                             'candidate_missing_vision_steps': int(raw['vision_missing_mask'][index].sum()),
                             'mask_source': 'zero_derived_candidates'})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} unlabeled Attachment 3 predictions to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--raw-inference-checkpoint')
    parser.add_argument('--base-compact', required=True)
    parser.add_argument('--delta-path', required=True)
    parser.add_argument('--bert-model-path')
    parser.add_argument('--aligned-pkl')
    parser.add_argument('--validation-report')
    parser.add_argument('--attachment3-dir')
    parser.add_argument('--predictions-output')
    args = parser.parse_args()
    delta_path = Path(args.delta_path)
    if args.source_checkpoint:
        save_delta(Path(args.source_checkpoint), delta_path)
    if args.aligned_pkl or args.attachment3_dir:
        if not args.bert_model_path:
            raise ValueError('Restoring the model requires --bert-model-path')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = (load_raw_model(Path(args.raw_inference_checkpoint), args.bert_model_path, device)
                 if args.raw_inference_checkpoint else
                 load_model(Path(args.base_compact), delta_path, args.bert_model_path, device))
        if args.aligned_pkl:
            if not args.validation_report:
                raise ValueError('Validation requires --validation-report')
            validate(model, args.aligned_pkl, Path(args.validation_report))
        if args.attachment3_dir:
            if not args.predictions_output:
                raise ValueError('Attachment 3 inference requires --predictions-output')
            infer_attachment3(model, args.attachment3_dir, Path(args.predictions_output))


if __name__ == '__main__':
    main()
