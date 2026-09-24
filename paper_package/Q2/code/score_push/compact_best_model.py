"""Package last-four-layer Q2 weights under the contest attachment size limit.

The compact artifact requires an external bert-base-uncased base checkpoint.
Validation here is read-only; test is never evaluated by this script.
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from diagnostics.oracle_text import OracleTextBaseline, evaluate_valid


def quantize_matrix(value):
    maximum = value.abs().amax(dim=1).clamp_min(1e-12)
    scale = maximum / 127.0
    values = torch.round(value / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return {'values': values, 'scale': scale}


def save_compact(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = dict(checkpoint['config'])
    if cfg.get('text_mode') != 'pretrained_bert' or cfg.get('unfreeze_last_n_layers') != 4:
        raise ValueError('Expected the validation-selected BERT last4 checkpoint')
    cfg = {key: value for key, value in cfg.items()
           if key not in {'aligned_pkl', 'output_dir', 'init_checkpoint', 'bert_model_path'}}
    cfg['bert_model_path'] = 'bert-base-uncased'
    finetuned = {}
    downstream = {}
    for key, value in checkpoint['state_dict'].items():
        if any(key.startswith(f'bert.encoder.layer.{layer}.') for layer in range(8, 12)):
            finetuned[key] = (quantize_matrix(value.float()) if value.ndim == 2
                              else {'values': value.float()})
        elif not key.startswith('bert.'):
            downstream[key] = value.cpu()
    payload = {'format': 'q2_bert_last4_row_int8_v1', 'base_model': 'bert-base-uncased',
               'source_epoch': int(checkpoint['epoch']), 'config': cfg,
               'finetuned_last4': finetuned, 'downstream': downstream}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f'Compact parameters: {output_path} ({output_path.stat().st_size / 2**20:.2f} MiB)')


def load_compact(path, bert_model_path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'q2_bert_last4_row_int8_v1':
        raise ValueError('Unknown compact parameter format')
    cfg = dict(payload['config'])
    cfg['bert_model_path'] = bert_model_path
    model = OracleTextBaseline(cfg)
    state = model.state_dict()
    state.update(payload['downstream'])
    for key, record in payload['finetuned_last4'].items():
        value = record['values']
        state[key] = (value.float() * record['scale'][:, None] if 'scale' in record
                      else value.float())
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def validate(path, bert_model_path, valid_pkl, report_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_compact(path, bert_model_path, device)
    with Path(valid_pkl).open('rb') as handle:
        source = pickle.load(handle)
    dataset = AlignedMoseiDataset(split='valid', source=source)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    score, rows = evaluate_valid(model, loader, device)
    report = {'split': 'valid', 'samples': len(rows), 'metrics': score,
              'compact_parameters_mib': round(path.stat().st_size / 2**20, 3),
              'test_evaluated': False}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--compact-path', required=True)
    parser.add_argument('--bert-model-path')
    parser.add_argument('--valid-pkl')
    parser.add_argument('--validation-report')
    args = parser.parse_args()
    path = Path(args.compact_path)
    if args.source_checkpoint:
        save_compact(args.source_checkpoint, path)
    if args.valid_pkl:
        if not args.bert_model_path or not args.validation_report:
            raise ValueError('Validation requires BERT path and report path')
        validate(path, args.bert_model_path, args.valid_pkl, Path(args.validation_report))


if __name__ == '__main__':
    main()
