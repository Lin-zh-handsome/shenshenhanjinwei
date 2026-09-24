"""Summarize only clean validation experiments on the 5790875 branch."""

import argparse
import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/q2_score_push'
RUN_A = ROOT / 'outputs/q2_v2/01_run_a_oracle_text'
FIELDS = ['run_name', 'accuracy', 'macro_f1', 'weighted_f1',
          'negative_precision', 'negative_recall', 'negative_f1',
          'neutral_precision', 'neutral_recall', 'neutral_f1',
          'positive_precision', 'positive_recall', 'positive_f1',
          'mae', 'pearson', 'best_epoch', 'fusion_mode', 'pool_mode',
          'head_mode', 'use_position', 'class_weight_mode', 'label_smoothing',
          'dropout', 'weight_decay', 'batch_size', 'gpu_memory_mb',
          'training_seconds', 'train_accuracy_at_selection', 'selected_model_kind',
          'directory']


def read_row(directory):
    metrics_path = directory / 'metrics.json'
    config_path = directory / 'config.yaml'
    if not metrics_path.exists() or not config_path.exists():
        return None
    result = json.loads(metrics_path.read_text(encoding='utf-8'))
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    valid = result['valid']
    row = {field: '' for field in FIELDS}
    row.update({
        'run_name': result.get('run_name', directory.name),
        'accuracy': valid['accuracy'],
        'macro_f1': valid['f1_macro'],
        'weighted_f1': valid['f1_weighted'],
        'mae': valid['mae'],
        'pearson': valid['pearson'],
        'best_epoch': result.get('selected_epoch', ''),
        'gpu_memory_mb': result.get('gpu_memory_mb', ''),
        'training_seconds': result.get('training_seconds', ''),
        'train_accuracy_at_selection': result.get('train_accuracy_at_selection', ''),
        'selected_model_kind': result.get('selected_model_kind', 'raw'),
        'directory': str(directory.relative_to(ROOT)),
    })
    for class_result in result.get('class_metrics', []):
        name = class_result['class_name'].lower()
        for metric in ('precision', 'recall', 'f1'):
            row[f'{name}_{metric}'] = class_result[metric]
    for key, default in [('fusion_mode', 'concat'), ('pool_mode', 'attention'),
                         ('head_mode', 'linear'), ('use_position', False),
                         ('class_weight_mode', 'inverse'), ('label_smoothing', 0),
                         ('dropout', ''), ('weight_decay', ''), ('batch_size', '')]:
        row[key] = config.get(key, default)
    return row


def write_csv(path, rows, fields=FIELDS):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    ordered_names = [
        '01_position', '02_text_centered', '03_attention_max', '04_mlp_heads',
        '05_weight_inverse', '05_weight_sqrt', '05_weight_none',
        '05_label_smoothing_005', '06_regularized', '07_ema',
        '08_modality_T', '08_modality_TA', '08_modality_TV', '08_modality_TAV',
    ]
    found = {path.parent.name: path.parent for path in output.glob('*/metrics.json')}
    directories = [RUN_A] + [found[name] for name in ordered_names if name in found]
    directories.extend(found[name] for name in sorted(set(found) - set(ordered_names)))
    rows = [row for directory in directories if (row := read_row(directory)) is not None]
    write_csv(output / 'score_push_summary.csv', rows)
    write_csv(output / 'score_push_by_macro_f1.csv',
              sorted(rows, key=lambda row: float(row['macro_f1']), reverse=True))
    write_csv(output / 'score_push_by_accuracy.csv',
              sorted(rows, key=lambda row: float(row['accuracy']), reverse=True))
    weights = [row for row in rows if Path(row['directory']).name.startswith('05_weight_')]
    if weights:
        write_csv(output / 'class_weight_comparison.csv', weights)
    modalities = [row for row in rows if Path(row['directory']).name.startswith('08_modality_')]
    if modalities:
        write_csv(output / 'modality_ablation.csv', modalities)
    print(f'Summarized {len(rows)} validation runs')


if __name__ == '__main__':
    main()
