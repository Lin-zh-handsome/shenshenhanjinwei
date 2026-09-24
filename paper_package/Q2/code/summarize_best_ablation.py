"""Summarize validation-only missing-module ablations of the selected BERT model."""

import argparse
import csv
import json
from pathlib import Path

from diagnostics.oracle_text import write_csv
from final_train import class_and_confusion


RUNS = (
    'R0_selected_clean', 'R1_missing_aug', 'R2_span_geometry',
    'R3_reconstruction', 'R4_reliability', 'R5_consistency',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', default='outputs/q2_best_ablation')
    args = parser.parse_args()
    root = Path(args.outputs_dir)
    rows = []
    for name in RUNS:
        source = root / name / 'metrics.json'
        result = json.loads(source.read_text(encoding='utf-8'))
        with (root / name / 'valid_missing_predictions.csv').open(newline='', encoding='utf-8') as handle:
            missing_rows = list(csv.DictReader(handle))
        for prediction in missing_rows:
            prediction['true_class'] = int(prediction['true_class'])
            prediction['pred_class'] = int(prediction['pred_class'])
        missing_classes, missing_confusion = class_and_confusion(missing_rows)
        write_csv(root / name / 'missing_class_metrics.csv', missing_classes)
        write_csv(root / name / 'missing_confusion_matrix.csv', missing_confusion)
        clean, missing = result['clean_valid'], result['missing_valid']
        class_metrics = {item['class_name']: item for item in result['class_metrics']}
        clean_cls = .55 * clean['f1_macro'] + .45 * clean['accuracy']
        missing_cls = .55 * missing['f1_macro'] + .45 * missing['accuracy']
        rows.append({
            'experiment': name, 'selected_epoch': result['selected_epoch'],
            'clean_accuracy': clean['accuracy'], 'clean_macro_f1': clean['f1_macro'],
            'clean_weighted_f1': clean['f1_weighted'],
            'clean_neutral_f1': class_metrics['Neutral']['f1'],
            'clean_mae': clean['mae'], 'clean_pearson': clean['pearson'],
            'missing_accuracy': missing['accuracy'],
            'missing_macro_f1': missing['f1_macro'],
            'missing_weighted_f1': missing['f1_weighted'],
            'missing_mae': missing['mae'], 'missing_pearson': missing['pearson'],
            'robust_drop_f1': clean['f1_macro'] - missing['f1_macro'],
            'robust_drop_pearson': clean['pearson'] - missing['pearson'],
            'balanced_classification': .5 * (clean_cls + missing_cls),
            'test_evaluated': False,
        })
    output = root / 'ablation_table.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row['balanced_classification'])
    print(f'Wrote {output}; selected {best["experiment"]} by validation balanced classification')


if __name__ == '__main__':
    main()
