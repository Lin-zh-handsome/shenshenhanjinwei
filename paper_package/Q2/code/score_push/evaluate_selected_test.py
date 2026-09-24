"""One final test evaluation of the checkpoint selected on validation."""

import json
import pickle
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from diagnostics.oracle_text import OracleTextBaseline, evaluate_valid, preflight, write_csv


CHECKPOINT = Path('outputs/q2_score_push/deployable_bert/partial_last4/best_joint.pt')
OUTPUT = CHECKPOINT.parent


def main():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    if cfg.get('text_mode') != 'pretrained_bert' or cfg.get('run_name') != 'Deployable_BERT_Last4':
        raise ValueError('The selected checkpoint is not Deployable_BERT_Last4')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OracleTextBaseline(cfg).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    with Path(cfg['aligned_pkl']).open('rb') as handle:
        source = pickle.load(handle)
    test_data = AlignedMoseiDataset(split='test', source=source)
    loader = DataLoader(test_data, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=0, pin_memory=device.type == 'cuda')
    preflight(next(iter(loader)))
    score, rows = evaluate_valid(model, loader, device)
    true = [row['true_class'] for row in rows]
    predicted = [row['pred_class'] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=[0, 1, 2], zero_division=0)
    class_rows = [
        {'class_id': index, 'class_name': name, 'precision': float(precision[index]),
         'recall': float(recall[index]), 'f1': float(f1[index]), 'support': int(support[index])}
        for index, name in enumerate(('Negative', 'Neutral', 'Positive'))
    ]
    matrix = confusion_matrix(true, predicted, labels=[0, 1, 2])
    confusion_rows = [
        {'true_class': name, 'pred_negative': int(matrix[index, 0]),
         'pred_neutral': int(matrix[index, 1]), 'pred_positive': int(matrix[index, 2])}
        for index, name in enumerate(('Negative', 'Neutral', 'Positive'))
    ]
    write_csv(OUTPUT / 'test_predictions.csv', rows)
    write_csv(OUTPUT / 'test_class_metrics.csv', class_rows)
    write_csv(OUTPUT / 'test_confusion_matrix.csv', confusion_rows)
    result = {
        'model': cfg['run_name'],
        'selected_on': 'validation_joint',
        'selected_epoch': checkpoint['epoch'],
        'checkpoint': str(CHECKPOINT),
        'test_sample_count': len(test_data),
        'test': score,
        'class_metrics': class_rows,
        'confusion_matrix': confusion_rows,
        'test_evaluated': True,
    }
    (OUTPUT / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
