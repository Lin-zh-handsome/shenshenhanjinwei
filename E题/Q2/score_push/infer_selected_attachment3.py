"""Run the validation-selected BERT checkpoint on unlabeled Attachment 3 aligned files."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.attachment3_dataset import Attachment3AlignedDataset
from diagnostics.oracle_text import OracleTextBaseline


CHECKPOINT = Path('outputs/q2_score_push/deployable_bert/partial_last4/best_joint.pt')
ATTACHMENT3_DIR = Path('/home/hanjinwei/math/data/E题数据/附件3-模态缺失特征样本/对齐版本')
OUTPUT = CHECKPOINT.parent
LABELS = ('Negative', 'Neutral', 'Positive')


@torch.inference_mode()
def main():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    if cfg.get('text_mode') != 'pretrained_bert' or cfg.get('run_name') != 'Deployable_BERT_Last4':
        raise ValueError('The selected checkpoint is not Deployable_BERT_Last4')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OracleTextBaseline(cfg).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    dataset = Attachment3AlignedDataset(ATTACHMENT3_DIR)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    rows = []
    for raw in loader:
        tokens = raw['text_bert']
        if tokens[:, 0].min() < 0 or tokens[:, 0].max() >= model.bert.config.vocab_size:
            raise ValueError('Attachment 3 token ids are outside the BERT vocabulary')
        if not torch.isin(tokens[:, 1:], torch.tensor([0, 1])).all():
            raise ValueError('Attachment 3 attention mask or token type ids are not binary')
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in raw.items()}
        logits, regression = model(batch)
        probabilities = logits.float().softmax(-1).cpu().numpy()
        regression = regression.float().cpu().numpy()
        for index, sample_id in enumerate(raw['sample_id']):
            class_id = int(np.argmax(probabilities[index]))
            rows.append({
                'sample_id': sample_id,
                'pred_class_id': class_id,
                'pred_label': LABELS[class_id],
                'prob_negative': float(probabilities[index, 0]),
                'prob_neutral': float(probabilities[index, 1]),
                'prob_positive': float(probabilities[index, 2]),
                'pred_intensity': float(regression[index]),
            })
    if len(rows) != len(dataset):
        raise RuntimeError(f'Expected {len(dataset)} predictions, got {len(rows)}')
    path = OUTPUT / 'attachment3_predictions.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'source': str(ATTACHMENT3_DIR),
        'checkpoint': str(CHECKPOINT),
        'sample_count': len(rows),
        'predicted_class_counts': dict(Counter(row['pred_label'] for row in rows)),
        'ground_truth_available': False,
        'metrics_computed': False,
        'predictions_file': str(path),
    }
    (OUTPUT / 'attachment3_inference.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
