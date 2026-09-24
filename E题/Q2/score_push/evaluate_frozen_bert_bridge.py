"""Compare official text features with a frozen BERT using the same Run A head."""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from diagnostics.oracle_text import OracleTextBaseline, evaluate_valid, write_csv


def main():
    root = Path('outputs/q2_score_push/deployable_bert/frozen_bridge')
    root.mkdir(parents=True, exist_ok=True)
    oracle_path = Path('outputs/q2_v2/01_run_a_oracle_text/best_macro_f1.pt')
    checkpoint = torch.load(oracle_path, map_location='cpu', weights_only=False)
    cfg = dict(checkpoint['config'])
    cfg['text_mode'] = 'pretrained_bert'
    cfg['bert_model_path'] = '/home/hanjinwei/math/model_cache/bert-base-uncased'
    cfg['unfreeze_last_n_layers'] = 0
    cfg['run_name'] = 'Deployable_BERT_Frozen_OracleHead'
    cfg['output_dir'] = str(root)
    (root / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                      encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OracleTextBaseline(cfg).to(device)
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if unexpected or any(not key.startswith('bert.') for key in missing):
        raise ValueError(f'Unexpected checkpoint mismatch: missing={missing}, extra={unexpected}')
    with Path(cfg['aligned_pkl']).open('rb') as handle:
        source = pickle.load(handle)
    valid = AlignedMoseiDataset(split='valid', source=source)
    loader = DataLoader(valid, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    score, rows = evaluate_valid(model, loader, device)
    write_csv(root / 'valid_predictions.csv', rows)
    old = pd.read_csv('outputs/q2_v2/01_run_a_oracle_text/valid_predictions.csv')
    new = pd.DataFrame(rows)
    if old['id'].astype(str).tolist() != new['id'].astype(str).tolist():
        raise ValueError('Oracle and BERT validation sample order differs')
    columns = ['p_negative', 'p_neutral', 'p_positive', 'pred_reg']
    max_abs_prediction_difference = float(np.max(np.abs(old[columns].to_numpy() -
                                                        new[columns].to_numpy())))
    p, r, f, n = precision_recall_fscore_support(new.true_class, new.pred_class,
                                                 labels=[0, 1, 2], zero_division=0)
    classes = [{'class_id': i, 'precision': float(p[i]), 'recall': float(r[i]),
                'f1': float(f[i]), 'support': int(n[i])} for i in range(3)]
    write_csv(root / 'class_metrics.csv', classes)
    result = {'run_name': cfg['run_name'], 'valid': score,
              'max_abs_prediction_difference_from_oracle': max_abs_prediction_difference,
              'oracle_checkpoint': str(oracle_path), 'bert_model_path': cfg['bert_model_path'],
              'class_metrics': classes, 'test_evaluated': False}
    (root / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
