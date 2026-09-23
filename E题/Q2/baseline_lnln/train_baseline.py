import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.aligned_dataset import load_aligned
from data.mask_utils import infer_shared_valid_mask
from data.missing_simulator import ContinuousSpanMasker
from baseline_lnln.common import (BaselineDataset, read_config, official_components, set_seed,
                                  original_random_masks, span_training_masks, make_inputs,
                                  training_labels, to_device, predict_regression, best_neutral_width,
                                  with_classes, regression_metrics, save_rows)


def train(cfg):
    set_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_model, loss_class, scheduler_factory, official_cfg = official_components(cfg)
    source = load_aligned(cfg['paths']['aligned_pkl'])
    train_ds = BaselineDataset(source, 'train')
    valid_ds = BaselineDataset(source, 'valid')
    del source
    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(official_cfg).to(device)
    loss_fn = loss_class(official_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=official_cfg['base']['lr'],
                                  weight_decay=official_cfg['base']['weight_decay'])
    scheduler = scheduler_factory(optimizer, official_cfg)
    q2_cfg = read_config(ROOT/'configs'/'q2_aligned.yaml')
    masker = ContinuousSpanMasker(q2_cfg['missing_aug'], cfg['training']['seed'])
    out = ROOT/cfg['paths']['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    best_mae, bad_epochs = float('inf'), 0
    history = []
    for epoch in range(1, cfg['training']['epochs']+1):
        model.train()
        losses = []
        for raw in train_loader:
            batch = to_device(raw, device)
            valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
            if cfg['training']['train_mask'] == 'official_random':
                masks, rates = original_random_masks(valid, batch['text_bert'])
            elif cfg['training']['train_mask'] == 'q2_span':
                masks, rates = span_training_masks(valid, masker)
            else:
                raise ValueError(cfg['training']['train_mask'])
            complete, incomplete = make_inputs(batch, masks)
            outputs = model(complete, incomplete)
            loss = loss_fn(outputs, training_labels(batch, rates))['loss']
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        valid_rows = predict_regression(model, valid_loader, device, missing=True, seed=101)
        y_true = np.array([r['true_reg'] for r in valid_rows])
        y_pred = np.array([r['pred_reg'] for r in valid_rows])
        mae = float(np.mean(np.abs(y_true-y_pred)))
        corr = float(np.corrcoef(y_true, y_pred)[0,1]) if np.std(y_pred) > 1e-12 else 0.0
        history.append({'epoch': epoch, 'train_loss': sum(losses)/len(losses),
                        'valid_missing30_mae': mae, 'valid_missing30_pearson': corr})
        with open(out/'train_history.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f'LNLN {cfg["training"]["train_mask"]} epoch={epoch} loss={history[-1]["train_loss"]:.5f} valid_span_mae={mae:.5f} corr={corr:.5f}', flush=True)
        if mae < best_mae:
            best_mae = mae
            bad_epochs = 0
            width = best_neutral_width([r['true_class'] for r in valid_rows], y_pred)
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'neutral_halfwidth': width,
                        'official_config': official_cfg, 'adaptation_config': cfg,
                        'valid_missing30_mae': mae}, out/'best_valid_mae.pt')
            scored = with_classes(valid_rows, width)
            save_rows(out/'valid_missing30_predictions.csv', scored)
            (out/'valid_metrics.json').write_text(json.dumps(regression_metrics(scored), indent=2), encoding='utf-8')
        else:
            bad_epochs += 1
        if bad_epochs >= cfg['training']['early_stop_patience']:
            break
    print('LNLN training finished; best_valid_mae', best_mae, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline_lnln/config.yaml')
    args = parser.parse_args()
    train(read_config(args.config))


if __name__ == '__main__':
    main()
