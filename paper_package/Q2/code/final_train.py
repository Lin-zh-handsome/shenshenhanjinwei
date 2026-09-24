"""Validation-only trainer for the final Q2 clean-head experiments."""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from diagnostics.oracle_text import preflight, to_device, write_csv
from models.clean_backbone import CleanMultimodalBackbone
from models.hierarchical_sentiment_head import hierarchical_classification_loss
from utils.metrics import metrics
from utils.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for raw in loader:
        batch = to_device(raw, device)
        output = model(batch, return_outputs=True)
        probabilities = output['cls_prob'].float().cpu().numpy()
        regression = output['reg_pred'].float().cpu().numpy()
        for index, sample_id in enumerate(raw['id']):
            rows.append({'id': str(sample_id), 'true_class': int(raw['y_cls'][index]),
                         'pred_class': int(probabilities[index].argmax()),
                         'p_negative': float(probabilities[index, 0]),
                         'p_neutral': float(probabilities[index, 1]),
                         'p_positive': float(probabilities[index, 2]),
                         'true_reg': float(raw['y_reg'][index]),
                         'pred_reg': float(regression[index])})
    score = metrics([row['true_class'] for row in rows], [row['pred_class'] for row in rows],
                    [row['true_reg'] for row in rows], [row['pred_reg'] for row in rows])
    return score, rows


def class_and_confusion(rows):
    true = [row['true_class'] for row in rows]
    predicted = [row['pred_class'] for row in rows]
    p, r, f, n = precision_recall_fscore_support(true, predicted, labels=[0, 1, 2], zero_division=0)
    classes = [{'class_id': index, 'class_name': name, 'precision': float(p[index]),
                'recall': float(r[index]), 'f1': float(f[index]), 'support': int(n[index])}
               for index, name in enumerate(('Negative', 'Neutral', 'Positive'))]
    matrix = confusion_matrix(true, predicted, labels=[0, 1, 2])
    confusion = [{'true_class': name, 'pred_negative': int(matrix[index, 0]),
                  'pred_neutral': int(matrix[index, 1]), 'pred_positive': int(matrix[index, 2])}
                 for index, name in enumerate(('Negative', 'Neutral', 'Positive'))]
    return classes, confusion


def classification_score(score):
    return 0.55 * score['f1_macro'] + 0.45 * score['accuracy']


def load_initial(model, cfg):
    shared_path = cfg.get('init_shared_checkpoint')
    full_path = cfg.get('init_full_checkpoint')
    if shared_path and full_path:
        raise ValueError('Select either shared or full initialization')
    path = shared_path or full_path
    if not path:
        return
    state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    if shared_path:
        excluded = ('cls_head.', 'reg_head.', 'hierarchical_head.', 'bert.')
        if cfg.get('init_skip_fusion', False):
            excluded += ('fusion.', 'text_centered_fusion.')
        state = {key: value for key, value in state.items()
                 if not key.startswith(excluded)}
    else:
        state = {key: value for key, value in state.items() if not key.startswith('bert.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f'Unexpected checkpoint keys: {unexpected}')
    allowed = (('bert.', 'cls_head.', 'reg_head.', 'hierarchical_head.', 'fusion.',
                'text_centered_fusion.') if cfg.get('init_skip_fusion', False)
               else ('bert.', 'cls_head.', 'reg_head.', 'hierarchical_head.')) if shared_path else ('bert.',)
    if any(not key.startswith(allowed) for key in missing):
        raise ValueError(f'Uninitialized shared backbone keys: {missing}')


def train(cfg):
    set_seed(cfg['seed'])
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                           encoding='utf-8')
    with Path(cfg['aligned_pkl']).open('rb') as handle:
        source = pickle.load(handle)
    train_data = AlignedMoseiDataset(split='train', source=source)
    valid_data = AlignedMoseiDataset(split='valid', source=source)
    del source
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CleanMultimodalBackbone(cfg['model']).to(device)
    load_initial(model, cfg)
    data_generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,
                        generator=data_generator, num_workers=0, pin_memory=device.type == 'cuda')
    valid_loader = DataLoader(valid_data, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=0, pin_memory=device.type == 'cuda')
    preflight(next(iter(loader)))
    counts = torch.bincount(torch.from_numpy(train_data.y_cls), minlength=3).float()
    class_weights = (len(train_data) / (3 * counts)).to(device)
    neutral_mode = cfg.get('neutral_pos_weight_mode', 'none')
    if neutral_mode not in {'none', 'sqrt'}:
        raise ValueError(f'Unknown neutral_pos_weight_mode: {neutral_mode}')
    neutral_weight = (torch.sqrt((counts[0] + counts[2]) / counts[1]).to(device)
                      if neutral_mode == 'sqrt' else None)
    bert_params = [parameter for parameter in model.bert.parameters() if parameter.requires_grad]
    downstream = [parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad and not name.startswith('bert.')]
    groups = [{'params': downstream, 'lr': cfg['lr']}]
    if bert_params:
        groups.append({'params': bert_params, 'lr': cfg.get('bert_lr', 1e-5)})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    total_steps = cfg['epochs'] * len(loader)
    warmup = max(1, int(total_steps * cfg['warmup_ratio']))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    amp = bool(cfg.get('amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    best = {'accuracy': -float('inf'), 'macro_f1': -float('inf'),
            'mae': float('inf'), 'pearson': -float('inf'),
            'classification': -float('inf')}
    best_epoch = {}
    history = []
    patience = 0

    def validate_and_save(epoch, train_loss=None):
        nonlocal patience
        score, _ = evaluate(model, valid_loader, device)
        candidates = {'accuracy': score['accuracy'], 'macro_f1': score['f1_macro'],
                      'mae': score['mae'], 'pearson': score['pearson'],
                      'classification': classification_score(score)}
        row = {'epoch': epoch, 'train_loss': train_loss if train_loss is not None else '',
               **score, 'classification_composite': candidates['classification']}
        history.append(row)
        write_csv(output_dir / 'train_history.csv', history)
        improved_classification = False
        for name, value in candidates.items():
            improved = value < best[name] if name == 'mae' else value > best[name]
            if improved:
                best[name] = value
                best_epoch[name] = epoch
                torch.save({'state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch},
                           output_dir / f'best_{name}.pt')
                if name == 'classification':
                    improved_classification = True
        patience = 0 if improved_classification else patience + 1
        print(f"{cfg['run_name']} epoch {epoch}: valid_acc={score['accuracy']:.4f} "
              f"valid_macro_f1={score['f1_macro']:.4f} valid_mae={score['mae']:.4f}", flush=True)

    # An initialized checkpoint is a real candidate, before any optimizer step.
    validate_and_save(0)
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        losses = []
        for raw in loader:
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                output = model(batch, return_outputs=True)
                if cfg['model'].get('head_mode', 'flat') == 'hierarchical':
                    cls_loss = hierarchical_classification_loss(
                        output, batch['y_cls'], neutral_weight,
                        cfg.get('lambda_neutral', 1.0), cfg.get('lambda_polarity', 1.0))
                else:
                    cls_loss = F.cross_entropy(output['cls_logits'], batch['y_cls'], weight=class_weights)
                loss = cls_loss + cfg['lambda_reg'] * F.smooth_l1_loss(
                    output['reg_pred'], batch['y_reg'], beta=0.5)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            losses.append(float(loss.detach()))
        validate_and_save(epoch, float(np.mean(losses)))
        if patience >= cfg['patience']:
            break

    checkpoint = torch.load(output_dir / 'best_classification.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    score, rows = evaluate(model, valid_loader, device)
    classes, confusion = class_and_confusion(rows)
    write_csv(output_dir / 'valid_predictions.csv', rows)
    write_csv(output_dir / 'class_metrics.csv', classes)
    write_csv(output_dir / 'confusion_matrix.csv', confusion)
    result = {'run_name': cfg['run_name'], 'selected_by': 'validation_classification_composite',
              'selected_epoch': best_epoch['classification'], 'best_epochs': best_epoch,
              'valid': score, 'class_metrics': classes, 'confusion_matrix': confusion,
              'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'gpu_memory_mb': (torch.cuda.max_memory_allocated(device) / 2**20
                                if device.type == 'cuda' else 0),
              'test_evaluated': False}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('selected valid metrics:', result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train(cfg)


if __name__ == '__main__':
    main()
