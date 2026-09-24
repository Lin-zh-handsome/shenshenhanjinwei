"""Clean Oracle Text diagnostic. This command never evaluates the test split."""

import argparse
import csv
import math
import pickle
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from models.clean_backbone import CleanMultimodalBackbone
from utils.metrics import metrics
from utils.seed import set_seed


class OracleTextBaseline(CleanMultimodalBackbone):
    """Compatibility name; the default config retains the 5790875 state dict."""


def to_device(raw, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in raw.items()}


def preflight(batch):
    expected = {'text_bert': (3, 50), 'text_teacher': (50, 768),
                'audio': (50, 74), 'vision': (50, 35)}
    for key, shape in expected.items():
        value = batch[key]
        if tuple(value.shape[1:]) != shape:
            raise ValueError(f'{key} shape {tuple(value.shape)} != [B,{shape}]')
        if key == 'text_bert' and value.dtype != torch.int64:
            raise TypeError(f'{key} must be int64, got {value.dtype}')
        if key != 'text_bert' and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f'{key} must be finite float32')
    if not torch.isin(batch['y_cls'], torch.tensor([0, 1, 2])).all():
        raise ValueError('Class targets must be 0, 1, 2')


@torch.no_grad()
def evaluate_valid(model, loader, device):
    model.eval()
    rows = []
    for raw in loader:
        batch = to_device(raw, device)
        logits, reg = model(batch)
        probs = logits.float().softmax(-1).cpu().numpy()
        pred_reg = reg.float().cpu().numpy()
        for i, sample_id in enumerate(raw['id']):
            rows.append({'id': str(sample_id), 'true_class': int(raw['y_cls'][i]),
                         'pred_class': int(probs[i].argmax()),
                         'p_negative': float(probs[i, 0]), 'p_neutral': float(probs[i, 1]),
                         'p_positive': float(probs[i, 2]),
                         'true_reg': float(raw['y_reg'][i]), 'pred_reg': float(pred_reg[i])})
    score = metrics([r['true_class'] for r in rows], [r['pred_class'] for r in rows],
                    [r['true_reg'] for r in rows], [r['pred_reg'] for r in rows])
    return score, rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_class_weights(labels, mode, device):
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=3).float()
    if mode == 'none':
        return None
    if mode == 'inverse':
        weights = counts.sum() / (len(counts) * counts)
    elif mode == 'sqrt':
        weights = torch.sqrt(counts.mean() / counts)
    else:
        raise ValueError(f'Unknown class_weight_mode: {mode}')
    return (weights / weights.mean()).to(device)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    source = dict(model.named_parameters())
    for name, ema_param in ema_model.named_parameters():
        if source[name].requires_grad:
            ema_param.lerp_(source[name], 1.0 - decay)
    for ema_buffer, source_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(source_buffer)


def train(cfg):
    set_seed(cfg['seed'])
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                        encoding='utf-8')
    with Path(cfg['aligned_pkl']).open('rb') as handle:
        source = pickle.load(handle)
    train_data = AlignedMoseiDataset(split='train', source=source)
    valid_data = AlignedMoseiDataset(split='valid', source=source)
    del source
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workers = cfg.get('num_workers', 0)
    loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=workers, pin_memory=device.type == 'cuda',
                        persistent_workers=workers > 0)
    valid_loader = DataLoader(valid_data, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=workers, pin_memory=device.type == 'cuda',
                              persistent_workers=workers > 0)
    preflight(next(iter(loader)))
    model = OracleTextBaseline(cfg).to(device)
    class_weights = build_class_weights(train_data.y_cls,
                                        cfg.get('class_weight_mode', 'inverse'), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    total_steps = cfg['epochs'] * len(loader)
    warmup = max(1, int(total_steps * cfg['warmup_ratio']))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    amp = bool(cfg['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ema_cfg = cfg.get('ema', {})
    ema_model = copy.deepcopy(model).eval().requires_grad_(False) if ema_cfg.get('enabled', False) else None
    best = {'accuracy': -float('inf'), 'macro_f1': -float('inf'),
            'mae': float('inf'), 'joint': -float('inf')}
    best_epoch = {}
    best_kind = {}
    history = []
    patience = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg['epochs']):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        correct = 0
        seen = 0
        for raw in loader:
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits, pred_reg = model(batch)
                loss = F.cross_entropy(logits, batch['y_cls'], weight=class_weights,
                                       label_smoothing=cfg.get('label_smoothing', 0.0))
                loss = loss + cfg['lambda_reg'] * F.smooth_l1_loss(pred_reg, batch['y_reg'], beta=0.5)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
                if ema_model is not None:
                    update_ema(ema_model, model, ema_cfg.get('decay', 0.999))
            losses.append(float(loss.detach()))
            correct += int((logits.detach().argmax(-1) == batch['y_cls']).sum())
            seen += int(batch['y_cls'].numel())
        score, _ = evaluate_valid(model, valid_loader, device)
        candidates_by_kind = {'raw': score}
        if ema_model is not None:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
            try:
                candidates_by_kind['ema'] = evaluate_valid(ema_model, valid_loader, device)[0]
            finally:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
               'train_accuracy': correct / seen, **score,
               'joint': score['f1_macro'] + 0.25 * score['pearson'] - 0.10 * score['mae'] / 3,
               'epoch_seconds': time.perf_counter() - epoch_start}
        if ema_model is not None:
            for metric_name, value in score.items():
                row[f'raw_{metric_name}'] = value
            for metric_name, value in candidates_by_kind['ema'].items():
                row[f'ema_{metric_name}'] = value
        history.append(row)
        write_csv(output / 'train_history.csv', history)
        print(f"{cfg['run_name']} epoch {epoch + 1}: loss={row['train_loss']:.4f} "
              f"valid_acc={score['accuracy']:.4f} valid_macro_f1={score['f1_macro']:.4f} "
              f"valid_mae={score['mae']:.4f}", flush=True)
        improved_f1 = False
        for kind, candidate_score in candidates_by_kind.items():
            joint = (candidate_score['f1_macro'] + 0.25 * candidate_score['pearson']
                     - 0.10 * candidate_score['mae'] / 3)
            candidates = {'accuracy': candidate_score['accuracy'],
                          'macro_f1': candidate_score['f1_macro'],
                          'mae': candidate_score['mae'], 'joint': joint}
            candidate_model = model if kind == 'raw' else ema_model
            for name, value in candidates.items():
                improved = value < best[name] if name == 'mae' else value > best[name]
                if improved:
                    best[name] = value
                    best_epoch[name] = epoch + 1
                    best_kind[name] = kind
                    torch.save({'state_dict': candidate_model.state_dict(), 'config': cfg,
                                'epoch': epoch + 1, 'model_kind': kind},
                               output / f'best_{name}.pt')
                    if name == 'macro_f1':
                        improved_f1 = True
        patience = 0 if improved_f1 else patience + 1
        if patience >= cfg['patience']:
            break

    checkpoint = torch.load(output / 'best_macro_f1.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    score, rows = evaluate_valid(model, valid_loader, device)
    write_csv(output / 'valid_predictions.csv', rows)
    true = [r['true_class'] for r in rows]
    pred = [r['pred_class'] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=[0, 1, 2], zero_division=0)
    class_rows = [{'class_id': i, 'class_name': name, 'precision': float(precision[i]),
                   'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
                  for i, name in enumerate(('Negative', 'Neutral', 'Positive'))]
    write_csv(output / 'class_metrics.csv', class_rows)
    matrix = confusion_matrix(true, pred, labels=[0, 1, 2])
    write_csv(output / 'confusion_matrix.csv',
              [{'true_class': name, **dict(zip(('pred_negative', 'pred_neutral', 'pred_positive'),
                                               [int(n) for n in matrix[i]]))}
               for i, name in enumerate(('Negative', 'Neutral', 'Positive'))])
    result = {'run_name': cfg['run_name'], 'selected_by': 'valid_macro_f1',
              'selected_epoch': best_epoch['macro_f1'], 'valid': score,
              'class_metrics': class_rows, 'best_epochs': best_epoch,
              'best_model_kind': best_kind, 'selected_model_kind': best_kind['macro_f1'],
              'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'gpu_memory_mb': (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                if device.type == 'cuda' else 0.0),
              'training_seconds': sum(row['epoch_seconds'] for row in history),
              'train_accuracy_at_selection': history[best_epoch['macro_f1'] - 1]['train_accuracy'],
              'test_evaluated': False}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('selected valid metrics:', result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train(cfg)


if __name__ == '__main__':
    main()
