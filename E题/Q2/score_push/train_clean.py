"""Train and compare clean sentiment backbones using train/valid only."""

import argparse
import copy
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from diagnostics.oracle_text import preflight, to_device, write_csv
from models.strong_clean_backbone import FeatureProjection, StrongCleanBackbone
from utils.metrics import metrics
from utils.seed import set_seed


def classification_loss(logits, target, kind, weights):
    if kind == 'ce':
        return F.cross_entropy(logits, target)
    if kind == 'ce_smooth':
        return F.cross_entropy(logits, target, label_smoothing=0.05)
    if kind in ('weighted_ce_sqrt', 'weighted_ce_inverse'):
        return F.cross_entropy(logits, target, weight=weights[kind])
    if kind == 'focal':
        ce = F.cross_entropy(logits, target, reduction='none')
        alpha = weights['weighted_ce_sqrt'][target]
        return (alpha * (1 - torch.exp(-ce)).pow(1.5) * ce).mean()
    raise ValueError(f'Unknown classification loss: {kind}')


def ordinal_loss(logits, target, temperature=0.4):
    centers = logits.new_tensor([-0.7, 0.0, 0.7])
    soft_target = torch.softmax(-(target[:, None] / 3 - centers).abs() / temperature, dim=-1)
    return F.kl_div(F.log_softmax(logits, dim=-1), soft_target, reduction='batchmean')


def class_weights(train_data, device):
    counts = torch.bincount(torch.from_numpy(train_data.y_cls), minlength=3).float().to(device)
    inverse = len(train_data) / (3 * counts)
    sqrt_inverse = counts.rsqrt()
    sqrt_inverse = sqrt_inverse / sqrt_inverse.mean()
    return {'weighted_ce_inverse': inverse, 'weighted_ce_sqrt': sqrt_inverse}


def make_teacher_projection(cfg, device):
    if cfg.get('lambda_distill', 0) <= 0:
        return None
    path = cfg.get('teacher_checkpoint')
    if not path or not Path(path).exists():
        raise FileNotFoundError('Distillation requires an Oracle teacher_checkpoint')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    prefix = 'text_projection.'
    state = {key[len(prefix):]: value for key, value in checkpoint['state_dict'].items()
             if key.startswith(prefix)}
    teacher = FeatureProjection(768, cfg['d_model'], cfg['dropout']).to(device)
    teacher.load_state_dict(state)
    teacher.eval().requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for raw in loader:
        batch = to_device(raw, device)
        output = model(batch)
        probs = output['logits'].float().softmax(-1).cpu().numpy()
        reg = output['regression'].float().cpu().numpy()
        for i, sample_id in enumerate(raw['id']):
            rows.append({'id': str(sample_id), 'true_class': int(raw['y_cls'][i]),
                         'pred_class': int(probs[i].argmax()),
                         'p_negative': float(probs[i, 0]), 'p_neutral': float(probs[i, 1]),
                         'p_positive': float(probs[i, 2]),
                         'true_reg': float(raw['y_reg'][i]), 'pred_reg': float(reg[i])})
    score = metrics([r['true_class'] for r in rows], [r['pred_class'] for r in rows],
                    [r['true_reg'] for r in rows], [r['pred_reg'] for r in rows])
    return score, rows


def score_cls(score):
    return 0.55 * score['f1_macro'] + 0.45 * score['accuracy']


def update_ema(ema, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.lerp_(param, 1 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def write_final_artifacts(output_dir, score, rows, cfg, epoch, model_kind, best_epochs,
                          trainable_params, gpu_memory_mb):
    write_csv(output_dir / 'valid_predictions.csv', rows)
    true = [r['true_class'] for r in rows]
    pred = [r['pred_class'] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=[0, 1, 2], zero_division=0)
    class_rows = [{'class_id': i, 'class_name': name, 'precision': float(precision[i]),
                   'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
                  for i, name in enumerate(('Negative', 'Neutral', 'Positive'))]
    write_csv(output_dir / 'class_metrics.csv', class_rows)
    matrix = confusion_matrix(true, pred, labels=[0, 1, 2])
    write_csv(output_dir / 'confusion_matrix.csv',
              [{'true_class': name, 'pred_negative': int(matrix[i, 0]),
                'pred_neutral': int(matrix[i, 1]), 'pred_positive': int(matrix[i, 2])}
               for i, name in enumerate(('Negative', 'Neutral', 'Positive'))])
    result = {'run_name': cfg['run_name'], 'selected_by': '0.55_macro_f1+0.45_accuracy',
              'selected_epoch': epoch, 'model_kind': model_kind, 'valid': score,
              'class_metrics': class_rows, 'best_epochs': best_epochs,
              'trainable_params': trainable_params, 'gpu_memory_mb': gpu_memory_mb,
              'test_evaluated': False}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('selected valid metrics:', result, flush=True)


def train(cfg):
    set_seed(cfg['seed'])
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                            encoding='utf-8')
    with Path(cfg['aligned_pkl']).open('rb') as handle:
        source = pickle.load(handle)
    train_data = AlignedMoseiDataset(split='train', source=source)
    valid_data = AlignedMoseiDataset(split='valid', source=source)
    del source
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = cfg['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=device.type == 'cuda')
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0,
                              pin_memory=device.type == 'cuda')
    first = next(iter(train_loader))
    preflight(first)
    if cfg.get('text_mode') == 'pretrained_bert':
        tokens = train_data.text_bert
        if tokens[:, 0].min() < 0 or tokens[:, 0].max() >= 30522:
            raise ValueError('Token ids are outside bert-base-uncased vocabulary')
        if not np.isin(tokens[:, 1:], [0, 1]).all():
            raise ValueError('BERT attention or token type mask is not binary')
    model = StrongCleanBackbone(cfg).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable_params', trainable_params, flush=True)
    teacher_projection = make_teacher_projection(cfg, device)
    weights = class_weights(train_data, device)
    optimizer_groups = []
    if cfg.get('text_mode') == 'pretrained_bert':
        bert_parameters = [p for p in model.bert.parameters() if p.requires_grad]
        downstream_parameters = [p for name, p in model.named_parameters()
                                 if p.requires_grad and not name.startswith('bert.')]
        optimizer_groups = [{'params': bert_parameters, 'lr': cfg.get('bert_lr', 1e-5)},
                            {'params': downstream_parameters, 'lr': cfg['lr']}]
    else:
        optimizer_groups = [{'params': [p for p in model.parameters() if p.requires_grad],
                             'lr': cfg['lr']}]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=cfg['weight_decay'])
    total_steps = cfg['epochs'] * len(train_loader)
    warmup = max(1, int(total_steps * cfg['warmup_ratio']))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    amp = bool(cfg.get('amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ema_decay = cfg.get('ema_decay', 0)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None
    best = {'accuracy': -1, 'macro_f1': -1, 'joint': -1, 'regression': -float('inf')}
    best_epochs = {}
    best_joint_model_kind = 'raw'
    patience = 0
    history = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_seen = 0
        for raw in train_loader:
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                prediction = model(batch)
                loss = classification_loss(prediction['logits'], batch['y_cls'],
                                           cfg['class_loss'], weights)
                loss = loss + cfg['lambda_reg'] * F.smooth_l1_loss(
                    prediction['regression'], batch['y_reg'], beta=0.5)
                if cfg.get('lambda_ordinal', 0) > 0:
                    loss = loss + cfg['lambda_ordinal'] * ordinal_loss(prediction['logits'], batch['y_reg'])
                if teacher_projection is not None:
                    with torch.no_grad():
                        teacher = teacher_projection(batch['text_teacher'])
                    observed = prediction['valid_mask']
                    loss = loss + cfg['lambda_distill'] * F.smooth_l1_loss(
                        prediction['text_projected'][observed], teacher[observed], beta=1.0)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
                if ema is not None:
                    update_ema(ema, model, ema_decay)
            running_loss += float(loss.detach()) * len(raw['id'])
            train_correct += int((prediction['logits'].argmax(-1) == batch['y_cls']).sum())
            train_seen += len(raw['id'])
        candidates = {'raw': model}
        if ema is not None:
            candidates['ema'] = ema
        epoch_best_joint = -float('inf')
        for kind, candidate in candidates.items():
            score, _ = evaluate(candidate, valid_loader, device)
            cls = score_cls(score)
            reg = score['pearson'] - 0.1 * score['mae'] / 3
            row = {'epoch': epoch + 1, 'model_kind': kind,
                   'train_loss': running_loss / train_seen,
                   'train_accuracy': train_correct / train_seen,
                   **score, 'classification_score': cls, 'regression_score': reg}
            history.append(row)
            print(f"{cfg['run_name']} epoch {epoch + 1} {kind}: "
                  f"train_acc={row['train_accuracy']:.4f} valid_acc={score['accuracy']:.4f} "
                  f"valid_f1={score['f1_macro']:.4f} mae={score['mae']:.4f} "
                  f"pearson={score['pearson']:.4f}", flush=True)
            epoch_best_joint = max(epoch_best_joint, cls)
            measures = {'accuracy': score['accuracy'], 'macro_f1': score['f1_macro'],
                        'joint': cls, 'regression': reg}
            for name, value in measures.items():
                if value > best[name]:
                    best[name] = value
                    best_epochs[name] = {'epoch': epoch + 1, 'model_kind': kind}
                    torch.save({'state_dict': candidate.state_dict(), 'config': cfg,
                                'epoch': epoch + 1, 'model_kind': kind},
                               output_dir / f'best_{name}.pt')
                    if name == 'joint':
                        best_joint_model_kind = kind
        write_csv(output_dir / 'train_history.csv', history)
        patience = 0 if epoch_best_joint >= best['joint'] - 1e-12 else patience + 1
        if patience >= cfg['patience']:
            break
    checkpoint = torch.load(output_dir / 'best_joint.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    score, rows = evaluate(model, valid_loader, device)
    gpu_memory_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                     if device.type == 'cuda' else 0.0)
    write_final_artifacts(output_dir, score, rows, cfg, checkpoint['epoch'],
                          best_joint_model_kind, best_epochs, trainable_params, gpu_memory_mb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
