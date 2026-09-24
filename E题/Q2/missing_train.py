"""Validation-only R0-R5 ablations on the selected BERT-backed Q2 model."""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.aligned_dataset import AlignedMoseiDataset
from data.mask_utils import apply_synthetic_missing
from data.missing_curriculum import CurriculumSpanMasker, deterministic_spans
from diagnostics.oracle_text import preflight, to_device, write_csv
from final_train import class_and_confusion, classification_score
from models.srf_msa import SRFMSA
from utils.metrics import metrics
from utils.seed import set_seed


COMBOS = ('T', 'A', 'V', 'TA', 'TV', 'AV')


def mixed_masks(valid, offset):
    masks = {modality: torch.zeros_like(valid) for modality in 'TAV'}
    for local in range(valid.shape[0]):
        combo = COMBOS[(offset + local) % len(COMBOS)]
        single = deterministic_spans(valid[local:local + 1], combo, ratio=0.3,
                                     position='middle', seed=2026 + offset + local)
        for modality in 'TAV':
            masks[modality][local] = single[modality][0]
    return masks


@torch.no_grad()
def evaluate(model, loader, device, missing=None):
    model.eval()
    rows = []
    offset = 0
    for raw in loader:
        batch = to_device(raw, device)
        if missing is not None:
            masks = (mixed_masks(batch['sequence_valid_mask'], offset) if missing == 'mixed'
                     else deterministic_spans(batch['sequence_valid_mask'], **missing))
            batch = apply_synthetic_missing(batch, masks)
        output = model(batch)
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
        offset += len(raw['id'])
    score = metrics([r['true_class'] for r in rows], [r['pred_class'] for r in rows],
                    [r['true_reg'] for r in rows], [r['pred_reg'] for r in rows])
    return score, rows


def initialize_from_selected_bert(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('config', {}).get('run_name') != 'Deployable_BERT_Last4':
        raise ValueError('The initializer is not the validation-selected BERT last4 model')
    model.backbone.load_state_dict(checkpoint['state_dict'], strict=True)


def subgroup_analysis(model, loader, device, output_dir):
    groups = {
        'missing_modality_analysis.csv': [('modality', combo, {'combo': combo, 'ratio': 0.3,
                                                               'position': 'middle'}) for combo in COMBOS],
        'missing_ratio_analysis.csv': [('ratio', str(ratio), {'combo': 'TAV', 'ratio': ratio,
                                                              'position': 'middle'}) for ratio in (.1, .2, .3, .4, .5)],
        'missing_position_analysis.csv': [('position', pos, {'combo': 'TAV', 'ratio': 0.3,
                                                              'position': pos}) for pos in ('early', 'middle', 'late')],
        'missing_span_length_analysis.csv': [('length', name, {'combo': 'TAV', 'duration': duration,
                                                                'position': 'middle'}) for name, duration in
                                             (('short', 4), ('medium', 10), ('long', 20))],
    }
    for filename, cases in groups.items():
        rows = []
        for dimension, label, kwargs in cases:
            score, _ = evaluate(model, loader, device, missing=kwargs)
            rows.append({dimension: label, 'accuracy': score['accuracy'],
                         'macro_f1': score['f1_macro'], 'mae': score['mae'],
                         'pearson': score['pearson']})
        write_csv(output_dir / filename, rows)


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
    model = SRFMSA(cfg['model']).to(device)
    initialize_from_selected_bert(model, cfg['init_checkpoint'])
    if cfg.get('freeze_clean_backbone', False):
        model.backbone.requires_grad_(False)
    loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg['seed']), num_workers=0,
                        pin_memory=device.type == 'cuda')
    valid_loader = DataLoader(valid_data, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=0, pin_memory=device.type == 'cuda')
    preflight(next(iter(loader)))
    counts = torch.bincount(torch.from_numpy(train_data.y_cls), minlength=3).float()
    class_weights = (len(train_data) / (3 * counts)).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps = max(1, cfg['epochs'] * len(loader))
    warmup = max(1, int(total_steps * cfg['warmup_ratio']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: ((step + 1) / warmup if step < warmup else
                                 0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                                     max(1, total_steps - warmup)))))
    amp = bool(cfg.get('amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    masker = CurriculumSpanMasker(cfg['seed'])
    best = {'accuracy': -float('inf'), 'macro_f1': -float('inf'),
            'mae': float('inf'), 'pearson': -float('inf'),
            'classification': -float('inf')}
    best_epochs = {}
    history = []
    patience = 0

    def validate(epoch, train_loss=None):
        nonlocal patience
        clean, _ = evaluate(model, valid_loader, device)
        missing, _ = evaluate(model, valid_loader, device, missing='mixed')
        # Predeclared balanced checkpoint rank, not an official competition score.
        balanced = 0.5 * (classification_score(clean) + classification_score(missing))
        candidates = {'accuracy': missing['accuracy'], 'macro_f1': missing['f1_macro'],
                      'mae': missing['mae'], 'pearson': missing['pearson'],
                      'classification': balanced}
        history.append({'epoch': epoch, 'train_loss': '' if train_loss is None else train_loss,
                        **{f'clean_{k}': v for k, v in clean.items()},
                        **{f'missing_{k}': v for k, v in missing.items()},
                        'balanced_classification': balanced})
        write_csv(output_dir / 'train_history.csv', history)
        improved = False
        for key, value in candidates.items():
            better = value < best[key] if key == 'mae' else value > best[key]
            if better:
                best[key] = value
                best_epochs[key] = epoch
                torch.save({'state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch},
                           output_dir / f'best_{key}.pt')
                if key == 'classification':
                    improved = True
        patience = 0 if improved else patience + 1
        print(f"{cfg['run_name']} epoch {epoch}: clean_acc={clean['accuracy']:.4f} "
              f"clean_f1={clean['f1_macro']:.4f} missing_acc={missing['accuracy']:.4f} "
              f"missing_f1={missing['f1_macro']:.4f} balanced={balanced:.4f}", flush=True)

    validate(0)
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        losses = []
        for raw in loader:
            clean_batch = to_device(raw, device)
            masks = masker.sample(clean_batch['sequence_valid_mask'], epoch)
            batch = apply_synthetic_missing(clean_batch, masks)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                output = model(batch, clean_batch=clean_batch if cfg['model'].get('use_reconstruction') else None)
                cls_loss = F.cross_entropy(output['cls_logits'], batch['y_cls'], weight=class_weights)
                reg_loss = F.smooth_l1_loss(output['reg_pred'], batch['y_reg'], beta=0.5)
                loss = (cls_loss + cfg['lambda_reg'] * reg_loss
                        + cfg.get('lambda_recon', 0.0) * output['reconstruction_loss'])
                if cfg.get('lambda_cons', 0.0) > 0:
                    affected = batch['synthetic_missing_mask'].any(dim=(1, 2))
                    if affected.any():
                        with torch.no_grad():
                            teacher = model(clean_batch)
                        masked_log_p = F.log_softmax(output['cls_logits'][affected].float(), dim=-1)
                        clean_p = teacher['cls_prob'][affected].float()
                        consistency = F.kl_div(masked_log_p, clean_p, reduction='batchmean')
                        consistency = consistency + F.smooth_l1_loss(
                            output['reg_pred'][affected].float(),
                            teacher['reg_pred'][affected].float())
                        loss = loss + cfg['lambda_cons'] * consistency
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at epoch {epoch}')
            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before:
                    scheduler.step()
            losses.append(float(loss.detach()))
        validate(epoch, float(np.mean(losses)))
        if epoch >= cfg.get('min_epochs', 14) and patience >= cfg['patience']:
            break
    selected = torch.load(output_dir / 'best_classification.pt', map_location=device,
                          weights_only=False)
    model.load_state_dict(selected['state_dict'], strict=True)
    clean, clean_rows = evaluate(model, valid_loader, device)
    missing, missing_rows = evaluate(model, valid_loader, device, missing='mixed')
    classes, confusion = class_and_confusion(clean_rows)
    missing_classes, missing_confusion = class_and_confusion(missing_rows)
    write_csv(output_dir / 'valid_predictions.csv', clean_rows)
    write_csv(output_dir / 'valid_missing_predictions.csv', missing_rows)
    write_csv(output_dir / 'class_metrics.csv', classes)
    write_csv(output_dir / 'confusion_matrix.csv', confusion)
    write_csv(output_dir / 'missing_class_metrics.csv', missing_classes)
    write_csv(output_dir / 'missing_confusion_matrix.csv', missing_confusion)
    subgroup_analysis(model, valid_loader, device, output_dir)
    result = {'run_name': cfg['run_name'], 'selected_epoch': best_epochs['classification'],
              'best_epochs': best_epochs, 'selected_by': 'validation_balanced_classification',
              'clean_valid': clean, 'missing_valid': missing, 'class_metrics': classes,
              'missing_class_metrics': missing_classes, 'confusion_matrix': confusion,
              'missing_confusion_matrix': missing_confusion, 'test_evaluated': False}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
