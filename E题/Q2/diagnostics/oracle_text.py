"""Clean Oracle Text diagnostic. This command never evaluates the test split."""

import argparse
import csv
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
from utils.metrics import metrics
from utils.seed import set_seed


class OracleTextBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['d_model']
        drop = cfg['dropout']
        self.use_position = cfg['use_position']
        self.text = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim))
        self.audio = nn.Sequential(nn.LayerNorm(74), nn.Linear(74, dim))
        self.vision = nn.Sequential(nn.LayerNorm(35), nn.Linear(35, dim))
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.position = nn.Embedding(50, dim) if self.use_position else None

        def temporal_encoder():
            layer = nn.TransformerEncoderLayer(dim, cfg['heads'], 2 * dim, drop,
                                               batch_first=True, activation='gelu')
            return nn.TransformerEncoder(layer, cfg['temporal_layers'],
                                         enable_nested_tensor=False)

        self.temporal = nn.ModuleList([temporal_encoder() for _ in range(3)])
        self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.fused_encoder = temporal_encoder()
        self.pool = nn.Linear(dim, 1)
        self.cls_head = nn.Linear(dim, 3)
        self.reg_head = nn.Linear(dim, 1)

    def forward(self, batch):
        valid = batch['text_bert'][:, 1, :] > 0
        if not bool(valid.any(1).all()):
            raise ValueError('An Oracle Text sample has no valid token positions')
        x = [self.text(batch['text_teacher']), self.audio(batch['audio']),
             self.vision(batch['vision'])]
        pos = None
        if self.position is not None:
            pos = self.position(torch.arange(valid.shape[1], device=valid.device))[None]
        encoded = []
        for index, (features, encoder) in enumerate(zip(x, self.temporal)):
            features = features + self.modality_embedding[index]
            if pos is not None:
                features = features + pos
            encoded.append(encoder(features, src_key_padding_mask=~valid))
        fused = self.fusion(torch.cat(encoded, dim=-1))
        if pos is not None:
            fused = fused + pos
        fused = self.fused_encoder(fused, src_key_padding_mask=~valid)
        attention = self.pool(fused).squeeze(-1).masked_fill(~valid, -1e4).softmax(1)
        pooled = (attention[..., None] * fused).sum(1)
        return self.cls_head(pooled), 3 * torch.tanh(self.reg_head(pooled).squeeze(-1))


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
    loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                        pin_memory=device.type == 'cuda')
    valid_loader = DataLoader(valid_data, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=0, pin_memory=device.type == 'cuda')
    preflight(next(iter(loader)))
    model = OracleTextBaseline(cfg).to(device)
    counts = torch.bincount(torch.from_numpy(train_data.y_cls), minlength=3).float()
    class_weights = (len(train_data) / (3 * counts)).to(device)
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
    best = {'macro_f1': -float('inf'), 'mae': float('inf'), 'joint': -float('inf')}
    best_epoch = {}
    history = []
    patience = 0
    for epoch in range(cfg['epochs']):
        model.train()
        losses = []
        for raw in loader:
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits, pred_reg = model(batch)
                loss = F.cross_entropy(logits, batch['y_cls'], weight=class_weights)
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
            losses.append(float(loss.detach()))
        score, _ = evaluate_valid(model, valid_loader, device)
        joint = score['f1_macro'] + 0.25 * score['pearson'] - 0.10 * score['mae'] / 3
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), **score, 'joint': joint}
        history.append(row)
        write_csv(output / 'train_history.csv', history)
        print(f"{cfg['run_name']} epoch {epoch + 1}: loss={row['train_loss']:.4f} "
              f"valid_acc={score['accuracy']:.4f} valid_macro_f1={score['f1_macro']:.4f} "
              f"valid_mae={score['mae']:.4f}", flush=True)
        candidates = {'macro_f1': score['f1_macro'], 'mae': score['mae'], 'joint': joint}
        improved_f1 = False
        for name, value in candidates.items():
            improved = value < best[name] if name == 'mae' else value > best[name]
            if improved:
                best[name] = value
                best_epoch[name] = epoch + 1
                torch.save({'state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch + 1},
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
    import json
    result = {'run_name': cfg['run_name'], 'selected_by': 'valid_macro_f1',
              'selected_epoch': best_epoch['macro_f1'], 'valid': score,
              'class_metrics': class_rows, 'best_epochs': best_epoch,
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
