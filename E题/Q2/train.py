import argparse
import copy
import csv
import math
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.aligned_dataset import load_aligned, AlignedMoseiDataset
from data.mask_utils import infer_shared_valid_mask, infer_missing_candidates, combine_masks
from data.missing_simulator import ContinuousSpanMasker
from losses.multitask_loss import compute_loss, masked_smooth_l1
from models.srf_msa import SRFMSA
from utils.inference import to_device, predict_dataset, fixed_missing
from utils.io import load_config, output_dir, device_for, save_json
from utils.metrics import joint_score
from utils.seed import set_seed


def get_datasets(cfg):
    source = load_aligned(cfg['paths']['aligned_pkl'])
    datasets = {s: AlignedMoseiDataset(split=s, source=source) for s in ('train', 'valid', 'test')}
    max_id = int(datasets['train'].text_bert[:, 0].max())
    cfg['model']['vocab_size'] = max(cfg['model']['vocab_size'], max_id+1)
    print('splits:', {k: len(v) for k, v in datasets.items()}, 'max_train_token:', max_id, flush=True)
    del source
    return datasets


def prewarm_bridge(model, train_ds, valid_ds, cfg, device, path):
    if path.exists():
        model.text_bridge.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        return
    opt = torch.optim.AdamW(model.text_bridge.parameters(), lr=cfg['training']['lr'])
    loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)
    for epoch in range(cfg['training']['bridge_warmup_epochs']):
        model.text_bridge.train()
        train_loss = []
        for raw in loader:
            batch = to_device(raw, device)
            valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
            original = infer_missing_candidates(batch['text_bert'], batch['audio'], batch['vision'], valid)
            _, recon = model.text_bridge(batch['text_bert'], valid, original['T'])
            loss = masked_smooth_l1(recon, batch['text_teacher'], valid & ~original['T'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss.append(float(loss.detach()))
        model.text_bridge.eval()
        val_loss = []
        with torch.no_grad():
            for raw in valid_loader:
                batch = to_device(raw, device)
                valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
                original = infer_missing_candidates(batch['text_bert'], batch['audio'], batch['vision'], valid)
                _, recon = model.text_bridge(batch['text_bert'], valid, original['T'])
                val_loss.append(float(masked_smooth_l1(recon, batch['text_teacher'], valid & ~original['T'])))
        print(f'bridge epoch {epoch+1}: train={sum(train_loss)/len(train_loss):.5f} valid={sum(val_loss)/len(val_loss):.5f}', flush=True)
    torch.save(model.text_bridge.state_dict(), path)


def train_variant(cfg, datasets, variant=5, out=None, warmup_path=None):
    set_seed(cfg['seed'])
    device = device_for(cfg)
    out = Path(out or output_dir(cfg))
    out.mkdir(parents=True, exist_ok=True)
    model = SRFMSA(cfg['model'], variant=variant).to(device)
    if warmup_path:
        prewarm_bridge(model, datasets['train'], datasets['valid'], cfg, device, Path(warmup_path))
    counts = torch.bincount(torch.from_numpy(datasets['train'].y_cls), minlength=3).float()
    class_weights = (len(datasets['train'])/(3*counts.clamp(min=1))).to(device)
    batch_size = cfg['training']['batch_size']
    loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == 'cuda')
    epochs = cfg['training']['epochs']
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    steps = epochs * len(loader)
    warmup = max(1, int(steps * cfg['training']['warmup_ratio']))
    def lr_scale(step):
        return (step+1)/warmup if step < warmup else 0.5*(1+math.cos(math.pi*(step-warmup)/max(1, steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    amp = bool(cfg['runtime']['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    masker = ContinuousSpanMasker(cfg['missing_aug'], cfg['seed'])
    best = {'cls': -1e9, 'reg': 1e9, 'joint': -1e9}
    tie_break = {'cls': -1e9, 'reg': -1e9}
    best_joint_metrics = None
    best_joint_epoch = None
    patience = 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for raw in tqdm(loader, desc=f'A{variant} epoch {epoch+1}', leave=False, disable=not sys.stderr.isatty()):
            batch = to_device(raw, device)
            valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
            original = infer_missing_candidates(batch['text_bert'], batch['audio'], batch['vision'], valid)
            artificial = {m: torch.zeros_like(valid) for m in 'TAV'} if variant == 0 else masker.sample(valid)
            missing = combine_masks(original, artificial)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                clean = model(batch, valid, original) if variant >= 3 else None
                masked = model(batch, valid, missing)
                loss, components = compute_loss(masked, clean, batch, valid, original, artificial,
                                                class_weights, cfg, variant)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            losses.append(float(loss.detach()))
        clean_m, _ = predict_dataset(model, datasets['valid'], device, batch_size)
        missing_m, _ = predict_dataset(model, datasets['valid'], device, batch_size,
                                       missing_fn=lambda v,o: fixed_missing(v, o, seed=101, rate=0.30))
        score = joint_score(missing_m, cfg['selection'])
        row = {'epoch': epoch+1, 'train_loss': sum(losses)/len(losses), 'selection_joint': score,
               **{f'clean_{k}': v for k,v in clean_m.items()},
               **{f'missing30_{k}': v for k,v in missing_m.items()}}
        history.append(row)
        print(f'A{variant} epoch {epoch+1}: loss={row["train_loss"]:.4f} clean_f1={clean_m["f1_macro"]:.4f} miss_f1={missing_m["f1_macro"]:.4f} joint={score:.4f}', flush=True)
        with open(out/'train_history.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        candidates = {'cls': (missing_m['f1_macro'], 'max'), 'reg': (missing_m['mae'], 'min'), 'joint': (score, 'max')}
        improved_joint = False
        for key, (value, direction) in candidates.items():
            improved = value > best[key] if direction == 'max' else value < best[key]
            if key == 'cls' and value == best[key] and missing_m['accuracy'] > tie_break['cls']:
                improved = True
            if key == 'reg' and value == best[key] and missing_m['pearson'] > tie_break['reg']:
                improved = True
            if improved:
                best[key] = value
                if key == 'cls':
                    tie_break['cls'] = missing_m['accuracy']
                if key == 'reg':
                    tie_break['reg'] = missing_m['pearson']
                torch.save({'state_dict': model.state_dict(), 'config': cfg, 'variant': variant, 'epoch': epoch+1,
                            'valid_clean': clean_m, 'valid_missing30': missing_m}, out/f'best_{key}.pt')
                if key == 'joint':
                    improved_joint = True
                    best_joint_metrics = {'clean': clean_m, 'missing30': missing_m}
                    best_joint_epoch = epoch + 1
        patience = 0 if improved_joint else patience+1
        save_json(out/'valid_metrics.json', {**best_joint_metrics, 'best_selection': best,
                                             'best_joint_epoch': best_joint_epoch})
        if patience >= cfg['training']['early_stop_patience']:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    parser.add_argument('--variant', type=int, default=5, choices=range(6))
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    cfg = load_config(args.config)
    datasets = get_datasets(cfg)
    root = output_dir(cfg)
    out = Path(args.output_dir) if args.output_dir else (root if args.variant == 5 else root/'ablation'/f'A{args.variant}')
    train_variant(cfg, datasets, variant=args.variant, out=out, warmup_path=root/'bridge_warmup.pt')


if __name__ == '__main__':
    main()
