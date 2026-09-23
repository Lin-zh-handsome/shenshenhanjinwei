import argparse
import csv
from pathlib import Path
import torch
from data.aligned_dataset import AlignedMoseiDataset
from models.srf_msa import SRFMSA
from utils.inference import predict_dataset
from utils.io import load_config, output_dir, device_for, load_checkpoint, save_json
from utils.plotting import confusion_plot, regression_plot


def load_model(checkpoint, device):
    ckpt = load_checkpoint(checkpoint, device)
    model = SRFMSA(ckpt['config']['model'], variant=ckpt['variant']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(cfg, checkpoint, split):
    device = device_for(cfg)
    dataset = AlignedMoseiDataset(cfg['paths']['aligned_pkl'], split)
    model, _ = load_model(checkpoint, device)
    score, rows = predict_dataset(model, dataset, device, cfg['training']['batch_size'])
    out = output_dir(cfg)
    if split == 'valid' and (out/'valid_metrics.json').exists():
        from json import loads
        existing = loads((out/'valid_metrics.json').read_text(encoding='utf-8'))
        existing['clean_evaluation'] = score
        save_json(out/'valid_metrics.json', existing)
    else:
        save_json(out/f'{split}_metrics.json', score)
    write_rows(out/f'{split}_predictions.csv', rows)
    if split == 'test':
        confusion_plot(rows, out/'fig_confusion_matrix.png')
        regression_plot(rows, out/'fig_regression_scatter.png')
    print(split, score, 'rows', len(rows), flush=True)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    parser.add_argument('--checkpoint', default='outputs/q2/best_joint.pt')
    parser.add_argument('--split', choices=['valid', 'test'], required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint, args.split)


if __name__ == '__main__':
    main()
