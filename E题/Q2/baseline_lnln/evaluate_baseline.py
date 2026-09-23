import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.aligned_dataset import load_aligned
from baseline_lnln.common import (BaselineDataset, read_config, official_components,
                                  predict_regression, with_classes, regression_metrics, save_rows)


def evaluate(cfg, split):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_model, _, _, _ = official_components(cfg)
    out = ROOT/cfg['paths']['output_dir']
    checkpoint = torch.load(out/'best_valid_mae.pt', map_location=device, weights_only=False)
    model = build_model(checkpoint['official_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    source = load_aligned(cfg['paths']['aligned_pkl'])
    dataset = BaselineDataset(source, split)
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)
    width = checkpoint['neutral_halfwidth']
    clean = with_classes(predict_regression(model, loader, device, missing=False), width)
    missing = with_classes(predict_regression(model, loader, device, missing=True, seed=101), width)
    results = {'clean': regression_metrics(clean), 'missing30': regression_metrics(missing),
               'neutral_halfwidth_from_valid': width, 'epoch_from_valid': checkpoint['epoch'],
               'train_mask': cfg['training']['train_mask']}
    out.mkdir(parents=True, exist_ok=True)
    save_rows(out/f'{split}_clean_predictions.csv', clean)
    save_rows(out/f'{split}_missing30_predictions.csv', missing)
    (out/f'{split}_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(split, results, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline_lnln/config.yaml')
    parser.add_argument('--split', choices=['valid', 'test'], required=True)
    args = parser.parse_args()
    evaluate(read_config(args.config), args.split)


if __name__ == '__main__':
    main()
