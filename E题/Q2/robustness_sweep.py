import argparse
import pandas as pd
from data.aligned_dataset import AlignedMoseiDataset
from data.missing_simulator import deterministic_span_mask
from evaluate import load_model
from utils.inference import predict_dataset
from utils.io import load_config, output_dir, device_for
from utils.plotting import condition_plot


def measure(model, dataset, device, batch_size, combo, rate, position, seeds, duration=None):
    scores = []
    for seed in seeds:
        score, _ = predict_dataset(model, dataset, device, batch_size,
                                   missing_fn=lambda v,o: deterministic_span_mask(v, combo, rate, position,
                                                                                   seed=seed+o, duration=duration))
        scores.append(score)
    return {key: sum(s[key] for s in scores)/len(scores) for key in scores[0]}


def row(condition, combo, rate_or_length, position, score):
    return {'condition': condition, 'modality': combo, 'rate_or_length': rate_or_length,
            'position': position, **score}


def sweep(cfg, checkpoint):
    device = device_for(cfg)
    dataset = AlignedMoseiDataset(cfg['paths']['aligned_pkl'], 'valid')
    model, _ = load_model(checkpoint, device)
    batch_size = cfg['training']['batch_size']
    out = output_dir(cfg)
    seeds = (101, 102, 103)
    by_type = []
    for combo in ('T', 'A', 'V', 'TA', 'TV', 'AV'):
        by_type.append(row('missing_type', combo, 0.30, 'random',
                           measure(model, dataset, device, batch_size, combo, 0.30, 'random', seeds)))
        print('type', combo, by_type[-1]['f1_macro'], flush=True)
    pd.DataFrame(by_type).to_csv(out/'robustness_by_type.csv', index=False)

    by_rate = []
    for combo in 'TAV':
        for rate in (0.10, 0.20, 0.30, 0.40, 0.50):
            by_rate.append(row('missing_rate', combo, rate, 'random',
                               measure(model, dataset, device, batch_size, combo, rate, 'random', seeds)))
            print('rate', combo, rate, by_rate[-1]['f1_macro'], flush=True)
    pd.DataFrame(by_rate).to_csv(out/'robustness_by_rate.csv', index=False)

    by_position = []
    for combo in 'TAV':
        for position in ('early', 'middle', 'late'):
            by_position.append(row('missing_position', combo, 0.30, position,
                                   measure(model, dataset, device, batch_size, combo, 0.30, position, seeds)))
            print('position', combo, position, by_position[-1]['f1_macro'], flush=True)
    pd.DataFrame(by_position).to_csv(out/'robustness_by_position.csv', index=False)

    by_duration = []
    for combo in 'TAV':
        for duration in (1, 3, 5, 10, 15):
            by_duration.append(row('missing_duration', combo, duration, 'random',
                                   measure(model, dataset, device, batch_size, combo, 0.30, 'random', seeds,
                                           duration=duration)))
            print('duration', combo, duration, by_duration[-1]['f1_macro'], flush=True)
    pd.DataFrame(by_duration).to_csv(out/'robustness_by_duration.csv', index=False)
    for name, rows, x in [('type', by_type, 'modality'), ('rate', by_rate, 'rate_or_length'),
                          ('position', by_position, 'position'), ('duration', by_duration, 'rate_or_length')]:
        condition_plot(pd.DataFrame(rows), x, out/f'fig_missing_{name}.png', f'Missing {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    parser.add_argument('--checkpoint', default='outputs/q2/best_joint.pt')
    args = parser.parse_args()
    sweep(load_config(args.config), args.checkpoint)


if __name__ == '__main__':
    main()
