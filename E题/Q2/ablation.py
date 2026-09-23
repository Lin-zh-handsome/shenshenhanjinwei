import argparse
from pathlib import Path
import pandas as pd
from evaluate import load_model
from train import get_datasets, train_variant
from utils.inference import predict_dataset, fixed_missing
from utils.io import load_config, output_dir, device_for


NAMES = ('BasicFusion', '+ContinuousSpanAugmentation', '+SpanGeometryEncoding',
         '+TemporalReconstruction', '+ReliabilityFusion', '+CleanMaskedConsistency')


def ablate(cfg):
    datasets = get_datasets(cfg)
    root = output_dir(cfg)
    device = device_for(cfg)
    rows = []
    for variant in range(6):
        out = root if variant == 5 else root/'ablation'/f'A{variant}'
        checkpoint = out/'best_joint.pt'
        if variant != 5:
            train_variant(cfg, datasets, variant, out, root/'bridge_warmup.pt')
        elif not checkpoint.exists():
            raise FileNotFoundError('Train the full A5 model before ablation')
        model, ckpt = load_model(checkpoint, device)
        clean, _ = predict_dataset(model, datasets['valid'], device, cfg['training']['batch_size'])
        masked, _ = predict_dataset(model, datasets['valid'], device, cfg['training']['batch_size'],
                                    missing_fn=lambda v,o: fixed_missing(v, o, seed=101, rate=0.30))
        row = {'variant': f'A{variant}', 'name': NAMES[variant], 'epoch': ckpt['epoch']}
        row.update({f'clean_{k}': v for k,v in clean.items()})
        row.update({f'missing30_{k}': v for k,v in masked.items()})
        row['robust_drop_f1'] = clean['f1_macro'] - masked['f1_macro']
        row['robust_drop_corr'] = clean['pearson'] - masked['pearson']
        rows.append(row)
        pd.DataFrame(rows).to_csv(root/'ablation_table.csv', index=False)
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    args = parser.parse_args()
    ablate(load_config(args.config))


if __name__ == '__main__':
    main()
