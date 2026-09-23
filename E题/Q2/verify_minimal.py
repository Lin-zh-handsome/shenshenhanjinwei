import argparse
import csv
import torch
from data.aligned_dataset import AlignedMoseiDataset
from data.attachment3_dataset import Attachment3AlignedDataset
from data.mask_utils import infer_shared_valid_mask
from data.missing_simulator import ContinuousSpanMasker
from models.span_geometry import SpanGeometryEncoder
from models.reliability_fusion import ReliabilityFusion
from models.srf_msa import SRFMSA
from utils.io import load_config, output_dir


def data_checks(cfg):
    ds = AlignedMoseiDataset(cfg['paths']['aligned_pkl'], 'train')
    sample = ds[0]
    batch = {k: v[None] for k,v in sample.items() if torch.is_tensor(v)}
    valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
    masks = ContinuousSpanMasker(cfg['missing_aug'], seed=42).sample(valid)
    for m in 'TAV':
        assert not bool((masks[m] & ~valid).any())
    for span in masks['meta'][0]['spans']:
        m, start, end = span['modality'], span['start'], span['end_exclusive']
        assert bool(masks[m][0, start:end].all())
    geom = SpanGeometryEncoder(cfg['model']['d_model'])(masks['T'], valid)
    assert geom.shape == (1, 50, cfg['model']['d_model'])
    d = cfg['model']['d_model']
    fusion = ReliabilityFusion(d)
    hidden = [torch.randn(1, 50, d) for _ in range(3)]
    _, rel = fusion(*hidden, *([geom]*3), *(masks[m] for m in 'TAV'), valid)
    assert torch.allclose(rel[valid].sum(-1), torch.ones_like(rel[valid].sum(-1)), atol=1e-5)
    cfg['model']['vocab_size'] = max(cfg['model']['vocab_size'], int(ds.text_bert[:,0].max())+1)
    model = SRFMSA(cfg['model'])
    out = model(batch, valid, {m: masks[m] for m in 'TAV'})
    assert bool(((out['reg_pred'] >= -3) & (out['reg_pred'] <= 3)).all())
    assert torch.allclose(out['cls_prob'].sum(-1), torch.ones(1), atol=1e-5)
    attachment = Attachment3AlignedDataset(cfg['paths']['attachment3_dir'])
    one = attachment[0]
    assert one['text_bert'].shape == (3,50) and one['audio'].shape == (50,74) and one['vision'].shape == (50,35)
    print('minimal data/model assertions passed; train', len(ds), 'attachment3', len(attachment), flush=True)


def final_checks(cfg):
    attachment = Attachment3AlignedDataset(cfg['paths']['attachment3_dir'])
    with open(output_dir(cfg)/'attachment3_predictions.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(attachment)
    assert [r['sample_id'] for r in rows] == [p.stem for p in attachment.files]
    assert all(abs(sum(float(r[k]) for k in ('prob_negative', 'prob_neutral', 'prob_positive')) - 1) < 1e-5 for r in rows)
    assert all(-3 <= float(r['pred_intensity']) <= 3 for r in rows)
    print('attachment3 full inference CSV checked:', len(rows), 'rows', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    parser.add_argument('--stage', choices=['data','final'], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    (data_checks if args.stage == 'data' else final_checks)(cfg)


if __name__ == '__main__':
    main()
