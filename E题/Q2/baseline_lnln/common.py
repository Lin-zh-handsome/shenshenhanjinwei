import csv
import importlib
import random
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from data.mask_utils import infer_shared_valid_mask
from data.missing_simulator import ContinuousSpanMasker
from utils.inference import fixed_missing


class BaselineDataset(Dataset):
    def __init__(self, source, split):
        x = source[split]
        self.ids = [str(v) for v in x['id']]
        self.text = np.asarray(x['text_bert'], dtype=np.int64)
        self.audio = np.asarray(x['audio'], dtype=np.float32)
        self.vision = np.asarray(x['vision'], dtype=np.float32)
        self.y_cls = np.asarray(x['classification_labels'], dtype=np.int64).reshape(-1)
        self.y_reg = np.asarray(x['regression_labels'], dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {'id': self.ids[index], 'text_bert': torch.from_numpy(self.text[index]),
                'audio': torch.from_numpy(self.audio[index]), 'vision': torch.from_numpy(self.vision[index]),
                'y_cls': torch.tensor(self.y_cls[index]), 'y_reg': torch.tensor(self.y_reg[index])}


def read_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def official_components(cfg):
    checkout = Path(cfg['source']['checkout'])
    if not (checkout/'models'/'lnln.py').exists():
        raise FileNotFoundError(f'Official LNLN checkout missing: {checkout}')
    sys.path.insert(0, str(checkout))
    model_module = importlib.import_module('models.lnln')
    loss_module = importlib.import_module('core.losses')
    scheduler_module = importlib.import_module('core.scheduler')
    with open(checkout/cfg['source']['official_config'], encoding='utf-8') as f:
        official_cfg = yaml.safe_load(f)
    official_cfg['dataset']['dataPath'] = cfg['paths']['aligned_pkl']
    official_cfg['model']['feature_extractor']['input_length'] = [50, 50, 50]
    official_cfg['model']['feature_extractor']['bert_pretrained'] = cfg['source']['bert_pretrained']
    official_cfg['base']['batch_size'] = cfg['training']['batch_size']
    official_cfg['base']['n_epochs'] = cfg['training']['epochs']
    official_cfg['base']['num_workers'] = 0
    official_cfg['base']['seed'] = cfg['training']['seed']
    return model_module.build_model, loss_module.MultimodalLoss, scheduler_module.get_scheduler, official_cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def original_random_masks(valid, text_bert):
    b, t = valid.shape
    masks, rates = {}, {}
    for m in 'TAV':
        rate = torch.rand(b, device=valid.device)
        rate[torch.randperm(b, device=valid.device)[:b//2]] = 0
        mask = (torch.rand(b, t, device=valid.device) < rate[:, None]) & valid
        if m == 'T':
            mask[:, 0] = False
            last_text = (text_bert[:, 1] > 0).sum(1).clamp(min=1) - 1
            mask[torch.arange(b, device=valid.device), last_text] = False
        masks[m], rates[m] = mask, rate[:, None]
    return masks, rates


def span_training_masks(valid, masker):
    masks = masker.sample(valid)
    rates = {m: masks[m].float().sum(1, keepdim=True)/valid.sum(1, keepdim=True).clamp(min=1) for m in 'TAV'}
    return masks, rates


def make_inputs(batch, masks):
    text = batch['text_bert'].float()
    text_m = text.clone()
    text_m[:, 0] = torch.where(masks['T'], torch.full_like(text_m[:, 0], 100), text_m[:, 0])
    audio_m = batch['audio'].masked_fill(masks['A'][..., None], 0)
    vision_m = batch['vision'].masked_fill(masks['V'][..., None], 0)
    complete = (batch['vision'], batch['audio'], text)
    incomplete = (vision_m, audio_m, text_m)
    return complete, incomplete


def training_labels(batch, rates):
    b = len(batch['y_reg'])
    device = batch['y_reg'].device
    return {'sentiment_labels': batch['y_reg'][:, None],
            'completeness_labels': 1 - rates['T'],
            'effectiveness_labels': torch.cat((torch.ones(b*8, device=device),
                                               torch.zeros(b*8, device=device))).long()}


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def class_from_reg(values, neutral_halfwidth):
    values = np.asarray(values)
    return np.where(values < -neutral_halfwidth, 0, np.where(values > neutral_halfwidth, 2, 1))


def best_neutral_width(y_true_cls, y_pred_reg):
    from sklearn.metrics import f1_score
    candidates = np.round(np.arange(0, 0.601, 0.025), 3)
    scored = [(float(f1_score(y_true_cls, class_from_reg(y_pred_reg, width),
                               labels=[0,1,2], average='macro', zero_division=0)), float(width)) for width in candidates]
    return max(scored, key=lambda item: (item[0], -item[1]))[1]


def regression_metrics(rows):
    from scipy.stats import pearsonr
    from sklearn.metrics import accuracy_score, f1_score
    y_cls = np.array([r['true_class'] for r in rows])
    p_cls = np.array([r['pred_class'] for r in rows])
    y_reg = np.array([r['true_reg'] for r in rows])
    p_reg = np.array([r['pred_reg'] for r in rows])
    pearson = float(pearsonr(y_reg, p_reg).statistic) if np.std(p_reg) > 1e-12 else 0.0
    return {'accuracy': float(accuracy_score(y_cls, p_cls)),
            'f1_macro': float(f1_score(y_cls, p_cls, labels=[0,1,2], average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_cls, p_cls, labels=[0,1,2], average='weighted', zero_division=0)),
            'mae': float(np.mean(np.abs(y_reg-p_reg))), 'pearson': pearson}


@torch.no_grad()
def predict_regression(model, loader, device, missing=False, seed=101):
    model.eval()
    rows = []
    offset = 0
    for raw in loader:
        batch = to_device(raw, device)
        valid = infer_shared_valid_mask(batch['text_bert'], batch['audio'], batch['vision'])
        masks = fixed_missing(valid, offset, seed=seed, rate=0.30) if missing else {m: torch.zeros_like(valid) for m in 'TAV'}
        _, incomplete = make_inputs(batch, masks)
        pred = model((None, None, None), incomplete)['sentiment_preds'].reshape(-1).float().cpu().numpy()
        for i, sample_id in enumerate(raw['id']):
            rows.append({'id': str(sample_id), 'true_class': int(raw['y_cls'][i]),
                         'true_reg': float(raw['y_reg'][i]), 'pred_reg': float(pred[i])})
        offset += len(raw['id'])
    return rows


def with_classes(rows, width):
    return [{**r, 'pred_class': int(class_from_reg(r['pred_reg'], width))} for r in rows]


def save_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
