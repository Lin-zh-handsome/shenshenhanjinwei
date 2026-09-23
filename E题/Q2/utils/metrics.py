import warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score


def metrics(y_cls, pred_cls, y_reg, pred_reg):
    y_cls = np.asarray(y_cls).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1)
    y_reg = np.asarray(y_reg).reshape(-1)
    pred_reg = np.asarray(pred_reg).reshape(-1)
    if len(y_cls) != len(pred_cls) or len(y_reg) != len(pred_reg) or len(y_cls) != len(y_reg):
        raise ValueError('Prediction and label lengths differ')
    if np.std(y_reg) < 1e-12 or np.std(pred_reg) < 1e-12:
        warnings.warn('Pearson undefined for constant series; returning 0')
        corr = 0.0
    else:
        corr = float(pearsonr(y_reg, pred_reg).statistic)
    return {'accuracy': float(accuracy_score(y_cls, pred_cls)),
            'f1_macro': float(f1_score(y_cls, pred_cls, labels=[0, 1, 2], average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_cls, pred_cls, labels=[0, 1, 2], average='weighted', zero_division=0)),
            'mae': float(np.mean(np.abs(y_reg - pred_reg))), 'pearson': corr}


def joint_score(m, weights):
    return (weights['macro_f1'] * m['f1_macro'] + weights['accuracy'] * m['accuracy']
            + weights['pearson_norm'] * (m['pearson'] + 1)/2
            + weights['mae_norm'] * (1 - m['mae']/3))
