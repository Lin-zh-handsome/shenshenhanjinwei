import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def compute_metrics(y_cls, probs, y_reg, pred_reg):
    y_cls = np.asarray(y_cls)
    probs = np.asarray(probs)
    y_reg = np.asarray(y_reg)
    pred_reg = np.asarray(pred_reg)
    pred = probs.argmax(axis=1)
    pearson = float(pearsonr(y_reg, pred_reg).statistic) if len(y_reg) > 1 and np.std(y_reg) > 0 and np.std(pred_reg) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_cls, pred)),
        "f1_macro": float(f1_score(y_cls, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_cls, pred, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_reg, pred_reg)),
        "pearson": pearson,
    }


def selection_score(metrics, weights):
    return (weights["macro_f1"] * metrics["f1_macro"] + weights["accuracy"] * metrics["accuracy"]
            + weights["pearson_norm"] * (metrics["pearson"] + 1) / 2
            + weights["mae_norm"] * (1 - metrics["mae"] / 3))
