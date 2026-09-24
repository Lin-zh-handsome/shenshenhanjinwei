from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


LABELS = ["Negative", "Neutral", "Positive"]


def confusion_figure(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def regression_figure(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5)
    ax.plot([-3, 3], [-3, 3], "k--", lw=1)
    ax.set(xlabel="True intensity", ylabel="Predicted intensity", xlim=(-3, 3), ylim=(-3, 3))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plot(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
