import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def confusion_plot(rows, path):
    matrix = confusion_matrix([r['true_class'] for r in rows], [r['pred_class'] for r in rows], labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
    ax.set_xticks(range(3), ['Negative', 'Neutral', 'Positive'])
    ax.set_yticks(range(3), ['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def regression_plot(rows, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([r['true_reg'] for r in rows], [r['pred_reg'] for r in rows], s=8, alpha=0.35)
    ax.plot([-3, 3], [-3, 3], color='black', linewidth=1)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='True intensity', ylabel='Predicted intensity')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def condition_plot(df, x, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if x == 'modality':
        order = ['T', 'A', 'V', 'TA', 'TV', 'AV']
        part = df.set_index('modality').loc[order]
        axes[0].bar_label(axes[0].bar(order, part['f1_macro']), fmt='%.3f', fontsize=7)
        axes[1].bar_label(axes[1].bar(order, part['mae']), fmt='%.3f', fontsize=7)
        axes[0].set_ylim(0, part['f1_macro'].max() * 1.18)
        axes[1].set_ylim(0, part['mae'].max() * 1.18)
    else:
        for modality, part in df.groupby('modality'):
            if x == 'position':
                part = part.set_index(x).loc[['early', 'middle', 'late']].reset_index()
            else:
                part = part.sort_values(x)
            axes[0].plot(part[x].astype(str), part['f1_macro'], marker='o', label=modality)
            axes[1].plot(part[x].astype(str), part['mae'], marker='o', label=modality)
    axes[0].set_ylabel('Macro F1')
    axes[1].set_ylabel('MAE')
    for ax in axes:
        ax.set_xlabel({'Missing type': 'Missing modality combination',
                       'Missing position': 'Span position',
                       'Missing rate': 'Missing rate',
                       'Missing duration': 'Span length (frames)'}.get(title, x))
        ax.grid(alpha=0.2)
        if x != 'modality':
            ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=35)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
