"""Four single-panel Q2 robustness figures from the archived validation CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

BASE = '#8A99A8'
FINAL = '#176B8A'
ROOT = Path(__file__).resolve().parent
RESULTS = (ROOT / 'outputs' / 'q2_best_ablation'
           if (ROOT / 'outputs' / 'q2_best_ablation').is_dir()
           else ROOT.parent / 'results' / 'ablation')


def read_rows(run, filename, key, expected):
    source = RESULTS / run / filename
    with source.open(newline='', encoding='utf-8') as handle:
        records = {row[key]: row for row in csv.DictReader(handle)}
    if tuple(records) != tuple(expected):
        raise ValueError(f'{source}: expected categories {expected}, got {tuple(records)}')
    values = np.asarray([float(records[label]['macro_f1']) for label in expected])
    if not np.isfinite(values).all():
        raise ValueError(f'{source}: nonfinite Macro-F1')
    return values


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{stem}.svg', bbox_inches='tight')
    fig.savefig(directory / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def bars(directory, stem, filename, key, categories, labels, title, ylim):
    base = read_rows('R0_selected_clean', filename, key, categories)
    final = read_rows('R3_reconstruction', filename, key, categories)
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout='constrained')
    ax.bar(x - .19, base, width=.36, color=BASE, label='R0 Clean BERT')
    ax.bar(x + .19, final, width=.36, color=FINAL, label='R3 Span reconstruction')
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel('Macro-F1')
    ax.set_title(title, loc='left', weight='bold')
    ax.grid(axis='y', color='#DCE1E5', linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=2)
    save(fig, directory, stem)


def ratio(directory):
    categories = ('0.1', '0.2', '0.3', '0.4', '0.5')
    filename = 'missing_ratio_analysis.csv'
    base = read_rows('R0_selected_clean', filename, 'ratio', categories)
    final = read_rows('R3_reconstruction', filename, 'ratio', categories)
    x = np.asarray([10, 20, 30, 40, 50])
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout='constrained')
    ax.plot(x, base, marker='o', color=BASE, linewidth=1.7, label='R0 Clean BERT')
    ax.plot(x, final, marker='s', color=FINAL, linewidth=1.7,
            label='R3 Span reconstruction')
    ax.set_xticks(x)
    ax.set_xlim(8, 52)
    ax.set_ylim(.25, .7)
    ax.set_xlabel('Contiguous missing ratio (%)')
    ax.set_ylabel('Macro-F1')
    ax.set_title('All three modalities: middle-position span', loc='left', weight='bold')
    ax.grid(axis='y', color='#DCE1E5', linewidth=.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.22), ncol=2)
    save(fig, directory, 'fig_missing_ratio')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(ROOT / 'figures'))
    args = parser.parse_args()
    directory = Path(args.output_dir)
    bars(directory, 'fig_missing_type', 'missing_modality_analysis.csv',
         'modality', ('T', 'A', 'V', 'TA', 'TV', 'AV'),
         ('T', 'A', 'V', 'T+A', 'T+V', 'A+V'),
         'Missing modality: 30% middle-position span', (0, .7))
    ratio(directory)
    bars(directory, 'fig_missing_position', 'missing_position_analysis.csv',
         'position', ('early', 'middle', 'late'),
         ('Early', 'Middle', 'Late'),
         'All three modalities: 30% span by position', (0, .7))
    bars(directory, 'fig_missing_span_length', 'missing_span_length_analysis.csv',
         'length', ('short', 'medium', 'long'),
         ('Short (4)', 'Medium (10)', 'Long (20)'),
         'All three modalities: middle-position span by length', (0, .7))
    print(f'Created four PDF/SVG/PNG figures in {directory}')


if __name__ == '__main__':
    main()
