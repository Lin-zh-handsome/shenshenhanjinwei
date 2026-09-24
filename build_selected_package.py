"""Build the Q1 plus selected-Q2 paper package from the curated repository tree."""

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
Q2 = ROOT / 'E题' / 'Q2'
PACKAGE = ROOT / 'paper_package'
TARGET = PACKAGE / 'Q2'

CODE = (
    'data/aligned_dataset.py', 'data/attachment3_dataset.py',
    'data/mask_utils.py', 'data/missing_curriculum.py',
    'diagnostics/oracle_text.py', 'diagnostics/oracle_text.yaml',
    'score_push/bert_last4.yaml', 'score_push/compact_best_model.py',
    'score_push/infer_attachment3_best.py',
    'models/bert_text_encoder.py', 'models/clean_backbone.py',
    'models/hierarchical_sentiment_head.py', 'models/text_centered_fusion.py',
    'models/span_geometry.py', 'models/temporal_reconstruction.py',
    'models/srf_msa.py', 'utils/metrics.py', 'utils/seed.py',
    'final_train.py', 'missing_train.py', 'compact_robust_model.py',
    'summarize_best_ablation.py', 'make_robustness_figures.py',
)
RUNS = (
    'R0_selected_clean', 'R1_missing_aug', 'R2_span_geometry',
    'R3_reconstruction', 'R4_reliability', 'R5_consistency',
)


def copy(source, destination):
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    base_compact = TARGET / 'model_parameters' / 'best_bert_last4_compact.pt'
    if not base_compact.is_file():
        raise FileNotFoundError(base_compact)
    retained = base_compact.read_bytes()
    requirements = (TARGET / 'code' / 'requirements.txt').read_bytes()
    if TARGET.resolve().parent != PACKAGE.resolve():
        raise ValueError('Unexpected package target')
    shutil.rmtree(TARGET)
    TARGET.mkdir()
    (TARGET / 'model_parameters').mkdir()
    base_compact.write_bytes(retained)
    (TARGET / 'code').mkdir()
    (TARGET / 'code' / 'requirements.txt').write_bytes(requirements)
    for name in CODE:
        copy(Q2 / name, TARGET / 'code' / name)
    for source in sorted((Q2 / 'configs').glob('best_*.yaml')):
        copy(source, TARGET / 'code' / 'configs' / source.name)
    copy(Q2 / 'outputs' / 'q2_best_ablation' / 'R3_reconstruction' /
         'best_robust_delta.pt', TARGET / 'model_parameters' / 'best_robust_delta.pt')

    clean = Q2 / 'outputs' / 'q2_score_push' / 'deployable_bert' / 'partial_last4'
    for source in sorted(clean.iterdir()):
        if source.is_file() and source.suffix in {'.csv', '.json', '.yaml'}:
            copy(source, TARGET / 'results' / 'clean_bert' / source.name)
    ablation = Q2 / 'outputs' / 'q2_best_ablation'
    copy(ablation / 'ablation_table.csv', TARGET / 'results' / 'ablation_table.csv')
    for run in RUNS:
        for source in sorted((ablation / run).iterdir()):
            if source.is_file() and source.suffix in {'.csv', '.json', '.yaml'}:
                copy(source, TARGET / 'results' / 'ablation' / run / source.name)
    for source in sorted((Q2 / 'figures').iterdir()):
        if source.is_file() and source.suffix in {'.pdf', '.svg', '.png', '.md'}:
            copy(source, TARGET / 'figures' / source.name)
    copy(Q2 / 'README_Q2.md', TARGET / 'README.md')
    copy(Q2 / 'Q2_BEST_MODEL_ABLATION.md', TARGET / 'Q2_BEST_MODEL_ABLATION.md')
    copy(ROOT / 'REPRODUCE_Q1_Q2.md', PACKAGE / 'REPRODUCE_Q1_Q2.md')
    readme = TARGET / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8')
                      .replace('../../paper_package/Q2/model_parameters/', 'model_parameters/')
                      .replace('../../REPRODUCE_Q1_Q2.md', '../REPRODUCE_Q1_Q2.md'),
                      encoding='utf-8')
    report = TARGET / 'Q2_BEST_MODEL_ABLATION.md'
    report.write_text(report.read_text(encoding='utf-8')
                      .replace('outputs/q2_best_ablation/R3_reconstruction/',
                               'results/ablation/R3_reconstruction/')
                      .replace('outputs/q2_best_ablation/ablation_table.csv',
                               'results/ablation_table.csv')
                      .replace('outputs/q2_best_ablation/R*/', 'results/ablation/R*/')
                      .replace('REPRODUCE_Q1_Q2.md', '../REPRODUCE_Q1_Q2.md'),
                      encoding='utf-8')
    captions = TARGET / 'figures' / 'FIGURE_CAPTIONS.md'
    captions.write_text(captions.read_text(encoding='utf-8')
                        .replace('outputs/q2_best_ablation/', 'results/ablation/'),
                        encoding='utf-8')

    archive = ROOT / 'Q1_Q2_paper_package.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=False) as handle:
        for source in sorted(PACKAGE.rglob('*')):
            if source.is_file():
                handle.write(source, source.relative_to(PACKAGE.parent))
    size_mib = archive.stat().st_size / 2**20
    if size_mib >= 50:
        raise ValueError(f'Paper package exceeds 50 MiB: {size_mib:.2f} MiB')
    print(f'Built {archive}: {size_mib:.2f} MiB')


if __name__ == '__main__':
    main()
