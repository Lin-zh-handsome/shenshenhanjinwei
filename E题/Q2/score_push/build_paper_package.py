"""Assemble the paper-facing Q1/Q2 source, evidence and small submission ZIP."""

from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'paper_package'
Q2 = ROOT / 'E题/Q2'


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    q1 = PACKAGE / 'Q1'
    if q1.exists():
        shutil.rmtree(q1)
    shutil.copytree(ROOT / 'E题/Q1', q1, ignore=shutil.ignore_patterns('__pycache__'))
    code = PACKAGE / 'Q2/code'
    code_files = [
        'data/aligned_dataset.py', 'data/attachment3_dataset.py',
        'diagnostics/oracle_text.py', 'utils/metrics.py', 'utils/seed.py',
        'score_push/bert_last4.yaml', 'score_push/infer_attachment3_best.py',
        'score_push/compact_best_model.py', 'score_push/evaluate_selected_test.py',
    ]
    for relative in code_files:
        copy(Q2 / relative, code / relative)
    (code / 'requirements.txt').write_text(
        'numpy>=1.26\ntorch>=2.2\ntransformers>=4.45\n'
        'scikit-learn>=1.4\nPyYAML>=6.0\n', encoding='utf-8')
    config = code / 'score_push/bert_last4.yaml'
    text = config.read_text(encoding='utf-8')
    text = text.replace('/home/hanjinwei/math/data/E题数据/附件2-数据集特征文件/aligned_50.pkl',
                        '/path/to/aligned_50.pkl')
    text = text.replace('/home/hanjinwei/math/model_cache/bert-base-uncased',
                        '/path/to/bert-base-uncased')
    config.write_text(text, encoding='utf-8')
    selected = Q2 / 'outputs/q2_score_push/deployable_bert/partial_last4'
    results = PACKAGE / 'Q2/results'
    result_files = [
        'metrics.json', 'class_metrics.csv', 'confusion_matrix.csv',
        'train_history.csv', 'valid_predictions.csv',
        'test_metrics.json', 'test_class_metrics.csv', 'test_confusion_matrix.csv',
        'test_predictions.csv', 'attachment3_predictions_best_bert.csv',
    ]
    for name in result_files:
        copy(selected / name, results / name)
    copy(Q2 / 'outputs/q2_score_push/score_push_ablation.csv',
         results / 'score_push_ablation.csv')
    copy(Q2 / 'outputs/q2_score_push/modality_ablation_clean.csv',
         results / 'modality_ablation_clean.csv')
    copy(Q2 / 'outputs/q2_v2/01_run_a_oracle_text/metrics.json',
         results / 'oracle_text_reference_metrics.json')
    copy(Q2 / 'SCORE_PUSH_REPORT.md', PACKAGE / 'Q2/SCORE_PUSH_REPORT.md')
    for cache in PACKAGE.rglob('__pycache__'):
        shutil.rmtree(cache)
    archive = ROOT / 'Q1_Q2_paper_package.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for file in sorted(PACKAGE.rglob('*')):
            if file.is_file():
                handle.write(file, file.relative_to(ROOT))
    size = archive.stat().st_size
    print(f'{archive}: {size / 2**20:.2f} MiB')
    if size > 50_000_000:
        raise ValueError('Contest package exceeds 50,000,000 bytes')


if __name__ == '__main__':
    main()
