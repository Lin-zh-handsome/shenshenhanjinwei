import argparse
from pathlib import Path
import pandas as pd
from utils.io import load_config, output_dir


def export(cfg):
    out = output_dir(cfg)
    required = ['robustness_by_type.csv', 'robustness_by_rate.csv', 'robustness_by_position.csv',
                'robustness_by_duration.csv', 'ablation_table.csv', 'valid_predictions.csv',
                'test_metrics.json', 'attachment3_predictions.csv']
    absent = [name for name in required if not (out/name).exists()]
    if absent:
        raise FileNotFoundError(f'Incomplete experiment outputs: {absent}')
    types = pd.read_csv(out/'robustness_by_type.csv')
    rates = pd.read_csv(out/'robustness_by_rate.csv')
    positions = pd.read_csv(out/'robustness_by_position.csv')
    durations = pd.read_csv(out/'robustness_by_duration.csv')
    ablations = pd.read_csv(out/'ablation_table.csv')
    valid = pd.read_csv(out/'valid_predictions.csv')
    valid['error_score'] = 1.5*(valid.pred_class != valid.true_class) + valid.abs_reg_error
    valid.sort_values('error_score', ascending=False).head(20).to_csv(out/'valid_error_top20.csv', index=False)
    worst_type = types.loc[types.f1_macro.idxmin()]
    first_drop = rates.sort_values('rate_or_length').groupby('rate_or_length').f1_macro.mean()
    strongest_decline = first_drop.diff().idxmin()
    worst_position = positions.groupby('position').f1_macro.mean().idxmin()
    duration_trend = durations.groupby('rate_or_length').f1_macro.mean().sort_index()
    ablations['increment'] = ablations.missing30_f1_macro.diff()
    best_increment = ablations.loc[ablations.increment.idxmax()]
    lines = ['# Q2 实验结果摘要', '',
             '本文件只根据服务器输出表格自动生成；以下分析基于附件 2 valid 的人工连续缺失。', '',
             f'- 30% 缺失时，F1_macro 最低的模态组合：{worst_type.modality}（{worst_type.f1_macro:.4f}）。',
             f'- 平均 F1_macro 相邻缺失率降幅最大的终点缺失率：{strongest_decline:.2f}。',
             f'- 30% 缺失时平均 F1_macro 最低的位置：{worst_position}。',
             f'- 缺失时长 1/3/5/10/15 的平均 F1_macro：' + ', '.join(f'{int(k)}={v:.4f}' for k,v in duration_trend.items()) + '。',
             f'- 消融相邻组中 missing-30% F1_macro 最大增量：{best_increment.variant}（{best_increment.increment:+.4f}）。',
             '', '上述为验证集敏感性分析，不作为附件 3 无标签样本的性能评估。', '']
    (out/'q2_results_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out/'q2_results_summary.md', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/q2_aligned.yaml')
    args = parser.parse_args()
    export(load_config(args.config))


if __name__ == '__main__':
    main()
