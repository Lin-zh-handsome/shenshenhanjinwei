# Q2 论文图

| 图 | paper 路径 | 原始绘图来源 |
|---|---|---|
| 消融 Macro-F1 | `paper/figures/q2_ablation.png` | `paper/tables/q2_ablation.csv` |
| 缺失比例 | `paper/figures/q2_missing_ratio.png` | `E题/Q2/figures/fig_missing_ratio.png` |
| 缺失模态 | `paper/figures/q2_missing_modality.png` | `E题/Q2/figures/fig_missing_type.png` |
| 缺失位置 | `paper/figures/q2_missing_position.png` | `E题/Q2/figures/fig_missing_position.png` |
| 缺失时长 | `paper/figures/q2_missing_length.png` | `E题/Q2/figures/fig_missing_span_length.png` |

既有四张图由 `E题/Q2/make_robustness_figures.py` 生成；统一入口 `paper/scripts/generate_all_figures.py` 收集既有成品，另由表格生成消融图。
