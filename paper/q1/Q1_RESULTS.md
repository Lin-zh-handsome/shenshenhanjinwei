# Q1 最终结果索引

| 要找的内容 | 来源 |
|---|---|
| 100 条处理结果 | `E题/Q1/q1_features.npz`、`q1_manifest.csv`；审计 `validation_report.json` |
| 特征维度 | `E题/Q1/validation_report.json` 的 `shapes`；[自动表](../tables/table_q1_feature_summary.md) |
| 时间对齐 | `E题/Q1/q1_alignment.jsonl`、`q1_manifest.csv` 的时长与边界 |
| 典型样本 | `E题/Q1/figures/q1_typical_short.png`、`q1_typical_medium.png`、`q1_typical_long.png` |
| 对齐图 | `E题/Q1/figures/q1_typical_medium.png`；[论文精选图](../figures/q1_alignment_example.png) |
| 提取参数 | `E题/Q1/feature_config.yaml`、`q1_run_info.json` |
| 质量审计 | `E题/Q1/Q1_FINAL_AUDIT.md`、`q1_quality_report.csv`、`validation_report.json` |
| 复现命令 | [REPRODUCTION](../REPRODUCTION.md#q1) |

这里不重新处理 100 条视频。无情感 Accuracy、Macro-F1 等指标。词级覆盖率和视觉有效窗比例只能按原审计中的定义引用。
