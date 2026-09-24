# Q2 最终结果索引

| 目标 | Source |
|---|---|
| 最终模型与配置 | `E题/Q2/models/srf_msa.py`、`configs/best_R3_reconstruction.yaml` |
| clean validation（完整验证） | `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/metrics.json` → `clean_valid` |
| mixed missing validation（混合缺失验证） | 同一 JSON → `missing_valid` |
| 消融 | `E题/Q2/outputs/q2_best_ablation/ablation_table.csv` |
| 缺失模态/比例/位置/时长 | R3 目录下 `missing_modality/ratio/position/span_length_analysis.csv` |
| 附件 3 原始/紧凑预测 | R3 目录下 `attachment3_predictions_raw.csv`、`attachment3_predictions_compact.csv` |
| 紧凑参数 validation | R3 目录下 `compact_validation_metrics.json`；与原始模型分开引用 |

[最终结果总表](../FINAL_RESULTS.csv)和[主结果 Markdown 表](../tables/table_q2_main_results.md)由 JSON 生成。R3 未独立重跑 test（测试集）。
