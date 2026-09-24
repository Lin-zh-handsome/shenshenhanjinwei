# 结果单一来源

机器可读映射见 [source_of_truth.json](source_of_truth.json)。所有论文表由该映射读取，原始结果不复制。

| 项目 | 唯一来源 |
|---|---|
| Q1 最终审计 | `E题/Q1/validation_report.json`、`E题/Q1/Q1_FINAL_AUDIT.md` |
| Q1 100 条特征/清单 | `E题/Q1/q1_features.npz`、`E题/Q1/q1_manifest.csv` |
| Q1 参数 | `E题/Q1/feature_config.yaml` |
| Q2 最终配置 | `E题/Q2/configs/best_R3_reconstruction.yaml` |
| Q2 clean/missing validation | `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/metrics.json` |
| Q2 R0–R5 消融 | `E题/Q2/outputs/q2_best_ablation/ablation_table.csv` |
| Q2 缺失模态/率/位置/长度 | `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/missing_*_analysis.csv` |
| Q2 附件 3 | `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/attachment3_predictions_compact.csv` |
| Q3 最终配置/权重 | `E题/Q3/configs/q3_v2_final.yaml`、`E题/Q3/outputs/q3_v2/final_model.pt` |
| Q3 最终预测 validation | `E题/Q3/outputs/q3_v2/01_a1_reproduction/valid_metrics.json` |
| Q3 v2 消融 | `E题/Q3/outputs/q3_v2/ablation_v2.csv` |
| Q3 最终 faithfulness（解释忠实性） | `E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json` |
| Q3 LOMO（逐模态删除）逐样本贡献 | `E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_explanations.csv` |
| Q3 附件 4 | `E题/Q3/outputs/q3_v2/attachment4/attachment4_predictions_explanations.csv`、`attachment4_explanations_long.csv`、`attachment4_mapping_notes.csv` |

Q2 已存 clean BERT test 仅属于基础模型；Q3 旧 `outputs/q3/test_metrics.json` 属于历史版本。两者均不进入最终结果总表。
