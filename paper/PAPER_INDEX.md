# 论文结构与材料索引

## 第 1 章 / 问题 1

| 内容 | 来源 |
|---|---|
| 数据 | `E题/Q1/q1_manifest.csv`、`q1_alignment.jsonl`、`q1_quality_report.csv` |
| 方法 | `E题/Q1/q1_pipeline.py`、`run_q1.py`、[Q1_METHOD](q1/Q1_METHOD.md) |
| 公式 | `time_edges[i,k]=k·T_i/50`；对齐和掩码见 `q1_pipeline.py`、`feature_config.yaml` |
| 表 | [特征维度表](tables/table_q1_feature_summary.md)、[Q1 表索引](q1/Q1_TABLES.md) |
| 图 | [Q1 图索引](q1/Q1_FIGURES.md)、`E题/Q1/figures/q1_typical_medium.png`；生成脚本 `E题/Q1/make_q1_report.py` |
| 结果 | `E题/Q1/validation_report.json`、`Q1_FINAL_AUDIT.md`、`q1_features.npz` |

## 第 2 章 / 问题 2

| 内容 | 来源 |
|---|---|
| 数据 | `E题/Q2/data/aligned_dataset.py`、`attachment3_dataset.py`、`data/mask_utils.py` |
| 方法 | `E题/Q2/models/clean_backbone.py`、`srf_msa.py`、`span_geometry.py`、`temporal_reconstruction.py`；[Q2_METHOD](q2/Q2_METHOD.md) |
| 公式/Loss | `E题/Q2/missing_train.py`、`models/span_geometry.py`、`models/temporal_reconstruction.py` |
| 表 | [主结果](tables/table_q2_main_results.md)、[消融](tables/table_q2_ablation.md)、[缺失专项](q2/Q2_TABLES.md) |
| 图 | [Q2 图索引](q2/Q2_FIGURES.md)、`E题/Q2/make_robustness_figures.py` |
| 结果 | `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/metrics.json`、`outputs/q2_best_ablation/ablation_table.csv`、附件 3 预测 |

## 第 3 章 / 问题 3

| 内容 | 来源 |
|---|---|
| 数据 | `E题/Q3/data/aligned_dataset.py`、`attachment4_dataset.py`、`data/feature_baseline.py` |
| 方法 | `E题/Q3/models/fer_msa.py`、`modality_encoder.py`、`evidence_gate.py`、`prediction_heads.py`；[Q3_METHOD](q3/Q3_METHOD.md) |
| 公式/Loss | `E题/Q3/losses/explainable_multitask_loss.py`；LOMO 在 `explain/modality_ablation.py`，局部遮挡在 `explain/local_occlusion.py` |
| 表 | [主结果](tables/table_q3_main_results.md)、[消融](tables/table_q3_ablation.md)、[faithfulness](tables/table_q3_faithfulness.md) |
| 图 | [Q3 图索引](q3/Q3_FIGURES.md)、`E题/Q3/outputs/q3_v2/attachment4/explanation_cards/` |
| 结果 | B1 `01_a1_reproduction/valid_metrics.json`、最终预算 0.25 faithfulness JSON、[解释卡索引](q3/Q3_EXPLANATION_CARD_INDEX.csv) |

所有相对路径以仓库根目录为起点。写结论前核对 [证据矩阵](CLAIM_EVIDENCE_MATRIX.md) 与 [禁止误写清单](PAPER_CAUTION.md)。
