# 最终附件提交清单

按赛事平台最终格式与体积限制人工组包；本索引不把原始数据、预训练 BERT 权重或大量历史输出装进新提交包。已有 `Q1_Q2_paper_package.zip` 是历史 Q1/Q2 包，不含本轮 Q3 索引；提交前需按本清单核对。

| 问题 | 必需材料 | 仓库位置 |
|---|---|---|
| Q1 | 代码、配置、100 条特征、清单、质量报告、说明 | `E题/Q1/{run_q1.py,q1_pipeline.py,feature_config.yaml,q1_features.npz,q1_manifest.csv,q1_quality_report.csv,README.md}` |
| Q2 | 核心代码、R3 final 配置、紧凑基础参数与增量、附件 3 CSV、说明 | `E题/Q2/`、`paper_package/Q2/model_parameters/`、`E题/Q2/outputs/q2_best_ablation/R3_reconstruction/attachment3_predictions_compact.csv` |
| Q3 | 核心代码、v2 final 配置、最终权重、附件 4 主 CSV/证据 CSV、解释卡/映射说明、README | `E题/Q3/`、`outputs/q3_v2/final_model.pt`、`outputs/q3_v2/attachment4/` |

最终需保留的 checkpoint（模型参数文件）：Q2 的 `paper_package/Q2/model_parameters/best_bert_last4_compact.pt` 和 `best_robust_delta.pt`（还需外部公开基础 BERT），Q3 的 `E题/Q3/outputs/q3_v2/final_model.pt`。Q1 无训练 checkpoint。完整清单见 [CHECKPOINT_INDEX](CHECKPOINT_INDEX.md)。

不要提交：赛题原始 dataset（数据集）、原始预训练模型、视频、大规模中间实验、缓存、除上述之外的历史 checkpoint。Q3 20 张精选解释卡与 40 张关键帧已经存在，是否随平台附件提交由最终体积与格式要求决定；其 CSV 足以索引。
