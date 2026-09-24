# 论文写作摘要

先看 [证据矩阵](CLAIM_EVIDENCE_MATRIX.md) 决定结论强度，再从 [论文索引](PAPER_INDEX.md)找原始结果。表中数值全部由脚本读取 [单一来源](SOURCE_OF_TRUTH.md)。

## Q1

**问题**：原始视频多模态信号如何形成统一、可审计的时间网格？

**方案**：按每条实际解码有效时长划分 50 窗，词、音频帧、视频帧映射到同一网格。

**三个方法点**：真实 PTS（显示时间戳）与时长口径；MFA/BERT/PCA（强制对齐/语言编码/主成分分析）文本链路；多帧视觉中位数及分模态质量掩码。Source: `E题/Q1/q1_pipeline.py`、`feature_config.yaml`。

**三个核心结果**：100 条样本完成；三模态及边界 shape（形状）见 [特征表](tables/table_q1_feature_summary.md)；对齐和质量异常见 `E题/Q1/Q1_FINAL_AUDIT.md`。Source: `E题/Q1/validation_report.json`。

**推荐两图**：`paper/figures/q1_alignment_example.png`、`E题/Q1/figures/q1_design_comparison.png`。**推荐两表**：特征维度表、`Q1_FINAL_AUDIT.md` 的质量/聚合比较表。

## Q2

**问题**：局部连续模态缺失下如何稳住极性和强度预测？

**方案/最终模型**：R3 = 已选 BERT 后四层基础网络 + 连续缺失训练 + 区间几何 + 局部重建。Source: `E题/Q2/configs/best_R3_reconstruction.yaml`、`models/srf_msa.py`。

**创新点**：连续缺失训练有分类支持；区间几何作为重建条件保留，单独分类消融未增益；局部重建带来混合缺失分类改善。Source: [消融表](tables/table_q2_ablation.md)。

**核心发现**：clean/missing 的 Accuracy、Macro-F1、MAE、Pearson 见 [主结果表](tables/table_q2_main_results.md)。缺失模态、比例、位置、时长见 [四张专项表](q2/Q2_MISSING_ANALYSIS.md)。附件 3：`E题/Q2/outputs/q2_best_ablation/R3_reconstruction/attachment3_predictions_compact.csv`。

**推荐图表**：Q2 消融图及四张缺失图；主结果、消融、缺失比例/模态表。[图索引](q2/Q2_FIGURES.md)。

## Q3

**问题**：情感预测如何给出可核验的模态和局部证据？

**方案/最终模型**：FER-MSA v2 B1 预测主干 + 仅诊断 Router；LOMO（逐模态删除）给模态贡献，证据门给候选，局部遮挡验证证据。Source: `E题/Q3/configs/q3_v2_final.yaml`、`models/fer_msa.py`。

**创新点与发现**：三层解释见 [Q3 论文笔记](q3/Q3_PAPER_NOTES.md)。预测见 [主表](tables/table_q3_main_results.md)，Top vs Random（关键证据对比随机片段）与回归边界见 [faithfulness 表](tables/table_q3_faithfulness.md)。Source: `E题/Q3/outputs/q3_v2/01_a1_reproduction/valid_metrics.json`、最终 faithfulness JSON。附件 4：`E题/Q3/outputs/q3_v2/attachment4/attachment4_predictions_explanations.csv`。

**推荐图表**：模态贡献图、典型证据卡、faithfulness 图；预测、消融、模态贡献、faithfulness 四表。[图索引](q3/Q3_FIGURES.md)。

## 不要写的内容

Q1 旧 59 窗或人工对齐准确率；Q2 R4 Reliability（可靠性）、R5 consistency（一致性）、Oracle（教师诊断）是最终创新；Q2 clean BERT test 是 R3 test；Q3 旧 A5/B5 是最终模型；Router 是正式解释；附件 4 近似秒数是真实标注；keep-only（仅保留证据）完全保持预测。详见 [完整注意事项](PAPER_CAUTION.md)。
