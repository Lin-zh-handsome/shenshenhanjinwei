# 已完成实验总览

## Q1

最终方案：按真实解码时长构造 50 个共享时间窗；MFA/BERT/PCA 文本、74 维音频、52 维视觉。结果：100 条样本、特征形状见 [自动表](tables/table_q1_feature_summary.md)。Source: `E题/Q1/validation_report.json`。推荐表：特征维度、质量覆盖；推荐图：中等时长对齐示例、设计比较图。Source: `E题/Q1/Q1_FINAL_AUDIT.md`、`E题/Q1/figures/`。

## Q2

Final model：R3，BERT 后四层基础模型 + 连续缺失 + 区间几何 + 局部重建。Clean/Missing performance（完整/缺失输入表现）见 [自动主表](tables/table_q2_main_results.md)。Source: `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/metrics.json`。关键消融与失败方案见 [自动消融表](tables/table_q2_ablation.md)。Source: `E题/Q2/outputs/q2_best_ablation/ablation_table.csv`。缺失模态、比例、位置、长度规律见 [专项](q2/Q2_MISSING_ANALYSIS.md)。附件 3 结果：`E题/Q2/outputs/q2_best_ablation/R3_reconstruction/attachment3_predictions_compact.csv`。

## Q3

Final model：FER-MSA v2 B1 + 仅诊断 Router；证据门、LOMO（逐模态删除）、局部遮挡核验。Prediction（预测）见 [自动主表](tables/table_q3_main_results.md)。Source: `E题/Q3/outputs/q3_v2/01_a1_reproduction/valid_metrics.json`。Faithfulness（解释忠实性）见 [自动表](tables/table_q3_faithfulness.md)。Source: `E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json`。消融见 [Q3 表](tables/table_q3_ablation.md)。附件 4 结果：`E题/Q3/outputs/q3_v2/attachment4/attachment4_predictions_explanations.csv`。
