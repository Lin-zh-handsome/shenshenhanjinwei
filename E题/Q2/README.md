# Q2 连续局部缺失下的鲁棒情感预测

## Final

1. **任务目标**：在官方附件 2 `aligned_50` train/valid/test 划分上预测三类情感与强度，并处理连续局部缺失。
2. **输入数据**：`text_bert` token、音频 `[B,50,74]`、视觉 `[B,50,35]`、有效位置与训练时人工缺失 mask（掩码）。
3. **最终方法与模型结构**：R3 = `Deployable_BERT_Last4` 冻结基础网络 + 连续缺失训练 + Span Geometry（缺失区间几何编码） + Temporal Reconstruction（局部时序重建）。[方法](../../paper/q2/Q2_METHOD.md)。
4. **主要创新**：[创新与支持证据](../../paper/q2/Q2_PAPER_NOTES.md)；R2 相对 R1 的分类结果未提升，表述须保守。
5. **训练方式**：基础 BERT 后四层微调，随后冻结基础网络并训练缺失分支；[最终配置](configs/best_R3_reconstruction.yaml)、[训练脚本](missing_train.py)。
6. **评价指标**：Accuracy、Macro-F1、Weighted-F1、MAE、Pearson；用验证集 clean 与固定 mixed-missing 场景选模。
7. **最终实验/结果**：[R3 clean/missing 验证](outputs/q2_best_ablation/R3_reconstruction/metrics.json)、[附件 3 紧凑预测](outputs/q2_best_ablation/R3_reconstruction/attachment3_predictions_compact.csv)。附件 3 无标签。
8. **消融实验**：[R0–R5 原始表](outputs/q2_best_ablation/ablation_table.csv)、[论文表](../../paper/tables/table_q2_ablation.md)、[缺失专项](../../paper/q2/Q2_MISSING_ANALYSIS.md)。
9. **推理方式**：`compact_robust_model.py` 用紧凑基础参数、R3 增量与公开 BERT 权重恢复模型；原始 R3 checkpoint 未入库。
10. **复现命令**：[复现指南](../../paper/REPRODUCTION.md#q2)。
11. **论文对应材料**：[Q2 索引](../../paper/q2/Q2_PAPER_NOTES.md)、[表](../../paper/q2/Q2_TABLES.md)、[图](../../paper/q2/Q2_FIGURES.md)。

## Ablation

R0 是 clean 基础模型，R1 缺失增强，R2 区间几何，R3 重建，R4 可靠性残差，R5 一致性。R4/R5 保留在消融表，**未进入最终 R3**。完整选择规则与局限见 [原报告](Q2_BEST_MODEL_ABLATION.md)。

## Historical / Diagnostic

`diagnostics/oracle_text.py` 是 teacher（教师特征）诊断；`score_push/` 中 clean BERT 的旧 test 指标不属于 R3；`models/text_centered_fusion.py` 与旧 SRF/Reliability 路径不是最终融合方式。历史代码和 CSV 不删除。[旧完整说明](README_Q2.md)保留。
