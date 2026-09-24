# Q3 可解释多模态情感预测

## Final

1. **任务目标**：在附件 2 `aligned_50` 上预测情感极性、强度，并给附件 4 提供可追溯的模态与局部证据解释。
2. **输入数据**：文本 `[B,50,768]`、音频 `[B,50,74]`、视觉 `[B,50,35]`、有效位置 mask；附件 4 只用对齐版。
3. **最终方法与模型结构**：FER-MSA v2 **B1** 三模态时序编码 + 上下文条件证据门 + 软残差池化 + 拼接融合 + 分类/回归头；LOMO（逐模态删除）和 local occlusion（局部遮挡）形成解释。诊断 Router 不改变预测。[方法](../../paper/q3/Q3_METHOD.md)。
4. **主要创新**：[三层解释](../../paper/q3/Q3_PAPER_NOTES.md)；最终解释经过反事实遮挡验证，模型内部权重不是正式重要性。
5. **训练方式**：[最终配置](configs/q3_v2_final.yaml)、[训练脚本](train.py)；最终权重 [final_model.pt](outputs/q3_v2/final_model.pt)，B1 预测主干 + 诊断 Router。
6. **评价指标**：Accuracy、Macro-F1、MAE、Pearson；解释看 keep-only（仅保留证据）、remove（删除证据）、random remove（随机删除）与置信度变化。
7. **最终实验/结果**：[B1 validation](outputs/q3_v2/01_a1_reproduction/valid_metrics.json)、[最终忠实性](outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json)、[附件 4 预测](outputs/q3_v2/attachment4/attachment4_predictions_explanations.csv)。
8. **消融实验**：[v2 B0–B5](outputs/q3_v2/ablation_v2.csv)；B2 位置编码、B3 门控感知池化、B4 正则及 B5 诊断 Router 不是最终 B1 结构。
9. **推理方式/复现命令**：[复现指南](../../paper/REPRODUCTION.md#q3)；附件 4 无标签，秒数和文本片段是近似回投。
10. **论文对应材料**：[Q3 论文索引](../../paper/q3/Q3_PAPER_NOTES.md)、[表](../../paper/q3/Q3_TABLES.md)、[图](../../paper/q3/Q3_FIGURES.md)、[解释卡索引](../../paper/q3/Q3_EXPLANATION_CARD_INDEX.csv)。

## Ablation

v2 B0–B5 的预测与 faithfulness（解释忠实性）结果都保留；部分版本分类更高，但回归和解释方向不满足最终选择。详见 [v2 报告](Q3_FAITHFULNESS_V2_REPORT.md)。

## Historical / Diagnostic

`outputs/q3/` 的 A0–A5、旧 `best_joint.pt` 与旧 test 属于历史；`configs/q3_aligned.yaml` 不是最终配置。Router 权重只用于诊断，不作正式模态重要性。旧 [README_Q3.md](README_Q3.md)保留兼容入口，最终论文以 v2 为准。
