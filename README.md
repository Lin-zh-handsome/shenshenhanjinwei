# 2026 中国研究生数学建模竞赛 E 题

复杂场景下多模态情感预测：从原始音视频特征、局部连续模态缺失下的鲁棒预测，到可解释预测。

## 三个问题

| 问题 | 任务 | 入口 |
|---|---|---|
| Q1 | 多模态特征提取与时序对齐 | [Q1 README](E题/Q1/README.md) |
| Q2 | 局部连续模态缺失条件下的鲁棒情感预测 | [Q2 README](E题/Q2/README.md) |
| Q3 | 可解释多模态情感预测 | [Q3 README](E题/Q3/README.md) |

## 最终方案

- **Q1**：100 条原始视频按各自有效时长划成 50 个共享时间窗；文本 128 维、音频 74 维、视觉 52 维。[最终审计](E题/Q1/Q1_FINAL_AUDIT.md)。
- **Q2**：R3，已选 BERT 后四层微调基础模型，加连续缺失训练、Span Geometry（缺失区间几何编码）和 Temporal Reconstruction（局部时序重建）。clean/missing validation（验证集）指标以 [原始结果](E题/Q2/outputs/q2_best_ablation/R3_reconstruction/metrics.json) 为准。
- **Q3**：FER-MSA v2 B1，三模态时序编码、上下文条件证据门、LOMO（逐模态删除）和局部遮挡核验；Router（路由器）只作诊断。[最终 validation](E题/Q3/outputs/q3_v2/01_a1_reproduction/valid_metrics.json) 与 [解释忠实性](E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json)。

## 论文材料与复现

- [论文材料总索引](paper/PAPER_INDEX.md) · [写作摘要](paper/PAPER_WRITING_BRIEF.md) · [结论证据矩阵](paper/CLAIM_EVIDENCE_MATRIX.md)
- [复现指南](paper/REPRODUCTION.md) · [提交清单](paper/SUBMISSION_MANIFEST.md) · [结果单一来源](paper/SOURCE_OF_TRUTH.md)

仓库不包含赛题原始数据和大型预训练 BERT 权重。Q1 与 Q2 使用不同的数据接口，不能直接把 Q1 特征当作 Q2/Q3 的 `aligned_50` 输入。历史实验、诊断及失败消融均保留，并在各问题 README 中单列。
