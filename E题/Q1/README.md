# Q1 多模态特征提取与时序对齐

## Final

1. **任务目标**：对附件 1 的 100 条视频抽取文本、音频、视觉特征，按每条视频有效时长统一成 50 个共享窗口。
2. **输入数据**：原视频与官方转写/标签表；标签不参与特征拟合。[原说明](README_Q1.md)。
3. **最终方法与结构**：MFA（强制对齐）词级边界和 BERT 文本向量；音频 37 项帧描述子均值及标准差；视觉逐帧 blendshape（面部形变系数）中位数；共用真实时间边界。详见 [方法索引](../../paper/q1/Q1_METHOD.md)。
4. **主要创新**：每样本真实时长 50 等分、三模态同窗聚合、独立质量掩码；无情感分类模型。
5. **训练方式**：无需情感模型训练；文本 PCA（主成分分析）仅在有效文本窗口无监督拟合。预训练模型需要单独准备。
6. **评价指标**：形状、解码完整率、覆盖率、对齐边界与质量审计；没有情感 Accuracy/F1。
7. **最终实验**：100 条输出见 [验证报告](validation_report.json)、[逐样本清单](q1_manifest.csv)、[完整审计](Q1_FINAL_AUDIT.md)。
8. **消融实验**：窗口粒度、视觉聚合、文本对齐对照见 [最终审计](Q1_FINAL_AUDIT.md)；没有独立人工时间标注精度。
9. **最终结果文件**：[特征 NPZ](q1_features.npz)、[对齐 JSONL](q1_alignment.jsonl)、[质量 CSV](q1_quality_report.csv)。
10. **推理方式**：`python run_q1.py --stage all` 生成特征；无需预测标签。
11. **复现命令**：[复现指南](../../paper/REPRODUCTION.md#q1)。
12. **论文对应材料**：[Q1 论文索引](../../paper/q1/Q1_PAPER_NOTES.md)、[表](../../paper/q1/Q1_TABLES.md)、[图](../../paper/q1/Q1_FIGURES.md)。

## Ablation

粒度和聚合比较用于方法选择；因旧版与新版分母不同，不可将差值归于单一模块。Source: `Q1_FINAL_AUDIT.md`。

## Historical / Diagnostic

[实现审计](Q1_IMPLEMENTATION_AUDIT.md)记录旧 0.5 秒/59 窗方案的偏差，不作为最终特征维度或结果引用。[README_Q1.md](README_Q1.md)保留完整技术说明。
