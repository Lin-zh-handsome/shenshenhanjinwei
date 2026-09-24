# Q3 FER-MSA 结果汇总

## 标准性能

- valid: Accuracy 0.6099, Macro F1 0.5886, MAE 0.6126, Pearson 0.6481。
- test: Accuracy 0.6699, Macro F1 0.6218, MAE 0.6597, Pearson 0.6513。

## 解释忠实性（valid）

- Sufficiency classification change: 0.2570; regression change: 0.6006。
- 删除 top evidence 后 Macro F1 变化: 0.0716; MAE 增量: -0.0196。
- Top 删除置信度变化 0.0626，随机对照 0.0420；top 是否大于随机：True。
- Top 删除后 Macro F1 0.5170，随机删除平均 0.4917；两种比较口径分别呈现。
- Deletion AUC（删除曲线下面积，分类置信度）0.2974。
- 平均已选位置比例 0.2400，每模态平均段数 1.984。
- Router 与 LOMO 主模态一致率 0.7390。

## 模态作用（valid LOMO 平均）

- text: 0.6183。
- audio: 0.0392。
- vision: 0.3425。

## 消融与附件 4

- 已记录 6 组 A0–A5 消融；详见 ablation_table.csv。
- A0 valid Macro F1 0.6163，A5 valid Macro F1 0.5886；完整模型在该指标上未超过 A0。
- 附件 4 aligned 无标签样本输出 20 行预测；这些样本不报告预测正确率。
- 16/20 个附件 4 视频的报告帧数与实际可解码帧数不同；秒级回投按可解码帧数计算，并在 mapping notes 逐条记录。
- 输入 grid 索引精确；秒级和文本回投采用均匀比例近似，缺少官方逐位置/逐词时间戳。
- 解释指标不构成 ground-truth localization accuracy（真实标注定位准确率）。
