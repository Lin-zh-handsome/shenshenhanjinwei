# Q2 最佳模型与连续局部缺失消融

## 模型身份与选择

基础模型是 `Deployable_BERT_Last4`：以附件 2 的官方 `train` 训练，在 `valid` 上选出 epoch 25。它使用 `text_bert`、音频和视觉，BERT 最后四层微调，三模态 128 维投影、各自一层时序 Transformer（变换器）、拼接融合、一层融合 Transformer 和注意力池化。最终报告的鲁棒版本 **R3** 在这个已选定的 checkpoint（模型参数文件）上训练连续缺失、区间几何与局部重建模块；其原基础网络权重冻结。R3 **不含** Reliability Fusion（可靠性融合）或 Clean-Masked Consistency（干净与遮挡输入一致性），因为后续消融未提高预先定义的综合排序。

附件 2 官方划分保持 `train=3395`、`valid=728`、`test=727`。六组消融使用 seed 42、相同的基础权重、划分、batch 和最长 20 epoch。R0 是加载基础权重后的零训练对照；R1–R5 是逐项增加结构与损失的独立训练运行，不把上一组训练出的模块参数当作下一组初值。基础网络冻结使 R1–R3 的完整输入输出与原模型一致。R1–R5 训练输入按固定 curriculum（渐进缺失计划）生成连续 span（连续区间）：epoch 1–3 无缺失，4–7 缺失概率 0.25，8–12 为 0.40，13 起为 0.50。每个区间的真实人工 mask（掩码）在增强时直接保存，padding（填充位置）不计为缺失。

每轮以验证集 clean（完整输入）和 mixed missing（混合缺失）各占一半进行排序；单侧分类分数为 `0.55 × Macro-F1 + 0.45 × Accuracy`。混合缺失评估对每个验证样本按固定顺序选择 T、A、V、TA、TV、AV 之一，在有效长度的中间遮挡 30%。该排序是本实验的选模规则，不是赛题官方综合评分。`test` 从未用于 R0–R5 的训练、调参或选模。

| 消融 | clean Acc | clean Macro-F1 | clean MAE | clean Pearson | missing Acc | missing Macro-F1 | missing MAE | missing Pearson | 综合排序 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 原最佳 BERT | .6429 | .6169 | .6297 | .6159 | .5797 | .5627 | .6716 | .5445 | .5995 |
| R1 + 连续缺失训练 | .6429 | .6169 | .6297 | .6159 | .5962 | .5744 | .6594 | .5656 | .6064 |
| R2 + 区间几何 | .6429 | .6169 | .6297 | .6159 | .5948 | .5721 | .6528 | .5667 | .6054 |
| **R3 + 局部重建** | **.6429** | **.6169** | .6297 | .6159 | **.6195** | **.5920** | .6686 | .5631 | **.6165** |
| R4 + 可靠性残差 | .6291 | .6072 | .6251 | .6155 | .6168 | .5909 | .6639 | .5621 | .6098 |
| R5 + 干净/遮挡一致性 | .6291 | .6075 | .6252 | .6153 | .6209 | .5965 | .6657 | .5698 | .6123 |

R3 相对 R0 在混合缺失下 Accuracy 提高 **3.98 个百分点**，Macro-F1 提高 **2.92 个百分点**；完整输入指标不变。R2 相比 R1 对缺失分类略降，但 MAE 与 Pearson 略好；R3 的主要收益来自重建。R4/R5 对部分缺失指标有增益，却降低了完整输入分类分数，因此不保留在最终模型。R3 缺失场景 Neutral F1 为 .4498，仍是弱项；完整输入 Neutral precision/recall/F1 为 .5321/.4511/.4882。

## R3 实际实现与训练目标

基础 BERT 与三模态时序/融合网络加载 `Deployable_BERT_Last4` 权重后冻结。每个被人工遮挡的位置以可学习 missing token（缺失标记向量）代替模态投影，并加入由区间比例、左右边界相对距离、区间长度和绝对位置组成的 5 维 Span Geometry（区间几何）编码。每个模态各有一层 Temporal Reconstruction Encoder（时序重建编码器），内部使用长度 50 的可学习时间位置向量；只在该模态确有人工缺失时重建隐藏表示。重建表示进入原三模态时序编码与拼接融合。**原基础时序网络的 `use_position=false` 没有被改写**；时间位置向量仅存在于新加的重建分支，论文不能声称整个模型各层都使用绝对位置编码。

训练目标为 `L = weighted CE + 0.5 × SmoothL1(regression) + 0.1 × L_recon`。类别权重为训练集 `N/(3N_c)`；`L_recon` 是人工缺失位置的重建表示与冻结基础网络在同一 clean（未遮挡）位置投影之间的平均 L1 距离，clean 目标停止梯度，padding 不计入。没有采用旧版 `p_positive - p_negative = regression/3` 约束。R4 另外试验独立 sigmoid（S 形门控）可靠性残差；R5 再增加 clean 教师预测到遮挡预测的 KL 散度与回归 SmoothL1 一致性，均未进入最终 R3。

## 缺失专项结果与限制

R3 的各组专项验证见 `outputs/q2_best_ablation/R3_reconstruction/`：`missing_modality_analysis.csv`、`missing_ratio_analysis.csv`、`missing_position_analysis.csv`、`missing_span_length_analysis.csv`。当 T/A/V 同时在中间缺失 30% 时，Accuracy/Macro-F1 为 .6030/.5698；缺失率增加到 50% 时降至 .5247/.4970。对 T 单模态缺失 30%，Accuracy/Macro-F1 为 .6044/.5711；对 A 单模态缺失为 .6429/.6171。模型仍高度依赖文本。连续缺失 20 个位置时 Accuracy/Macro-F1 为 .4052/.3959，不能声称长缺失完全解决。

这些缺失场景由附件 2 验证集的**确定性人工遮挡**生成，指标不是附件 3 的准确率。附件 3 无标签，也没有官方逐位置缺失真值；推理时仅以有效区间内 `attention_mask=0` 或音/视整帧全零建立 **zero-derived missing candidates（零值推断缺失候选）**，不能称为真实缺失标注。30 条预测和每条候选缺失位置计数见 `attachment3_predictions_raw.csv` 与 `attachment3_predictions_compact.csv`。原始与紧凑参数在这 30 条上的类别相同，强度最大绝对差约 .0230。

R0–R5 是单 seed 消融；本报告不声称跨 seed 显著性。没有在这些鲁棒版本上重新评估 `test`。仓库中既有的一次 `test` 结果属于选定的 **clean BERT 基础模型**：Accuracy .6657、Macro-F1 .6025、MAE .6771、Pearson .6639，不能当成 R3 的独立测试结果。R3 的紧凑参数由 BERT 基础权重、基础模型 8 位量化包和 1.85 MiB 局部模块增量共同还原；紧凑版验证集 clean 指标为 .6415/.6164/.6299/.6153，混合缺失为 .6209/.5931/.6688/.5625。量化版与原始参数指标分别报告。

## 复核文件

- 六组完整表：`outputs/q2_best_ablation/ablation_table.csv`。
- 各组 `metrics.json`、`train_history.csv`、clean/missing 预测和混淆矩阵：`outputs/q2_best_ablation/R*/`。
- 最终 R3 的原始和紧凑附件 3 预测、紧凑验证结果：`outputs/q2_best_ablation/R3_reconstruction/`。
- 四张论文图、中文图注与数据来源：`figures/`；绘图脚本为 `make_robustness_figures.py`。
- 训练与消融代码：`missing_train.py`、`summarize_best_ablation.py`、`models/`、`data/`；可交付参数还原与无标签推理：`compact_robust_model.py`。
- 从赛题数据重新训练和从压缩包参数复核的命令：仓库根目录 `REPRODUCE_Q1_Q2.md`。
