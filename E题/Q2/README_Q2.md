# Q2 最佳验证集模型

主分支只保留从最佳 clean 基础模型 `Deployable_BERT_Last4` 出发的代码、结果和 R0–R5 连续局部缺失消融。按事先定义的 clean/missing 验证集分类综合排序，最终选择 **R3：BERT 后四层微调基础模型 + 连续缺失训练 + Span Geometry Encoding（缺失区间几何编码） + Temporal Reconstruction（局部时序重建）**。R4 的 Reliability Fusion（可靠性融合）及 R5 的 Clean-Masked Consistency（干净与遮挡输入一致性）仅作消融，未放入最终模型。

| 模型与场景 | Accuracy | Macro-F1 | Weighted-F1 | MAE | Pearson | Neutral F1 |
|---|---:|---:|---:|---:|---:|---:|
| R3 原始参数，clean valid | .6429 | .6169 | .6383 | .6297 | .6159 | .4882 |
| R3 原始参数，混合缺失 valid | .6195 | .5920 | .6134 | .6686 | .5631 | .4498 |
| R3 紧凑参数，clean valid | .6415 | .6164 | .6374 | .6299 | .6153 | — |
| R3 紧凑参数，混合缺失 valid | .6209 | .5931 | .6147 | .6688 | .5625 | — |

这里的 `valid` 是附件 2 官方验证集 728 条；混合缺失为固定人工遮挡，按样本轮换缺失 T/A/V/TA/TV/AV，有效区间中间连续遮挡 30%。完整的六组结果、各类 precision（精确率）/recall（召回率）/F1、混淆矩阵以及按缺失模态、比例、位置和长度的专项表，见 [消融报告](Q2_BEST_MODEL_ABLATION.md) 与 `outputs/q2_best_ablation/`。原始基础模型的既有一次 clean test 结果单独保存在 `outputs/q2_score_push/deployable_bert/partial_last4/`；**R3 没有重新跑 test**，论文不能把该 test 数值称为 R3 独立测试结果。

## 主要文件

| 文件 | 用途 |
|---|---|
| `diagnostics/oracle_text.py`、`diagnostics/oracle_text.yaml` | 官方 768 维文本 teacher（教师特征）第一阶段训练 |
| `score_push/bert_last4.yaml` | 真实预训练 BERT 最后四层微调配置 |
| `missing_train.py`、`configs/best_*.yaml` | 在已选定 BERT 权重上完成 R0–R5 消融 |
| `models/`、`data/` | 最佳基础网络、连续缺失掩码、几何与重建实现 |
| `summarize_best_ablation.py` | 从已保存结果生成六组消融表 |
| `score_push/compact_best_model.py` | 加载 8 位紧凑的基础 BERT 微调参数 |
| `compact_robust_model.py` | 加载 R3 局部模块增量、复核验证集与推理附件 3 |
| `outputs/q2_best_ablation/R3_reconstruction/` | R3 原始/紧凑验证、专项表、30 条附件 3 预测 |
| `../../paper_package/Q2/model_parameters/` | 小于 50 MiB 材料包中的基础紧凑参数与 R3 模块增量 |

基础模型选模只看验证集 `0.55 × Macro-F1 + 0.45 × Accuracy`；R0–R5 选模看 clean 与 mixed missing 两个验证场景的同一分类分数均值。缺失模块训练时冻结基础 BERT 与下游网络，以防止已有 clean 最佳模型被随意改写。R3 的原始完整 checkpoint 约 420 MB，未入 Git；材料包提供约 29.96 MiB 的基础量化参数与约 1.85 MiB 的局部模块增量。还原时需公开的 `bert-base-uncased` 基础权重，量化后指标应与原始指标分别引用。

附件 3 无标签，30 条预测仅可用于提交与定性检查。附件 3 没有官方逐时间步缺失 mask；输出中的 zero-derived missing candidates（零值推断缺失候选）来自有效区间内的零值规则，不能称为缺失真值，也不能从附件 3 算 Accuracy 或 F1。完整数据、命令和论文引用边界见仓库根目录 [复现指南](../../REPRODUCE_Q1_Q2.md)。

四张可用于论文的缺失模态、比例、位置、长度对照图在 `figures/`，中文图注和数据来源见 `figures/FIGURE_CAPTIONS.md`；运行 `python make_robustness_figures.py` 可从归档 CSV 重新生成 PDF/SVG/PNG。
