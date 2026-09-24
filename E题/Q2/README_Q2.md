# Q2 当前最佳可部署分类模型

当前选定版本是 `Deployable_BERT_Last4`，以附件 2 官方训练集训练，在验证集上按预先使用的分类排序规则选择 epoch 25 参数。它读取 `text_bert/audio/vision`，可在无官方 768 维文本 teacher（教师特征）的附件 3 上推理。其分类指标高于当前其他已完成的可部署单模型，但回归 MAE、Pearson 低于冻结 BERT 的 Run A；“最佳”在此专指验证集分类排序，并非四指标同时最优。

## 代码与结果位置

| 用途 | 路径 |
|---|---|
| 两阶段训练的共用模型与训练器 | `diagnostics/oracle_text.py` |
| Run A：官方文本特征训练配置 | `diagnostics/oracle_text.yaml` |
| BERT 最后 4 层微调配置 | `score_push/bert_last4.yaml` |
| 附件 3 最佳模型推理 | `score_push/infer_attachment3_best.py` |
| 紧凑参数还原与验证 | `score_push/compact_best_model.py` |
| 选定模型的验证和既有 test 结果 | `outputs/q2_score_push/deployable_bert/partial_last4/` |
| 论文附件与紧凑参数 | `../../paper_package/Q2/` |
| 全部 clean 消融 | `SCORE_PUSH_REPORT.md`、`outputs/q2_score_push/score_push_ablation.csv` |

原始训练用 `bert-base-uncased` 的最后 4 层微调、128 维三模态投影、每模态一层时序 Transformer（变换器）、拼接融合、一层融合 Transformer、有效位置注意力池化、三分类头与强度回归头。分类损失是类别加权交叉熵，回归损失是 SmoothL1，权重为 0.5。标签顺序为 Negative=0、Neutral=1、Positive=2，预测强度约束在 `[-3,3]`。该版本没有启用后来尚未完成验证的 Span Geometry（缺失区间几何）、Temporal Reconstruction（时序重建）或 Reliability Fusion（可靠性融合）。

| 划分 | 样本 | Accuracy | Macro-F1 | Weighted-F1 | MAE | Pearson | Neutral F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 728 | 0.6429 | 0.6169 | 0.6383 | 0.6297 | 0.6159 | 0.4882 |
| test（既有一次） | 727 | 0.6657 | 0.6025 | 0.6531 | 0.6771 | 0.6639 | 0.3516 |

`best_joint.pt` 的旧文件名容易误导：它按照 `0.55 × Macro-F1 + 0.45 × Accuracy` 的验证集分类排序保存，不是赛题官方四指标联合最优。原始参数文件约 419 MB，未放入 50 MB 竞赛附件。`paper_package/Q2/model_parameters/best_bert_last4_compact.pt` 是需要公开基础 BERT 权重的 29.96 MiB 量化参数；它在验证集复核得到 0.6415 Accuracy、0.6164 Macro-F1、0.6299 MAE、0.6153 Pearson，应与原始结果分开报告。

附件 3 的 30 条无标签预测保存在 `outputs/q2_score_push/deployable_bert/partial_last4/attachment3_predictions_best_bert.csv`，材料包同时有紧凑版预测。附件 3 不能计算分类或回归指标。

完整命令、数据路径和环境见仓库根目录的 `REPRODUCE_Q1_Q2.md`。`README_Q2_LEGACY_SRF.md` 与 `outputs/q2/` 保存旧版连续缺失实验；它们的鲁棒性结果不属于当前选定的 BERT 模型。当前模型尚无可归属的连续缺失类型/缺失率消融，论文须如实标注这一限制。
