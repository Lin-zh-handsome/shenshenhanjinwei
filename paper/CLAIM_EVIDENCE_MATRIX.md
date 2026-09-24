# Claim–Evidence Matrix（结论证据矩阵）

| Claim | Problem | Evidence | Source File | Safe to Write? |
|---|---|---|---|---|
| 100 条样本使用共享 50 时间窗且三个模态维度固定 | Q1 | validation 报告的样本数、shape 与 time_edges | `E题/Q1/validation_report.json` | YES |
| MFA 词边界具有人工标注级准确率 | Q1 | 无独立人工时间戳真值 | `E题/Q1/Q1_FINAL_AUDIT.md` | NO |
| R3 提升固定人工混合缺失下的分类鲁棒性 | Q2 | R0 与 R3 的 missing Accuracy、Macro-F1；同一 validation 设定 | `E题/Q2/outputs/q2_best_ablation/ablation_table.csv` | YES |
| Span Geometry 单独提高缺失分类 | Q2 | R2 相对 R1 的 missing Accuracy/Macro-F1 略降，回归有小幅改善 | 同上 | NO |
| 重建分支改善混合缺失分类 | Q2 | R3 相对 R2 的 missing 指标 | 同上 | YES |
| R3 改善真实附件 3 标签准确率 | Q2 | 附件 3 无标签 | `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/attachment3_predictions_compact.csv` | NO |
| Q3 证据门提升预测 | Q3 | B0→B1 分类指标小幅增加；单 seed、回归近似持平 | `E题/Q3/outputs/q3_v2/ablation_v2.csv` | PARTIAL |
| Q3 关键证据比匹配随机片段更影响分类 | Q3 | Top vs Random 的置信度下降、Macro-F1 下降 | `E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json` | YES |
| Q3 删除证据显著恶化回归 | Q3 | MAE 仅小幅上升；无显著性分析 | 同上 | PARTIAL |
| Q3 只保留证据即可完整保留预测 | Q3 | keep-only Macro-F1 低于 Full | 同上 | NO |
| 文本是当前验证集最多样本的主模态 | Q3 | 最终逐样本 LOMO 聚合；限定在该验证集 | `paper/generated/q3_modality_importance_plot.csv` | YES |
| Router 权重就是正式模态贡献 | Q3 | Router 仅诊断；正式贡献来自 LOMO | `E题/Q3/models/fer_msa.py` | NO |
| 附件 4 秒数和文本片段是真实时间标注 | Q3 | 仅均匀网格近似回投 | `E题/Q3/outputs/q3_v2/attachment4/attachment4_mapping_notes.csv` | NO |

YES 只表示可在列明的场景和限定条件下写；不推导跨数据集、跨 seed 或因果显著性。缺失场景由固定人工遮挡得到。
