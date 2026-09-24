# Q2 R0–R5 消融说明

[自动生成的论文表](../tables/table_q2_ablation.md)与 [CSV](../tables/q2_ablation.csv)逐行读取 `E题/Q2/outputs/q2_best_ablation/ablation_table.csv`，每组分 clean（完整）与 missing（缺失）两行，指标为 Accuracy、Macro-F1、MAE、Pearson，标出最终 R3。

R0：已选 clean BERT；R1：连续缺失训练；R2：区间几何；R3：局部重建；R4：可靠性残差；R5：一致性。R2 相对 R1 的缺失分类略降；R4/R5 的某些单项更好，但最终选择依 `E题/Q2/Q2_BEST_MODEL_ABLATION.md` 的预设综合规则。失败或未采用消融仍完整保留。
