# Q2 连续局部缺失图注与数据来源

四张图均使用附件 2 官方验证集的 728 条样本，比较同一基础 BERT 权重上的 R0（未进行连续缺失训练）与最终 R3（连续缺失训练、区间几何、局部重建）。纵轴为 Macro-F1（宏平均 F1）。结果来自单一 seed 42 和确定性的人工连续区间遮挡；柱高与折线点是全验证集指标，不是跨 seed 均值，因此没有误差条，也不表示统计显著性。模型和 checkpoint（参数文件）的选择只使用验证集，图中不含 `test` 或附件 3 指标。

1. **`fig_missing_type`：缺失模态组合。** 每条样本在有效区间中部遮挡长度约 30% 的连续区间，分别遮挡 T（文本）、A（音频）、V（视觉）、T+A、T+V、A+V。文本缺失对两种模型的影响普遍较大。数据：`results/ablation/R0_selected_clean/missing_modality_analysis.csv` 和 `R3_reconstruction` 下同名文件。
2. **`fig_missing_ratio`：缺失比例。** 同时遮挡三模态的中部连续区间，比例为 10%、20%、30%、40%、50%。R3 的 Macro-F1 在五个比例上均高于 R0，但两者随缺失扩大而下降。数据：两组的 `missing_ratio_analysis.csv`。
3. **`fig_missing_position`：缺失位置。** 同时遮挡三模态 30% 的连续区间，比较 early（前段）、middle（中段）、late（后段）。R3 在中段增益最大，前段不优于 R0。数据：两组的 `missing_position_analysis.csv`。
4. **`fig_missing_span_length`：区间长度。** 同时遮挡三模态中部的 4、10、20 个有效位置。R3 在三种长度下均高于 R0，但长度 20 时仍只有约 0.396 Macro-F1。数据：两组的 `missing_span_length_analysis.csv`。

每幅图提供可编辑文字的 PDF/SVG 和 300 dpi PNG 预览；源脚本为 `make_robustness_figures.py`。图中不能把人工遮挡指标解释为附件 3 的真实缺失测评。
