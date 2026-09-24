# Q2 最终 R3 缺失专项分析

四张候选论文表均由脚本直接读取 R3 既有 validation（验证集）CSV，不重算模型预测。固定人工遮挡的条件与 mixed missing 综合场景不完全相同，引用时写明。

1. [缺失模态](../tables/table_q2_missing_type.md)：T、A、V、TA、TV、AV；Source: `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/missing_modality_analysis.csv`。
2. [缺失比例](../tables/table_q2_missing_ratio.md)：10%–50%；Source: `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/missing_ratio_analysis.csv`。
3. [缺失位置](../tables/table_q2_missing_position.md)：early/middle/late（前/中/后）；Source: `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/missing_position_analysis.csv`。
4. [缺失时长](../tables/table_q2_missing_duration.md)：short/medium/long（短/中/长）；Source: `E题/Q2/outputs/q2_best_ablation/R3_reconstruction/missing_span_length_analysis.csv`。

完整、可作图的逐表数据在 `paper/generated/q2_missing_*_plot.csv`，每行保存 `source_file`。附件 3 无标签，不在这些表中。
