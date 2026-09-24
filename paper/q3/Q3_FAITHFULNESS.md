# Q3 最终解释忠实性

验证集、最终 B1 + 诊断 Router、证据预算 0.25 的四种条件，见 [自动生成表](../tables/table_q3_faithfulness.md)：Full（完整输入）、Keep-only Evidence（仅保留证据）、Remove Evidence（删除证据）、Random Remove（匹配随机删除）。Source: `E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json`。

Top vs Random（关键证据对比随机片段）：同一 JSON 的 `confidence_drop_top` 与 `confidence_drop_random`，以及 `macro_f1_drop_top` 与 `macro_f1_drop_random`。可说关键片段在该验证设定下对分类影响更大，但不能声称每条样本都如此。

回归：删除关键证据后的 MAE 比 Full 小幅上升，JSON 的 `regression_comprehensiveness_failed=false`；Random Remove 的 MAE 反而低于 Full。结论应限定为最终 B1 的方向符合预期、幅度较小。Keep-only 保留部分预测能力，不能说证据充分覆盖全部预测。

`paper/generated/q3_faithfulness_plot.csv` 每行记录原始来源，供绘图。B2–B5 和不同预算的历史比较在 `E题/Q3/outputs/q3_v2/faithfulness/`，见 [原报告](../../E题/Q3/Q3_FAITHFULNESS_V2_REPORT.md)。
