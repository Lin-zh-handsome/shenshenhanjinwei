# Q3 v2 B0–B5 消融

[自动论文表](../tables/table_q3_ablation.md)逐行读取 `E题/Q3/outputs/q3_v2/ablation_v2.csv`。B0 无证据门；B1 加上下文证据门，是最终预测主干；B2 加绝对时间位置，B3 加门控感知池化，B4 加轻量正则，B5 在 B4 上添加仅诊断 Router（路由器）。**最终是 B1 + 单独拟合的诊断 Router，不等于 B5**。

分类、回归和解释忠实性须并列看。B3 分类局部更好，但回归 MAE 与 keep-only（仅保留证据）表现更差；B2 未带来稳定增益。最终选择和预算依据见 `E题/Q3/Q3_FAITHFULNESS_V2_REPORT.md`。旧 A0–A5 位于 `outputs/q3/`，只能作历史背景。
