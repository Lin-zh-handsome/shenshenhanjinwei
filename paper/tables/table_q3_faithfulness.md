# Q3 最终模型解释忠实性

| 条件 | Accuracy | Macro-F1 | MAE | Pearson | 置信度下降 |
| --- | --- | --- | --- | --- | --- |
| Full | 0.6332 | 0.6194 | 0.6149 | 0.6576 |  |
| Keep-only Evidence | 0.5398 | 0.4012 | 0.6722 | 0.6209 |  |
| Remove Evidence | 0.5975 | 0.5632 | 0.6178 | 0.6405 | 0.0562 |
| Random Remove | 0.6161 | 0.5814 | 0.6117 | 0.6579 | 0.0329 |

Source:
- `E题/Q3/outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_faithfulness_v2.json`

验证集、预算 0.25；回归 MAE 删除证据后仅小幅上升，不宜夸大。
