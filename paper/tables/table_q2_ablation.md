# Q2 R0–R5 消融

| 实验 | 输入 | Accuracy | Macro-F1 | MAE | Pearson | 最终 |
| --- | --- | --- | --- | --- | --- | --- |
| R0_selected_clean | clean | 0.6429 | 0.6169 | 0.6297 | 0.6159 | NO |
| R0_selected_clean | missing | 0.5797 | 0.5627 | 0.6716 | 0.5445 | NO |
| R1_missing_aug | clean | 0.6429 | 0.6169 | 0.6297 | 0.6159 | NO |
| R1_missing_aug | missing | 0.5962 | 0.5744 | 0.6594 | 0.5656 | NO |
| R2_span_geometry | clean | 0.6429 | 0.6169 | 0.6297 | 0.6159 | NO |
| R2_span_geometry | missing | 0.5948 | 0.5721 | 0.6528 | 0.5667 | NO |
| R3_reconstruction | clean | 0.6429 | 0.6169 | 0.6297 | 0.6159 | YES |
| R3_reconstruction | missing | 0.6195 | 0.5920 | 0.6686 | 0.5631 | YES |
| R4_reliability | clean | 0.6291 | 0.6072 | 0.6251 | 0.6155 | NO |
| R4_reliability | missing | 0.6168 | 0.5909 | 0.6639 | 0.5621 | NO |
| R5_consistency | clean | 0.6291 | 0.6075 | 0.6252 | 0.6153 | NO |
| R5_consistency | missing | 0.6209 | 0.5965 | 0.6657 | 0.5698 | NO |

Source:
- `E题/Q2/outputs/q2_best_ablation/ablation_table.csv`

R5 的某些单项缺失指标更高；最终选择依据见 `E题/Q2/Q2_BEST_MODEL_ABLATION.md`。
