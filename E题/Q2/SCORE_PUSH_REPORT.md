# Q2 clean validation score push

All model and checkpoint choices below use the official validation split. The test split was not evaluated in this round. The full experiment list is in `outputs/q2_score_push/score_push_ablation.csv`; predictions and per-class metrics are stored with each run.

## Results

| Model / controlled change | Accuracy | Macro-F1 | Neutral F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|
| Old SRF-MSA | 0.5055 | 0.4719 | — | 0.7510 | 0.3752 |
| Run A Oracle Text | 0.6387 | 0.6126 | 0.4524 | 0.6066 | 0.6421 |
| C1 position | 0.6250 | 0.6052 | 0.4611 | 0.6091 | 0.6311 |
| C1 position, small initialization | 0.6264 | 0.6140 | 0.4776 | 0.5955 | 0.6576 |
| C2 256-d temporal encoders | 0.6209 | 0.6024 | 0.4582 | 0.6334 | 0.6171 |
| C3 attention + max pooling | 0.6291 | 0.5890 | 0.3922 | 0.6517 | 0.6160 |
| C4 text-centered residual fusion | 0.6085 | 0.5919 | 0.4439 | 0.6128 | 0.6451 |
| C5 CE + smoothing | 0.6236 | 0.6114 | 0.4888 | 0.5981 | 0.6327 |
| C6 EMA run, selected raw model | 0.6360 | 0.5781 | 0.3472 | 0.6495 | 0.6412 |
| Deployable BERT, frozen + Run A downstream | 0.6387 | 0.6126 | 0.4524 | 0.6066 | 0.6421 |
| Deployable BERT, last 4 layers trained | **0.6429** | **0.6169** | 0.4882 | 0.6297 | 0.6159 |
| BERT + distillation, weight 0.05 | 0.6195 | 0.5950 | 0.4624 | 0.6369 | 0.6225 |
| BERT + distillation, weight 0.10 | 0.6195 | 0.5950 | 0.4624 | 0.6367 | 0.6227 |
| BERT + distillation, weight 0.20 | 0.6195 | 0.5950 | 0.4624 | 0.6363 | 0.6231 |
| Mild Neutral oversampling | 0.6016 | 0.6004 | 0.4956 | 0.6142 | 0.6440 |
| Oracle three-seed ensemble | 0.6277 | 0.6085 | 0.4492 | 0.6110 | **0.6640** |

The 256-d architecture overfit the 3,395 training examples: one run reached roughly 0.99 training accuracy while validation stayed near 0.62. Its separate temporal encoders, attention pooling variants, text-centered residual fusion, loss choices, and EMA did not produce a stable classification gain over Run A. The best individual clean classification checkpoint is the last-four-layer BERT run at `outputs/q2_score_push/deployable_bert/partial_last4/best_joint.pt`. Its small Accuracy/Macro-F1 gain comes with worse regression. For balanced classification and regression, retain the original Run A checkpoint at `outputs/q2_v2/01_run_a_oracle_text/best_macro_f1.pt` until another validation experiment improves it convincingly.

## Text bridge and modality evidence

For five training samples, the cached `bert-base-uncased` last hidden states and official 768-d text features matched to measured cosine similarity 1.0 at valid positions. Across the full validation split, replacing Run A's official text features with the frozen BERT bridge changed predictions by at most 4.9e-6 and left all reported metrics unchanged. Thus this pretrained checkpoint reproduces the Oracle text channel for the measured data. Fine-tuning the last four layers gave only a small classification gain, and the tested distillation weights 0.05, 0.10, and 0.20 reduced validation scores.

| Modalities | Accuracy | Macro-F1 | Neutral F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|
| T | 0.6223 | 0.6150 | 0.5036 | 0.6311 | 0.6173 |
| T + A | 0.6223 | 0.6163 | 0.5058 | **0.5889** | 0.6566 |
| T + V | 0.6058 | 0.6027 | 0.5045 | 0.6470 | 0.5822 |
| T + A + V | **0.6387** | 0.6126 | 0.4524 | 0.6066 | 0.6421 |

Audio improves regression and slightly improves Macro-F1 relative to text only. Vision alone lowers these metrics, while all three modalities give the highest Accuracy. These are separate trained ablations, so they show configuration-level effects rather than an isolated causal contribution of one feature at inference.

Neutral remains the weakest class in the best-accuracy model: precision 0.5321, recall 0.4511, F1 0.4882. A mild sampler targeting 30% Negative, 28% Neutral, 42% Positive raised Neutral recall to 0.6141 and F1 to 0.4956, but lowered Accuracy to 0.6016 and Macro-F1 to 0.6004; it is not selected. Three seeds 42/2026/3407 averaged Accuracy 0.6291 ± 0.0096 and Macro-F1 0.6075 ± 0.0047. Their probability ensemble improved Pearson to 0.6640 while lowering classification metrics against Run A.

The clean validation targets (Accuracy 0.67, Macro-F1 0.64) were not reached. The deployable BERT model clears the lower 0.65/0.62 gate in neither metric, so this round does not authorize restoring the missing-data modules under the stated stage rule. No test result is claimed.
