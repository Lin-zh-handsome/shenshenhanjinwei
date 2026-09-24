# Q2 clean score push from `5790875`

## Scope and evidence

This branch starts from commit `5790875`. It uses the original aligned train/validation split, official Oracle text features, and the original 128-dimensional three-modality temporal architecture. It does not run the test split, attachment 3, BERT fine-tuning, missing augmentation, reconstruction, span geometry, or reliability fusion. Selection uses validation Macro-F1 while retaining separate best Accuracy, joint, and MAE checkpoints. The full ordered results and separate metric sort orders are in `outputs/q2_score_push/score_push_summary.csv`, `score_push_by_macro_f1.csv`, and `score_push_by_accuracy.csv`.

The default `CleanMultimodalBackbone` preserved the original Run A state-dict keys and produced identical logits, regression outputs, shapes, and multitask loss when loaded with the same weights. The inverse-class-weight control also exactly reproduced the original Run A validation scores.

## Ordered validation results

| Step | Change relative to preceding structural run or selected baseline | Accuracy | Macro-F1 | Neutral F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|---:|
| A0 | Original Run A | **0.6387** | 0.6126 | 0.4524 | 0.6066 | 0.6421 |
| A1 | Enable temporal position | 0.6250 | 0.6052 | 0.4611 | 0.6091 | 0.6311 |
| A2 | A1 + text-centered fusion | 0.6085 | 0.6074 | 0.5042 | 0.6227 | 0.6196 |
| A3 | A2 + attention/max pooling | 0.6168 | 0.6046 | 0.4578 | 0.6279 | 0.6431 |
| A4 | A3 + MLP heads | 0.6223 | 0.5982 | 0.4425 | **0.5999** | 0.6346 |
| A5 W0 | Run A + inverse weights | **0.6387** | 0.6126 | 0.4524 | 0.6066 | 0.6421 |
| A5 W1 | Run A + sqrt weights | 0.6277 | 0.6036 | 0.4571 | 0.6035 | 0.6174 |
| A5 W2 | Run A + no class weights | 0.6181 | 0.6037 | 0.4752 | 0.6169 | 0.6044 |
| A6 | Inverse weights + smoothing 0.05 | 0.6113 | 0.6035 | 0.4952 | 0.6033 | 0.6223 |
| A7 | Dropout 0.20, weight decay 0.01, batch 32 | 0.6305 | **0.6194** | **0.5000** | 0.6028 | 0.6415 |
| A8 | A7 + EMA; selected raw weights | 0.6305 | **0.6194** | **0.5000** | 0.6028 | 0.6415 |

Position alone reduced Accuracy by 1.37 percentage points and Macro-F1 by 0.74 points. Relative to A1, text-centered fusion raised Macro-F1 by only 0.22 points while reducing Accuracy by 1.65 points; its Neutral F1 rose to 0.5042 but the overall regression metrics deteriorated. Attention/max pooling raised Accuracy by 0.82 points relative to A2 but reduced Macro-F1 by 0.29 points. MLP heads raised Accuracy by 0.55 points relative to A3 but reduced Macro-F1 by 0.64 points. None of these structural changes beat Run A on both classification metrics, so they are not retained in the selected clean backbone.

Inverse weights remain preferable to sqrt or no weights for both Accuracy and Macro-F1. Label smoothing 0.05 raises Neutral F1 to 0.4952 but lowers Accuracy and Macro-F1. The fixed regularization combination raises Macro-F1 by 0.0068, Neutral F1 by 0.0476, and slightly reduces MAE, while Accuracy falls by 0.0082. This is a small tradeoff, not a demonstrated stable improvement. EMA was evaluated separately from raw weights. Its extra validation pass initially changed the next epoch's shuffle RNG; the corrected run preserves the RNG state around EMA validation. Corrected raw training exactly matches A7, and the best checkpoint remains raw. EMA is not retained.

## Modality ablation on the A7 backbone

| Enabled modalities | Accuracy | Macro-F1 | Neutral F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|
| T | 0.6181 | 0.6089 | 0.4880 | 0.6311 | 0.6107 |
| T + A | 0.6181 | 0.6143 | **0.5149** | **0.5922** | **0.6483** |
| T + V | **0.6305** | 0.6172 | 0.4988 | 0.6525 | 0.6086 |
| T + A + V | **0.6305** | **0.6194** | 0.5000 | 0.6028 | 0.6415 |

Audio improves Macro-F1 and regression relative to text only. Vision improves classification relative to text only, but worsens regression. All three modalities yield the highest Macro-F1, while T+V ties for highest Accuracy and T+A has the best regression and Neutral F1. These comparisons use separately trained models with the same settings; they do not prove a causal contribution for an individual validation prediction.

## Selection, runtime, and limits

The highest Accuracy remains original Run A, checkpoint `outputs/q2_score_push/05_weight_inverse/best_accuracy.pt` and config `diagnostics/weight_inverse.yaml`. The highest Macro-F1 is A7, checkpoint `outputs/q2_score_push/06_regularized/best_macro_f1.pt` and config `diagnostics/oracle_regularized.yaml`; its best joint checkpoint is `outputs/q2_score_push/06_regularized/best_joint.pt`. A7 is the exploratory clean candidate because it improves Macro-F1, Neutral F1, and MAE with a 0.82-point Accuracy tradeoff. Its gain needs seed replication before calling it stable. The originally reported Run A checkpoint belongs to the earlier checkout and was not required for these experiments; the W0 rerun recreates the equivalent checkpoint in this branch.

| Run | Selected epoch | Training seconds | Peak PyTorch GPU allocation (MiB) | Train Accuracy − valid Accuracy |
|---|---:|---:|---:|---:|
| A1 position | 5 | 18.9 | 47.6 | 0.055 |
| A2 text-centered | 3 | 22.6 | 59.4 | 0.034 |
| A3 attention/max | 5 | 26.6 | 59.8 | 0.062 |
| A4 MLP heads | 8 | 31.2 | 60.2 | 0.120 |
| A5 inverse | 6 | 392.5 | 47.5 | 0.080 |
| A5 sqrt | 4 | 396.6 | 47.5 | 0.054 |
| A5 none | 3 | 16.5 | 47.5 | 0.048 |
| A6 smoothing | 4 | 19.8 | 47.5 | 0.045 |
| A7 regularized | 6 | 12.5 | 69.8 | 0.063 |
| A8 EMA | 6 | 13.6 | 72.5 | 0.063 |
| T | 7 | 37.4 | 52.0 | 0.118 |
| T+A | 7 | 15.9 | 60.6 | 0.098 |
| T+V | 3 | 10.6 | 61.4 | 0.014 |
| T+A+V | 6 | 13.9 | 69.8 | 0.063 |

The two class-weight runs executed concurrently and took much longer than isolated runs because of shared-server contention; their durations should not be used to compare algorithms. The GPU figures are PyTorch's peak allocated memory for the process, not total device occupancy. At A7's selected epoch, training Accuracy was 0.6940 and validation Accuracy 0.6305. The larger 0.120 gap for A4 reinforces the overfitting concern.

Neutral remains the weakest class in A7: precision 0.4518, recall 0.5598, F1 0.5000, compared with Negative F1 0.6651 and Positive F1 0.6932. The clean targets Accuracy ≥ 0.67 and Macro-F1 ≥ 0.64 were not reached. No test metric is reported. The largest positive whole-model change in this branch is the fixed regularization combination's +0.0068 Macro-F1, with a concurrent -0.0082 Accuracy. Further score gains require new validation evidence; this branch does not change SRF-MSA or enter missing-robustness experiments.
