# Q2 已完成模型与结果说明

## 模型身份与数据

此目录对应 `Deployable_BERT_Last4`（可部署的 BERT 后四层微调模型），原始代码来自仓库提交 `e723fb9`。附件 2 使用官方 `aligned_50.pkl` 划分：训练 3395 条、验证 728 条、测试 727 条，保持原划分。`text_bert` 为 `[3,50]` 的 token ID、attention mask（注意力掩码）和 token type ID；音频为 `[50,74]`，视觉为 `[50,35]`。标签映射为 Negative=0、Neutral=1、Positive=2，强度范围为 `[-3,3]`。

基础模型为 `bert-base-uncased`，仅微调最后 4 层；三个模态分别投影到 128 维，经过各自一层 temporal Transformer（时序变换器），拼接融合后再经过一层融合 Transformer、有效位置注意力池化，输出三分类与强度回归。该已完成版本的 `use_position=false`；不能在论文中声称它使用了后续尚未确定的时间位置编码、区间几何、重建或可靠性门控。分类使用带类别权重的交叉熵，回归使用 SmoothL1，总损失为 `L = L_cls + 0.5 L_reg`。训练配置原样见 `code/score_push/bert_last4.yaml`；其中数据和模型路径改成待填写占位符。

运行记录环境：Python 3.11.16、PyTorch 2.12.1+cu130、Transformers 5.17.0、NumPy 2.4.6、scikit-learn 1.9.1、PyYAML 6.0.3。若需安装包，可使用国内源：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r code/requirements.txt
```

## 原始结果与选择原则

验证集上，选定 checkpoint（模型参数文件）为 epoch 25 的 `best_joint.pt`。旧文件名中的 `joint` 指 `0.55 × Macro-F1 + 0.45 × Accuracy` 分类排序；不是赛题官方四指标联合评分。原始 419 MB checkpoint 保存在原服务器实验目录，未放入 50 MB 附件。

| 数据划分 | 样本 | Accuracy | Macro-F1 | Weighted-F1 | MAE | Pearson | Neutral F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 728 | 0.6429 | 0.6169 | 0.6383 | 0.6297 | 0.6159 | 0.4882 |
| test（既有一次评估） | 727 | 0.6657 | 0.6025 | 0.6531 | 0.6771 | 0.6639 | 0.3516 |

这轮打包没有重新评估 test。Neutral 是主要错误来源：既有 test 中 158 条 Neutral 有 81 条被预测为 Positive。与 Oracle Text（直接使用附件 2 的官方 768 维文本特征）相比，last4 模型的 validation Accuracy/Macro-F1 略高，但 MAE/Pearson 较差；Oracle 的四指标为 0.6387/0.6126/0.6066/0.6421。完整结构对比见 `SCORE_PUSH_REPORT.md` 与 `results/score_push_ablation.csv`。

## 50 MB 内的参数还原

`model_parameters/best_bert_last4_compact.pt` 只保存后四层 BERT 的逐行 int8（8 位整数量化）权重、其余小张量的浮点权重及下游模型权重。运行时需先取得公开的 `bert-base-uncased` 基础模型，再由 `code/score_push/compact_best_model.py` 还原。紧凑包不是原始 checkpoint 的无损压缩；独立验证集复核见 `results/compact_validation_metrics.json`：Accuracy 0.6415、Macro-F1 0.6164、MAE 0.6299、Pearson 0.6153。相比原始版，validation 中约一条分类预测改变。

从本目录运行紧凑参数验证：

```bash
cd code
PYTHONPATH=. python score_push/compact_best_model.py \
  --compact-path ../model_parameters/best_bert_last4_compact.pt \
  --bert-model-path /path/to/bert-base-uncased \
  --valid-pkl /path/to/aligned_50.pkl \
  --validation-report ../results/compact_validation_metrics_recheck.json
```

附件 3 无标签推理只需 `text_bert/audio/vision`：

```bash
cd code
PYTHONPATH=. python score_push/infer_attachment3_best.py \
  --compact-path ../model_parameters/best_bert_last4_compact.pt \
  --bert-model-path /path/to/bert-base-uncased \
  --attachment3-dir /path/to/附件3/对齐版本 \
  --output ../results/attachment3_predictions_compact_recheck.csv
```

`results/attachment3_predictions_best_bert.csv` 是原始 checkpoint 的 30 条预测；`results/attachment3_predictions_compact.csv` 是紧凑模型的 30 条预测，两者分类完全一致，预测强度最大绝对差约 0.0235。两份 CSV 均不是有标签的测评指标。

## 论文与复现边界

问题二要求连续局部缺失的鲁棒性分析。当前选定模型是完整三模态输入的高分基础模型，不能把旧 SRF-MSA 的缺失类型、缺失率、位置、时长实验结果归到此模型名下。论文若报告问题二的完整鲁棒性结论，必须另行完成同一模型的缺失专项验证；本包仅提供已经完成且可核查的 clean 指标、既有单次 test 与附件 3 全量推理。
