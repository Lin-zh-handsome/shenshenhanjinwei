# Q1 与 Q2 复现指南

本指南对应主分支中已经完成的 Q1 全量特征工程，以及 Q2 验证集选出的 BERT 后四层微调模型。所有命令在 Linux 服务器上运行；原始赛题数据、公开基础预训练权重需由使用者分别准备。不要把 `Q1` 自生成的 `(50,128)/(50,74)/(50,52)` 特征直接喂给 Q2 模型：Q2 使用附件 2 的 `aligned_50.pkl`，其文本/音频/视觉维度为 `(50,768)/(50,74)/(50,35)`。

## 1 数据与环境

| 项目 | 所需材料 | 用途 |
|---|---|---|
| Q1 | 附件 1 的 100 条原始视频与 `label-100.xlsx` 中的转写 | 重新生成 `q1_features.npz` 及审计表 |
| Q1 基础模型 | `face_landmarker.task`、YuNet ONNX、MFA 英语声学/词典模型、通用 `bert-base-uncased` | 特征提取与强制对齐；不以额外情感数据训练 |
| Q2 | 附件 2 的 `aligned_50.pkl`，保留原 `train/valid/test` | 两阶段训练和验证；仅最终冻结后评估 test |
| Q2 专项推理 | 附件 3 的 aligned 文件目录 | 30 条无标签预测 |
| Q2 基础模型 | 公开 `bert-base-uncased` 权重 | 还原紧凑参数或重新训练 |

服务器原实验使用 Python 3.11.16。Q1 实际软件版本在 `E题/Q1/software_versions.json`；Q2 记录为 PyTorch 2.12.1+cu130、Transformers 5.17.0、NumPy 2.4.6、scikit-learn 1.9.1、PyYAML 6.0.3。已有环境可直接使用；需补装 Q2 Python 依赖时，在 `paper_package/Q2/code` 运行：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

CUDA 与 PyTorch 的匹配安装应依服务器驱动确定；上面的通用依赖列表没有固定 CUDA wheel（安装包）。

## 2 Q1 从附件 1 复现 100 条特征

先按 `E题/Q1/README_Q1.md` 准备原始视频及基础模型目录，然后设置三个路径；`Q1_DATA_DIR` 是原始 100 条视频目录，`Q1_MODEL_DIR` 放 Q1 基础模型，`Q1_BASE_DIR` 是可写的运行目录：

```bash
cd 'E题/Q1'
conda activate math
export Q1_DATA_DIR='/path/to/附件1原始视频目录'
export Q1_MODEL_DIR='/path/to/q1_models'
export Q1_BASE_DIR='/path/to/q1_run'
python run_q1.py --stage all
python validate_q1.py --data-dir "$Q1_DATA_DIR" --result-dir "$Q1_BASE_DIR/results"
python make_q1_report.py --data-dir "$Q1_DATA_DIR" --result-dir "$Q1_BASE_DIR/results"
```

已完成实验的结果已直接提供：`E题/Q1/q1_features.npz`（100 条多模态特征）、`q1_manifest.csv`（每条样本的时长、特征形状、解码和告警）、`q1_quality_report.csv`（每条样本的 50 个时间窗质量）、`q1_alignment.jsonl`（词级时间映射）、`Q1_FINAL_AUDIT.md` 与 `figures/`。预期形状为文本 `(100,50,128)`、音频 `(100,50,74)`、视觉 `(100,50,52)`，`validation_report.json` 记录原运行 100 条样本状态为 PASS。材料未包含赛题原视频或 Q1 预训练权重。

## 3 Q2 使用包内参数复核验证集

此路径无需重新训练最佳模型。首先准备公开 `bert-base-uncased` 基础权重；紧凑参数只保存微调后的最后四层和下游参数。进入包内代码目录：

```bash
cd 'paper_package/Q2/code'
PYTHONPATH=. python score_push/compact_best_model.py \
  --compact-path ../model_parameters/best_bert_last4_compact.pt \
  --bert-model-path /path/to/bert-base-uncased \
  --valid-pkl /path/to/aligned_50.pkl \
  --validation-report ../results/compact_validation_metrics_recheck.json
```

这复核的是**量化后的紧凑版**，预期验证集约为 Accuracy 0.6415、Macro-F1 0.6164、MAE 0.6299、Pearson 0.6153。原始 epoch 25 参数文件约 419 MB，超出比赛 50 MB 附件限制；其未量化的验证指标为 0.6429、0.6169、0.6297、0.6159，见 `results/metrics.json`。两套指标不可混写。

## 4 Q2 从附件 2 重新训练

包内保留训练器与两份配置。先编辑 `paper_package/Q2/code/diagnostics/oracle_text.yaml` 的 `aligned_pkl`，然后编辑 `paper_package/Q2/code/score_push/bert_last4.yaml` 的 `aligned_pkl`、`bert_model_path`。两份配置默认使用相对输出目录；第二阶段的 `init_checkpoint` 指向第一阶段生成的 `outputs/q2_v2/01_run_a_oracle_text/best_macro_f1.pt`。

```bash
cd 'paper_package/Q2/code'
PYTHONPATH=. python diagnostics/oracle_text.py --config diagnostics/oracle_text.yaml
PYTHONPATH=. python diagnostics/oracle_text.py --config score_push/bert_last4.yaml
```

第一阶段使用附件 2 已给的 `text (50,768)` 官方特征训练 Run A。第二阶段以真实预训练 BERT 的 `text_bert (3,50)` 为输入，解冻 BERT 最后四层并加载 Run A 的下游模型。训练只用官方 `train`，结构、epoch 与参数选择看官方 `valid`。旧文件名 `best_joint.pt` 依据 `0.55 × Macro-F1 + 0.45 × Accuracy` 排序，是本实验的分类选择规则，不是赛题官方四指标联合评分。训练数据不包含附件 3 标签，也不引入额外情感数据。

由于环境、底层运算和随机性差异，重训数值可能与归档结果略有差别。论文中的已完成实验数值应以 `paper_package/Q2/results/` 中的原始归档文件为准，不用重新查看 test 选择参数。既有 test 结果保存在 `test_metrics.json`、`test_class_metrics.csv`、`test_confusion_matrix.csv` 和 `test_predictions.csv`；本次材料整理未重评 test。

## 5 Q2 附件 3 无标签推理

使用包内紧凑参数：

```bash
cd 'paper_package/Q2/code'
PYTHONPATH=. python score_push/infer_attachment3_best.py \
  --compact-path ../model_parameters/best_bert_last4_compact.pt \
  --bert-model-path /path/to/bert-base-uncased \
  --attachment3-dir /path/to/附件3/对齐版本 \
  --output ../results/attachment3_predictions_compact_recheck.csv
```

该推理仅要求每条样本含 `text_bert/audio/vision`，不要求标签或官方文本 teacher。已归档的原始参数预测为 `attachment3_predictions_best_bert.csv`（30 行），紧凑参数预测为 `attachment3_predictions_compact.csv`（30 行）；两者类别一致，强度值最大绝对差约 0.0235。附件 3 无标签，不能从这两份文件计算 Accuracy、F1、MAE 或 Pearson。

## 6 比赛附件与论文引用

`Q1_Q2_paper_package.zip` 为 31.80 MiB，低于赛题规定的 50 MB。其 `Q1/` 包含 100 条原始提取特征、逐样本与逐窗报告和代码；`Q2/` 包含当前最佳分类模型代码、配置、紧凑参数、实验 CSV/JSON 和说明。附件中不含参赛队身份字段。论文正文仍须呈现 Q1 方法与全量结果、Q2 已完成模型的结构/损失/验证指标/错误分析，并明确：**最佳 BERT 模型尚未完成连续局部缺失鲁棒性验证**。旧 SRF-MSA 的鲁棒性表与最佳 BERT 结果属于不同模型，不得拼成同一模型的消融结论。
