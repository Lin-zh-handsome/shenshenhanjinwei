# Q1 与 Q2 复现指南

本指南对应主分支当前文件树：Q1 的全量 100 条特征与 Q2 的最佳 BERT 基础模型、R0–R5 连续局部缺失消融。全部训练和评估在 Linux GPU 服务器进行。赛题原始附件及公开预训练基础模型由复现者自行放在指定路径；不得混用 Q1 自生成 `(50,128)/(50,74)/(50,52)` 与 Q2 附件 2 的 `aligned_50.pkl` 接口。

## 1. 环境与数据

Q2 原运行环境：Python 3.11.16、PyTorch 2.12.1+cu130、Transformers 5.17.0、NumPy 2.4.6、scikit-learn 1.9.1、PyYAML 6.0.3；实际服务器 GPU 为 RTX 3090 24 GB。Q1 软件记录见 `E题/Q1/software_versions.json`。建议在独立 Conda 环境安装与服务器 CUDA 驱动匹配的 PyTorch；其余 Q2 依赖可用国内源：

```bash
cd 'paper_package/Q2/code'
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

所需路径：

| 变量 | 内容 |
|---|---|
| `Q1_DATA_DIR` | 附件 1 的 100 条视频和官方转写表所在目录 |
| `Q1_MODEL_DIR` | Q1 所需的 face landmarker、YuNet、MFA、公开 BERT 基础模型 |
| `Q2_ALIGNED_PKL` | 附件 2 `aligned_50.pkl`，原始 `train/valid/test` 划分不改 |
| `BERT_MODEL_DIR` | 公开 `bert-base-uncased` 的本地权重与 tokenizer |
| `ATTACHMENT3_DIR` | 附件 3 的 `对齐版本` 30 个无标签 `.pkl` 文件 |

Q2 分类为 Negative=0、Neutral=1、Positive=2；`text_bert` 为 `[3,50]` 的 token ID、attention mask（注意力掩码）与 token type ID。若存储 dtype（数据类型）为 float，读取器检查近似整数后取整转 `long`，不会把 token ID 当连续特征。音频为 `[50,74]`，视觉为 `[50,35]`。

## 2. Q1 重新生成 100 条特征

按 `E题/Q1/README_Q1.md` 放好原始视频与基础模型，在服务器运行：

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

仓库已有 `E题/Q1/q1_features.npz`、`q1_manifest.csv`、`q1_quality_report.csv`、`q1_alignment.jsonl`、四张图及 `Q1_FINAL_AUDIT.md`。数据集为 100 条，统一 50 个真实时间窗；特征形状文本 `(100,50,128)`、音频 `(100,50,74)`、视觉 `(100,50,52)`。已有报告的质量统计与警告须与论文一起引用，尤其不能把词级边界覆盖率称为人工时间戳准确率。

## 3. 直接复核 Q2 论文材料包参数

材料包以公开 BERT 基础权重、8 位量化的基础模型参数和 R3 的局部模块增量还原可运行模型，合计参数文件约 31.8 MiB。下列命令**只读取附件 2 的 valid**，不调用 test：

```bash
cd 'paper_package/Q2/code'
PYTHONPATH=. python compact_robust_model.py \
  --base-compact ../model_parameters/best_bert_last4_compact.pt \
  --delta-path ../model_parameters/best_robust_delta.pt \
  --bert-model-path /path/to/bert-base-uncased \
  --aligned-pkl /path/to/aligned_50.pkl \
  --validation-report ../results/rechecked_compact_validation.json
```

应分别得到 clean 验证集约 `.6415/.6164/.6299/.6153`、固定混合缺失验证集约 `.6209/.5931/.6688/.5625`（顺序为 Accuracy/Macro-F1/MAE/Pearson）。这些是**紧凑参数版**，与报告的 R3 原始参数 `.6429/.6169/.6297/.6159` 和 `.6195/.5920/.6686/.5631` 分开引用。压缩包保留了六组原始 `metrics.json`、`train_history.csv`、逐样本预测与分组表。

## 4. 从附件 2 完整重新训练

在 `paper_package/Q2/code` 或仓库 `E题/Q2` 运行。先将 `diagnostics/oracle_text.yaml`、`score_push/bert_last4.yaml`、`configs/best_*.yaml` 中的 `/path/to/...` 改为本机路径；R0–R5 的 `init_checkpoint` 指向第二阶段选出的 `best_joint.pt`。第一阶段的官方连续 `text` 特征仅作训练教师，不用于最终附件 3 推理。

```bash
cd 'E题/Q2'
PYTHONPATH=. python diagnostics/oracle_text.py --config diagnostics/oracle_text.yaml
PYTHONPATH=. python diagnostics/oracle_text.py --config score_push/bert_last4.yaml
PYTHONPATH=. python missing_train.py --config configs/best_r0_selected_bert.yaml
for cfg in configs/best_R1_missing_aug.yaml \
           configs/best_R2_span_geometry.yaml \
           configs/best_R3_reconstruction.yaml \
           configs/best_R4_reliability.yaml \
           configs/best_R5_consistency.yaml; do
  PYTHONPATH=. python missing_train.py --config "$cfg"
done
PYTHONPATH=. python summarize_best_ablation.py
MPLBACKEND=Agg python make_robustness_figures.py
```

模型选择仅使用 `valid`。R0 必须复现第二阶段的 clean 验证指标，作为初始化对齐依据。R1–R5 仅对训练集人工生成连续缺失，验证集固定遮挡以保证可比；`sequence_valid_mask`（真实序列位置）、`observed_mask`（当前有观测）与 `synthetic_missing_mask`（人工遮挡）分开传递。六组结果见 `outputs/q2_best_ablation/ablation_table.csv`，每组都保存最优分类 checkpoint 与完整曲线。重训因随机性和环境可能略有差异，不应用 test 选择结构、epoch 或 seed。

绘图脚本从 R0/R3 专项 CSV 生成四张单幅 PDF/SVG/PNG 对照图，分别展示缺失模态、比例、位置和区间长度；图注与统计边界见 `figures/FIGURE_CAPTIONS.md`。在材料包的 `Q2/code` 下也可运行该脚本，默认读取 `../results/ablation/`；使用 `--output-dir ../figures` 可覆盖包内图片。

若要为重训结果制作同样的紧凑参数，先从第二阶段 `best_joint.pt` 生成基础模型包，再从 R3 `best_classification.pt` 提取局部模块增量：

```bash
PYTHONPATH=. python score_push/compact_best_model.py \
  --source-checkpoint outputs/q2_score_push/deployable_bert/partial_last4/best_joint.pt \
  --compact-path /path/to/best_bert_last4_compact.pt
PYTHONPATH=. python compact_robust_model.py \
  --source-checkpoint outputs/q2_best_ablation/R3_reconstruction/best_classification.pt \
  --base-compact /path/to/best_bert_last4_compact.pt \
  --delta-path /path/to/best_robust_delta.pt
```

原始完整参数约 420 MB，故不进入 50 MB 竞赛材料包。代码与配置由 Git 管理，数据路径按本机修改；实验结果归档后应明确选择版本和参数来源。

## 5. 附件 3 的 30 条无标签预测

```bash
cd 'paper_package/Q2/code'
PYTHONPATH=. python compact_robust_model.py \
  --base-compact ../model_parameters/best_bert_last4_compact.pt \
  --delta-path ../model_parameters/best_robust_delta.pt \
  --bert-model-path /path/to/bert-base-uncased \
  --attachment3-dir /path/to/附件3/对齐版本 \
  --predictions-output ../results/attachment3_predictions_recheck.csv
```

归档结果为 `results/ablation/R3_reconstruction/attachment3_predictions_raw.csv` 和 `attachment3_predictions_compact.csv`。附件 3 没有缺失真值或情感标签；程序对有效区间内的零特征构造 zero-derived missing candidates（零值推断缺失候选），预测 CSV 写出各模态候选数，不把它们当作官方标注。不能计算附件 3 Accuracy、F1、MAE 或 Pearson。

## 6. 论文引用边界

Q2 最终选用 R3，是因为它在预定义的 clean/missing 验证集分类综合排序中最高；R4/R5 消融不能冒充最终结构。最终报告包含单 seed 结果和缺失率 10%–50%、模态组合、早中晚位置、短中长区间的分析。现有一次 `test` 指标 `.6657/.6025/.6771/.6639` 属于此前冻结的 clean BERT 基础模型；本轮**没有**把 R3 送入 test，因此论文表格应分开列出该历史基础模型 test 与 R3 validation。附件 3 只报告 30 条无标签预测。
