# 复现入口

本页命令对应仓库现有 CLI。论文整理本身只运行 `python paper/scripts/build_all.py` 读取归档结果；以下训练、验证、推理命令供复现者在具备数据和模型权重的环境中执行。请把 `<DATA_ROOT>`、`<OUTPUT_ROOT>` 改为本机目录；不改变附件 2 官方 train/valid/test 划分。

## Environment（环境）

Python 依赖见 [requirements-paper.txt](../requirements-paper.txt)、`E题/Q3/requirements.txt`、`paper_package/Q2/code/requirements.txt`。Q1 的完整历史软件记录为 `E题/Q1/software_versions.json`，此外需 FFmpeg、ffprobe、MFA（强制对齐）和基础模型。Q2/Q3 的 CUDA（显卡计算平台）版 PyTorch 按本机 GPU 环境安装。原始附件、公开 BERT 和 Q1 预训练模型不在仓库。

## Q1

在 Linux shell 中从仓库根目录运行；模型文件要求见 `E题/Q1/README_Q1.md`：

```bash
cd 'E题/Q1'
export Q1_DATA_DIR='<DATA_ROOT>/附件1原始视频'
export Q1_MODEL_DIR='<DATA_ROOT>/q1_models'
export Q1_BASE_DIR='<OUTPUT_ROOT>/q1_run'
python run_q1.py --stage all
python validate_q1.py --data-dir "$Q1_DATA_DIR" --result-dir "$Q1_BASE_DIR/results"
python make_q1_report.py --data-dir "$Q1_DATA_DIR" --result-dir "$Q1_BASE_DIR/results"
```

已有最终输出在 `E题/Q1/`，不需要为写论文重新处理 100 条视频。

## Q2

`configs/best_R3_reconstruction.yaml` 的 `aligned_pkl`、BERT 本地路径和 `init_checkpoint` 需指向本机实际文件；若只复核已归档紧凑参数，不需要重新训练。运行目录为 `E题/Q2`。

**训练（需要重建基础模型时）**：基础 BERT 的前两阶段及 R0–R5 完整流程见 `REPRODUCE_Q1_Q2.md`。最终 R3 已有基础权重时，实际 CLI 为：

```bash
cd 'E题/Q2'
PYTHONPATH=. python missing_train.py --config configs/best_R3_reconstruction.yaml
```

**Validation（验证）**：用已入库的紧凑基础模型和局部增量，在附件 2 的官方 valid 上复核；输出到新目录，避免覆盖原 JSON：

```bash
cd 'E题/Q2'
PYTHONPATH=. python compact_robust_model.py \
  --base-compact ../../paper_package/Q2/model_parameters/best_bert_last4_compact.pt \
  --delta-path ../../paper_package/Q2/model_parameters/best_robust_delta.pt \
  --bert-model-path '<DATA_ROOT>/bert-base-uncased' \
  --aligned-pkl '<DATA_ROOT>/aligned_50.pkl' \
  --validation-report '<OUTPUT_ROOT>/q2_compact_valid.json'
```

**Attachment3 inference（附件 3 推理）**：仓库已存最终 30 条预测；重跑命令仅供复现：

```bash
cd 'E题/Q2'
PYTHONPATH=. python compact_robust_model.py \
  --base-compact ../../paper_package/Q2/model_parameters/best_bert_last4_compact.pt \
  --delta-path ../../paper_package/Q2/model_parameters/best_robust_delta.pt \
  --bert-model-path '<DATA_ROOT>/bert-base-uncased' \
  --attachment3-dir '<DATA_ROOT>/附件3/对齐版本' \
  --predictions-output '<OUTPUT_ROOT>/attachment3_predictions.csv'
```

Q2 的原始 R3 大 checkpoint 未入库；紧凑版指标与原始模型指标须分开引用。附件 3 无标签。

## Q3

先复制 `E题/Q3/configs/q3_v2_final.yaml` 到 `<OUTPUT_ROOT>/q3_v2_local.yaml` 并填入三个 `SET_PATH_TO_...` 数据占位符及新的输出目录；原配置保持不变。运行目录为 `E题/Q3`。

```bash
cd 'E题/Q3'
python train.py --config '<OUTPUT_ROOT>/q3_v2_local.yaml' --variant A1 --output-dir '<OUTPUT_ROOT>/q3_train'
python evaluate.py --config '<OUTPUT_ROOT>/q3_v2_local.yaml' --checkpoint outputs/q3_v2/final_model.pt --split valid
python faithfulness_eval_v2.py --config '<OUTPUT_ROOT>/q3_v2_local.yaml' \
  --checkpoint outputs/q3_v2/final_model.pt --output-dir '<OUTPUT_ROOT>/q3_faithfulness' --budget 0.25
python infer_attachment4.py --config '<OUTPUT_ROOT>/q3_v2_local.yaml' \
  --checkpoint outputs/q3_v2/final_model.pt --output-dir '<OUTPUT_ROOT>/q3_attachment4'
```

`train.py` 的 `A1` 是旧训练 CLI 变体名，最终结构需同时满足 v2 final 配置；若重训，应重新选择 validation checkpoint，不覆盖 `final_model.pt`。复核解释前，train 基线需要存在；仓库已有 `outputs/q3_v2/train_feature_baselines.npz`。完整命令与附件 4 清单检查见 `E题/Q3/Q3_V2_REPRODUCE.md`。本任务不重跑 test（测试集）或附件 3/4。

## 只生成论文材料

```bash
python paper/scripts/build_all.py
python paper/scripts/generate_all_figures.py
```
