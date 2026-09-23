# Q1：特征提取与时序对齐

本目录实现 `03_Q1_特征提取与时序对齐算法.md` 的冻结方案。主接口固定为 50 个真实时间窗，三模态共享每条视频的边界 `time_edges[i,k] = k * duration_used_sec[i] / 50`。特征从无标签的转写、音视频信号和预训练模型生成；情感标签不参与特征选择、阈值、PCA 或时间窗设置。

## 在服务器运行

先进入仓库中的 Q1 目录并激活运行环境：

```bash
cd "/path/to/repository/E题/Q1"
conda activate math
export Q1_DATA_DIR="/path/to/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"
export Q1_MODEL_DIR="/path/to/models"
export Q1_BASE_DIR="/path/to/q1_run"
python run_q1.py --stage all
python validate_q1.py --data-dir "$Q1_DATA_DIR" --result-dir "$Q1_BASE_DIR/results"
python make_q1_report.py --data-dir "$Q1_DATA_DIR" --result-dir "$Q1_BASE_DIR/results"
```

预训练权重保存在服务器的模型目录，不纳入代码仓库。`Q1_MODEL_DIR` 下需有 `face_landmarker.task`、`english_us_arpa_acoustic.zip`、`english_us_arpa_dictionary.dict`，以及 `q1_doc03/bert-base-uncased/` 和 `q1_doc03/face_detection_yunet_2023mar.onnx`。文本模型来自固定 revision 的通用 BERT，未使用情感数据微调。

单样本端到端检查示例：

```bash
python run_q1.py --stage all --sample-id=-mJ2ud6oKI8/2 --work-dir /tmp/q1-smoke-work --output-dir /tmp/q1-smoke-results
```

该路径只生成单样本诊断结果，不拟合最终 PCA，也不覆盖正式 100 样本结果。

## 算法配置摘要

- **文本**：官方 `label-100.xlsx` 英文转写；NFKC Unicode 规范化并保留原文；MFA 词级强制对齐，局部缺失插值、无锚点时按词序比例降级；BERT-base-uncased 的子词均值形成词向量；按 `overlap / word_duration` 聚合到 50 窗；只在有效文本窗口上无监督 PCA 到 128 维。
- **音频**：FFmpeg 转换为单声道 16 kHz PCM WAV；37 项帧描述子采用 25 ms 帧长、10 ms 帧移，按窗统计均值和标准差，共 74 维。pYIN 基频估计使用 403 个采样点（25.19 ms）支撑，并插值到同一 25 ms 帧中心；此微小支撑差异写入完整配置。F0 仅在有声帧统计；`audio_present` 与可靠音频 `audio_mask` 分开。
- **视觉**：完整视频逐帧解码并读取真实 PTS；MediaPipe Face Landmarker 的 52 个命名 blendshape（面部形变系数，模型输出，不是人工 AU 真值）逐帧提取；YuNet 检测分数提供独立置信度；每窗用多帧中位数聚合，并按覆盖率和置信度生成掩码。
- **时间**：有效时长按首末已解码视频帧真实 PTS 及末帧时长确定；若无可用 PTS 才按配置降级到流/容器时长并记录 warning。所有边界和实际口径见 `feature_config.yaml`、`q1_manifest.csv`。
- **归一化**：`q1_features.npz` 保留原始聚合特征；`normalization_stats.npz` 单独记录仅基于有效位置拟合的模态均值、标准差和计数。

## 输出

- `q1_features.npz`：`ids`、`text (N,50,128)`、`audio (N,50,74)`、`vision (N,50,52)`、模态掩码、真实秒级边界、质量数组、特征名和 PCA 参数。
- `q1_manifest.csv`：逐样本解码、时间、对齐、可靠窗、形状和 warning。
- `q1_alignment.jsonl`：原始/规范化文本、每词字符范围、时间、来源、重叠窗口和降级原因。
- `q1_quality_report.csv`：逐窗口 T/A/V 覆盖与质量指标。
- `normalization_stats.npz`、`feature_config.yaml`、`software_versions.json`、`q1_run_info.json`、`processing_log.jsonl`：复现配置、统计参数和运行记录。
- `Q1_FINAL_AUDIT.md`、`Q1_论文正文.md`、三张典型样本图、设计比较图、人工抽查 CSV：由 `make_q1_report.py` 从运行结果生成。

`word_boundary_coverage` 表示获得词级时间边界的比例，不是对齐准确率；无人工标注时间戳时不声称准确率。

## 最终运行记录

- 样本：100；实际解码时长：2.236–29.267 s（中位数 6.700 s）。
- 形状：text `(100, 50, 128)`；audio `(100, 50, 74)`；vision `(100, 50, 52)`。
- K=50 有效窗口：text 4038/5000；audio 4698/5000；vision 3894/5000。
- 解码到结束：视频 100/100；有音频流的样本 100/100。
- 特征文件大小：1.97 MiB。完整逐样本指标见 `Q1_FINAL_AUDIT.md` 和 `q1_manifest.csv`。