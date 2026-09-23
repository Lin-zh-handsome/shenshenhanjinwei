# Q1 特征数据交付说明

本目录包含 E 题第一问的 100 条附件1视频处理代码、实验结果与审阅材料。

## 结果

- q1_features.npz：压缩多模态特征、逐窗时间边界、有效长度和模态掩码。
- q1_manifest.csv：100 条样本的特征与质量摘要。
- q1_alignment.jsonl：词级起止时间及对齐状态。
- q1_quality_issues.csv：未对齐、无有效模态或词典外词的样本记录。
- q1_alignment_example.csv 与两张 PNG 图：代表样本对齐、特征和视频帧展示。
- Q1_论文正文.md：方法、结果、质量状态与限制。
- q1_run_info.json 和 MFA 运行日志：环境、参数与执行记录。
- run_q1.py、make_q1_report.py：特征生成与报告脚本。

## 张量

文本、语音、视觉张量分别为 100×59×128、100×59×40、100×59×52。每条样本实际窗数由 lengths 给出；缺失模态由对应掩码表示，不以数值零代替观测。

## 复现说明

代码需要题面附件1原始数据及 Montreal Forced Aligner（蒙特利尔强制对齐器）、通用 BERT（双向语言表示模型）、MediaPipe（多媒体感知工具）模型文件。本目录不包含原始视频、赛题数据或任何预训练权重。

运行前准备数据目录与模型目录；默认路径分别为仓库根目录下的 data/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条 和 models。也可设置 Q1_DATA_DIR、Q1_MODEL_DIR、Q1_BASE_DIR 环境变量或使用命令行路径参数。环境与版本记录见 q1_run_info.json。

在本目录执行：
    python run_q1.py --stage all --num-jobs 8 --window-sec 0.5 --beam 100 --retry-beam 400
    python make_q1_report.py

运行仅读取样本键与英文转写，不读取答案标签列。公开仓库中的结果可用于审阅；复现时请遵守赛题数据使用规定。