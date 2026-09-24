# Q1 论文写作笔记

一句话：把附件 1 原始视频的文本、音频、视觉信号投射到每样本 50 个真实时间窗，并显式保留质量和时间来源。

方法三点：实际解码 PTS（显示时间戳）决定有效时长；MFA（强制对齐）与 BERT/PCA（主成分分析）形成 128 维文本；音频和视觉按同一窗聚合并分别生成质量 mask（掩码）。Source: `E题/Q1/q1_pipeline.py`、`feature_config.yaml`。

写作边界：词级边界覆盖率不是人工时间定位准确率；不把 Q1 输出直接称作 Q2 模型输入。结果和图表入口见 [Q1_RESULTS](Q1_RESULTS.md)、[Q1_TABLES](Q1_TABLES.md)、[Q1_FIGURES](Q1_FIGURES.md)。
