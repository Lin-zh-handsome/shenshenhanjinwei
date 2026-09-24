# Q1 方法

对样本 i 的实际解码有效时长 `T_i`，设 `I_{i,k}=[kT_i/50,(k+1)T_i/50)`，`k=0,…,49`。词、音频帧、视频帧都用同一组边界分箱。具体时长回退、质量阈值和预处理参数以 `E题/Q1/feature_config.yaml` 为准。

文本：官方英文转写 → NFKC 规范化 → MFA（强制对齐）词边界（无锚点保留降级标记） → BERT 子词均值 → overlap（时间重叠）加权窗聚合 → 有效窗上无监督 PCA 到 128 维。Source: `E题/Q1/q1_pipeline.py`、`q1_alignment.jsonl`。

音频：16 kHz 单声道、25 ms 帧、10 ms 帧移；37 项描述子的窗内均值与标准差为 74 维。视觉：逐帧真实 PTS（显示时间戳）、52 项面部 blendshape（面部形变系数）、人脸置信度和覆盖率门槛，窗内多帧中位数聚合。Source: `E题/Q1/q1_pipeline.py`、`feature_config.yaml`。
