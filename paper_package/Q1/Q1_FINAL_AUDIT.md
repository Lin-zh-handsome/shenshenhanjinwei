# Q1_FINAL_AUDIT：最终结果自动审计

生成来源：`q1_manifest.csv`、`q1_features.npz`、`q1_alignment.jsonl`、`q1_run_info.json`、`validation_report.json`。本报告由 `make_q1_report.py` 从实际运行数据生成，不手工填入实验结果。

## 覆盖、接口与存储

- Excel / manifest / NPZ 样本 ID 集合完全一致：100 条；重复 ID：0。
- 特征形状：text `(100, 50, 128)`，audio `(100, 50, 74)`，vision `(100, 50, 52)`；真实时间边界 `(100, 51)`。
- 统一步数 K=50；未使用时间维 padding；所有首边界为 0、每条严格递增、末边界与 `duration_used_sec` 一致：PASS。
- 实际时长范围：2.236–29.267 s (median 6.700 s)；NPZ 压缩后大小：1.97 MiB。
- PTS 解码得到的 FPS 中位数 `29.846`，范围 `23.691–30.000`；实际帧逐一按时间戳分箱。
- 旧实现记录为 1,084/1,627 个中心帧视觉有效窗、25/100 条视频完全无有效视觉窗。新版统计使用 K=50 的 5,000 窗和多帧覆盖率/置信度规则，分母不同；本报告同时提供新版数值，不能把差值单独归因于某一个因素。
- float32 转 float16 误差：text 最大/平均 `0.003880501` / `8.946e-05`；audio `2` / `0.01069208`；vision `0.0002441406` / `1.154521e-05`。
- 有效文本窗 4038/5000；有效音频窗 4698/5000；存在可解码音频帧窗 4966/5000；有效视觉窗 3894/5000。
- 特征 NaN/Inf 检查：{"text": 0, "audio": 0, "vision": 0}。

## 视频、音频解码及时长口径

- 完整解码到结束：视频 100/100；音频 100/100（有音频流样本）。未完整样本在本报告末列出，保留在结果中。
- 视频流时长减实际解码 PTS 时长：中位数 `0.0193` s，最大绝对差 `0.1000` s。
- 音频解码时长减视频解码时长：中位数 `-0.0903` s，最大绝对差 `0.1744` s。
- 每条 `duration_used_sec` 定义为最后一个有效解码视频帧 PTS 减第一个有效解码视频帧 PTS，再加末帧实际 duration（缺失时用解码 PTS 间隔中位数）；窗口边界为该时长的 50 等分。PTS 来自 PyAV 帧的 `pts * time_base`，不由帧序号/FPS 推算。只有无有效视频 PTS 时才降级到 ffprobe 视频流/容器时长，并在 warning 中注明。
- 负 PTS 样本：0；逆序 PTS 样本：0；缺失 PTS 帧总数：0。

## 文本对齐与模态质量

- 有词级时间边界的词：1934/1934；其中 MFA 强制对齐 1923，内部插值 0，词序比例降级 11。边界覆盖率不等于准确率。
- MFA 运行状态：`ok`；G2P（字素到音素）返回码 `0`；TextGrid（词级时间标注文件）数量 `99`。
- 10 条随机样本、每条最多 6 个词的人工核查 CSV：`q1_manual_alignment_review.csv`（实际 60 行）。人工真值及抽查结论未自动推断；无独立人工时间戳时不报告准确率。
- 音频可靠窗要求存在分析帧且窗口平均 RMS 不低于配置阈值 `0.001`；F0 只在满足有声判断的帧统计。其 pYIN 估计支撑为 403 个采样点（25.19 ms），输出插值到 400 点（25 ms）的共同帧中心；这样保留 25 ms 主分析网格，同时为 80 Hz 下限提供足够的周期支撑。零可靠音频窗样本：-mJ2ud6oKI8/1, -mJ2ud6oKI8/2。
- 视觉 mask 采用 face coverage ≥0.20 且 YuNet 平均检测分数 ≥0.50；blendshape（面部形变系数）不是人工 AU 真值，也不是情感标签。低覆盖/无有效视觉样本：-HwX2H8Z4hY/9, -NFrJFQijFE/1, -NFrJFQijFE/2, -UuX1xuaiiE/1, -iRBcNs9oI8/3, -iRBcNs9oI8/7, -iRBcNs9oI8/6, -iRBcNs9oI8/9, -iRBcNs9oI8/8, -mJ2ud6oKI8/1, -mJ2ud6oKI8/2, -wny0OAz3g8/3, -wny0OAz3g8/5, -wny0OAz3g8/7, -hnBHBN8p5A/7, -hnBHBN8p5A/6, -ri04Z7vwnc/0。

## 方案比较

### 时间粒度

| K | 每窗平均音频帧 | 每窗平均视频帧 | 空文本窗比例 | 有效视觉窗比例 | 压缩变体大小（PCA前，字节） |
|---:|---:|---:|---:|---:|---:|
| 25 | 31.00 | 9.30 | 0.1528 | 0.7800 | 3,000,815 |
| 50 | 15.50 | 4.65 | 0.1924 | 0.7788 | 4,623,774 |
| 100 | 7.75 | 2.32 | 0.2195 | 0.7607 | 6,488,168 |

### 视觉聚合

| 方法 | 有效视觉窗比例 | 完全无有效视觉样本数 |
|---|---:|---:|
| center_frame | 0.7682 | 13 |
| multi_frame_mean | 0.7788 | 12 |
| multi_frame_median | 0.7788 | 12 |

中位数与多帧均值的特征平均绝对差：`0.003696026`；中位数与中心帧的特征平均绝对差：`0.007541077`；检测到的人脸窗口 blendshape 越界比例：`0`。所有视觉策略共用同一覆盖率与置信度阈值。

### 文本时间映射

| 方法 | 词级时间边界覆盖率 | 时间逆序数 | 人工真值 |
|---|---:|---:|---|
| uniform_word_order | 1.0000 | 0 | 未提供时间戳真值 |
| mfa_only | 0.9943 | 0 | 未提供时间戳真值 |
| mfa_plus_fallback | 1.0000 | 0 | 未提供时间戳真值 |

没有提供人工标注时间戳，所以本比较不把边界覆盖率称为定位准确率。

## 全部 warning 汇总

| warning code | 样本数 |
|---|---:|
| `low_global_face_coverage` | 17 |
| `mfa_word_alignment_unavailable` | 1 |
| `no_reliable_audio_windows` | 2 |
| `no_reliable_vision_windows` | 12 |
| `text_words_proportional_fallback` | 1 |

视频未完整解码样本：无。音频未完整解码样本：无。

## 设计与复核材料

- 短/中/长样本图分别为 `figures/q1_typical_short.png`、`figures/q1_typical_medium.png`、`figures/q1_typical_long.png`。
- `figures/q1_design_comparison.png` 对照时间粒度、视觉聚合有效率与文本时间边界覆盖率。
- 图中关键帧、音频波形、词起止时间、三模态特征摘要和 mask 使用同一秒级时间轴；每个 k 可由 `q1_quality_report.csv` 的 `[start_sec,end_sec)` 回查。
