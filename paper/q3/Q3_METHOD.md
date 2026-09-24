# Q3 最终 FER-MSA v2 B1 方法

1. **三模态 encoder（编码器）**：文本 768、音频 74、视觉 35 维分别投影到 128 维，经过模态专属 Transformer（变换器）时序编码；`valid_mask` 去除填充。Source: `E题/Q3/models/modality_encoder.py`。
2. **Temporal Position（时间位置）**：代码支持位置嵌入，但最终 `configs/q3_v2_final.yaml` 为 `use_position=false`。B2 位置版本属于消融，不能写进最终结构。
3. **Context-conditioned Evidence Gate（上下文条件证据门）**：三模态编码拼接形成上下文，针对每个模态、每个有效位置预测门值。Source: `models/fer_msa.py`、`evidence_gate.py`。
4. **Pooling（池化）**：最终 `pooling_mode=legacy`，使用 `(1-ε)g_m+ε` 的软残差加权池化，`ε=0.1`；不是 B3 的门控感知池化。Source: `models/fer_msa.py`、最终配置。
5. **Fusion 与 heads（融合与输出头）**：三个模态池化表示拼接，经简单融合后输出三类分类概率与连续强度。Source: `models/evidence_router.py` 中 `SimpleMultimodalFusion`、`prediction_heads.py`。
6. **LOMO（逐模态删除）**：用 train（训练集）均值作遮挡基线，度量删去各模态后的分类和回归变化并归一化，最大者为 main modality（主模态）。Source: `explain/modality_ablation.py`、`data/feature_baseline.py`。
7. **Local occlusion（局部遮挡）**：证据门先提供局部候选，再在同一模态遮挡窗口检查预测变化；最终分数综合 gate 与遮挡幅度，并保留 signed effect（带符号效应）和 supportive/contradictory/mixed（支持/反向/混合）方向。Source: `explain/local_occlusion.py`、`evidence_segments.py`。
8. **Router（路由器）**：最终权重含 train 上拟合的诊断 Router，但 `FERMSA._predict` 使用普通拼接融合，Router 不缩放预测表示。正式解释不采用 Router 权重。Source: `models/fer_msa.py`。
9. **Faithfulness（忠实性）**：validation 对 Full、Keep-only、Remove 与匹配 Random Remove 做同条件比较，预算 0.25。Source: `faithfulness_eval_v2.py`、最终 JSON。

最终 B1 与旧 A5 结构不同。旧 `outputs/q3/` 的 test（测试集）和解释结果不得代替 v2 validation。
