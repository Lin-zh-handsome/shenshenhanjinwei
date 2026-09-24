# Q3 三层解释写作结构

## Level 1: Prediction（预测）

FER-MSA v2 B1 在附件 2 validation（验证集）输出三类情感极性与连续情感强度。Source: `E题/Q3/models/fer_msa.py`、`outputs/q3_v2/01_a1_reproduction/valid_metrics.json`。

## Level 2: Modality Explanation（模态解释）

LOMO（逐模态删除）计算 Text/Audio/Vision 对预测的反事实作用，得到模态贡献和 Main Modality（主模态）。Router（路由器）仅为辅助诊断，不能替代 LOMO。Source: `E题/Q3/explain/modality_ablation.py`、`outputs/q3_v2/faithfulness/final_b1_router/budget_025/valid_explanations.csv`。

## Level 3: Local Evidence（局部证据）

Evidence Gate（证据门）给 Candidate（候选片段），Local Occlusion（局部遮挡）逐个验证，最终输出 Verified Evidence Segment（已核验证据段）及支持/反向/混合方向。模型内部权重不等于最终解释。Source: `E题/Q3/explain/{evidence_segments,local_occlusion}.py`、`outputs/q3_v2/attachment4/attachment4_explanations_long.csv`。

**论文贡献措辞**：上下文条件证据门、LOMO 模态贡献、候选与遮挡双验证。位置编码、门控感知池化、正则化和预测主路 Router 均未进入最终 B1。Top vs Random（关键证据对比随机片段）的定量支持见 [faithfulness](Q3_FAITHFULNESS.md)。
