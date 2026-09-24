# Q2 论文创新点与证据

### 创新点 1：连续区间缺失训练

动机：模型需处理局部连续遮挡，而非整模态消失。定义：在有效长度内采样连续 span（区间）`M_m[k]∈{0,1}`，缺失投影由可学习 token（缺失标记）替换。代码：`E题/Q2/data/missing_curriculum.py`、`data/mask_utils.py`、`models/srf_msa.py`。消融：R0→R1；mixed missing 的 Accuracy/Macro-F1 提高，clean 保持。Source: `E题/Q2/outputs/q2_best_ablation/ablation_table.csv`。**支持：YES（固定人工缺失验证场景）**。

### 创新点 2：Span Geometry（缺失区间几何编码）

动机：让重建分支知道遮挡边界和长度。定义：缺失处的五维输入依次为样本缺失比例、到区间左边界的归一化距离、到右边界的归一化距离、区间长度占比和绝对位置占比；非缺失处的编码输出乘 mask（掩码）归零。代码：`E题/Q2/models/span_geometry.py`。消融：R1→R2，缺失分类未提升，MAE/Pearson 略好；与 R3 重建共同使用。Source: `E题/Q2/outputs/q2_best_ablation/ablation_table.csv`。**支持：PARTIAL；不能单独声称分类增益**。

### 创新点 3：Temporal Reconstruction（局部时序重建）

动机：用同模态周围有效位置恢复被遮挡的隐藏表示。定义：`L_rec = mean_{m,k:M_m[k]=1} |\hat h_m[k]-h_m^{clean}[k]|`，clean 目标停止梯度；正式实现见代码。代码：`E题/Q2/models/temporal_reconstruction.py`、`models/srf_msa.py`。消融：R2→R3，混合缺失分类改善；回归并非全面最优。Source: `E题/Q2/outputs/q2_best_ablation/ablation_table.csv`。**支持：YES（分类），回归需分别报告**。

## 未采用方案

Reliability（可靠性残差）R4、consistency（一致性）R5、text-centered fusion（文本中心融合）和 Oracle teacher（教师特征诊断）均不作为最终创新点。R5 的部分缺失单项指标高于 R3，但预先定义的 clean/missing 综合分类排序低于 R3；见 `E题/Q2/Q2_BEST_MODEL_ABLATION.md`。
