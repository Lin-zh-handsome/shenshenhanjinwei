# Q2 最终 R3 方法

1. **输入**：官方 `aligned_50.pkl` 的 `text_bert`、音频 74 维、视觉 35 维、有效位置；训练缺失由确定性连续 span（区间）mask（掩码）产生。Source: `E题/Q2/data/aligned_dataset.py`、`missing_curriculum.py`。
2. **文本编码**：`bert-base-uncased` 的后四层微调基础模型，输出每位置 768 维，线性投影到 128 维。Source: `models/bert_text_encoder.py`、`clean_backbone.py`、最终配置。
3. **音频/视觉编码**：各自 LayerNorm + 线性投影到 128 维，再经一层模态专属 Transformer（变换器）时序编码。Source: `models/clean_backbone.py`。
4. **Temporal encoding（时序编码）**：基础模型 `use_position=false`；重建分支有内部可学习时间向量。不要写成全网络绝对位置编码。Source: `configs/best_R3_reconstruction.yaml`、`models/temporal_reconstruction.py`。
5. **Multimodal fusion（多模态融合）**：三模态按通道拼接、投影、融合时序 Transformer、注意力池化。最终 `fusion_mode=concat`；text-centered（文本中心）实现仅为历史可选路径。Source: `models/clean_backbone.py`。
6. **输出头**：三类分类头与线性回归头，分别预测极性和强度。Source: `models/clean_backbone.py`。
7. **Continuous span missing（连续区间缺失）**：训练按课程式概率生成真值 mask；验证固定 T/A/V/TA/TV/AV 轮换、中段遮挡 30%。附件 3 无官方逐格 mask，仅产生零值推断候选。Source: `missing_train.py`、`data/mask_utils.py`。
8. **Span Geometry（区间几何）**：利用缺失位置、左右边界距离和区间长度等 5 维几何输入，编码到 128 维。Source: `models/span_geometry.py`。
9. **Temporal Reconstruction（局部时序重建）**：以缺失 token（缺失标记）和几何表示估计缺失位置投影；仅人工遮挡位置计 L1 重建损失，目标为冻结基础网络的 clean 投影。Source: `models/srf_msa.py`、`models/temporal_reconstruction.py`、`missing_train.py`。
10. **Loss（损失）**：加权交叉熵 + `0.5 × SmoothL1` 回归 + `0.1 × L_rec` 重建。Source: `E题/Q2/configs/best_R3_reconstruction.yaml`、`missing_train.py`。

最终保留缺失训练、几何与重建。R4 reliability（可靠性残差）、R5 clean-masked consistency（干净与遮挡一致性）、text-centered fusion（文本中心融合）及 oracle（教师特征）诊断不在 R3 最终路径。完整代码与结果见 [Q2 README](../../E题/Q2/README.md)。
