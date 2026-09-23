# Q2 根因诊断：Run A Oracle Text

本阶段从提交 `8c2ecf6` 建立 `codex/q2-oracle-diagnostic` 分支。所有训练和验证均在服务器进行。本阶段没有重新评估 test，也没有运行 BERT 重构或 A1–A5。

## 0. 旧版复现

将旧版 A5 从头复训，使用原配置、固定划分与 seed 42，输出到 `outputs/q2_v2/00_baseline_reproduction/`。最佳联合检查点仍在第 12 轮，clean valid 的 Accuracy、Macro F1、Weighted F1、MAE、Pearson 与提交 `8c2ecf6` 的记录逐项一致。详见 `baseline_reproduction.json`。本步只比较 valid，没有重新计算 test。

## 1. Run_A_OracleText

直接读取附件 2 的 50×768 `text` 特征，以 `LayerNorm + Linear` 投影；audio、vision 各自投影后经单层模态时间编码器，拼接并投影到单层融合编码器，最后做有效位置注意力池化和分类、回归。有效位置由 `text_bert` 的 attention mask（注意力掩码）确定，因为 `text` 在尾部 padding（填充）处也可能非零。此阶段**未加入显式时间位置编码**，为下一阶段单独测量其增益留下对照。

训练只使用加权 CE（交叉熵）与 Smooth L1（平滑绝对误差），保持旧版的类别权重计算方式；关闭人工缺失、几何编码、重建、可靠性融合和一致性项。模型由 valid Macro F1（宏平均 F1）选择，第 6 轮最佳。没有使用 test 或附件 3 选模。

| valid clean 指标 | 旧版 `8c2ecf6` | Run A | 变化 |
|---|---:|---:|---:|
| Accuracy | 0.5055 | **0.6387** | +0.1332 |
| Macro F1 | 0.4719 | **0.6126** | +0.1407 |
| Weighted F1 | 0.4991 | **0.6338** | +0.1347 |
| MAE | 0.7510 | **0.6066** | -0.1444 |
| Pearson | 0.3752 | **0.6421** | +0.2669 |

| 类别 | Precision（精确率） | Recall（召回率） | F1 |
|---|---:|---:|---:|
| Negative | 0.6250 | 0.7524 | 0.6828 |
| Neutral | **0.5000** | **0.4130** | **0.4524** |
| Positive | 0.7134 | 0.6923 | 0.7027 |

旧版 Neutral 的精确率、召回率、F1 分别为 0.3631、0.3098、0.3343。Run A 的 Neutral F1 提升到 0.4524，但仍明显弱于另外两类。Run A 的完整混淆矩阵、逐样本 valid 预测和训练曲线保存在 `outputs/q2_v2/01_run_a_oracle_text/`。

## 判断与边界

直接使用现成文本特征后，clean valid 的分类和回归指标都明显改善；这强烈支持现有随机 token TextFeatureBridge（文本特征桥接器）是主要瓶颈之一。**Run A 同时简化了下游架构和损失，因此这一对比不能单独证明全部增益都来自文本桥。** 下一阶段应在相同 Oracle 模型中只加入 Temporal Position Embedding（时间位置编码）得到 `02_run_a_oracle_text_pos`，之后再用冻结预训练 BERT 建立可部署的文本输入链路，并继续比较。

根据用户设置的阶段门槛，本阶段在 Run A 结果处停止，待确认后再执行下一阶段；A1–A5 未运行，test 未重新访问。
