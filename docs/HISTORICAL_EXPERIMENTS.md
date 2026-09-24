# 历史实验与诊断

保留这些文件是为追溯方法选择，不用于替换 [单一来源](../paper/SOURCE_OF_TRUTH.md) 中的最终版本。

| 项目 | 为什么保留 | 论文角色 | 最终采用？ |
|---|---|---|---|
| Q1 旧 0.5 秒/59 窗偏差记录 | 说明冻结规格修正背景 | 历史审计 | 否 |
| Q2 Oracle teacher（教师特征） | clean backbone（基础网络）诊断/阶段训练 | 诊断 | 否 |
| Q2 score_push clean BERT test | 选定基础模型历史评估 | 历史基线 | R3 不采用其 test 数值 |
| Q2 R0/R1/R2/R4/R5 | 证明最终 R3 选择依据与边界 | 基线/消融 | 模块按 R3 配置决定 |
| Q2 TextCenteredResidualFusion（文本中心残差融合） | 可选历史结构 | 历史代码 | 否 |
| Q3 `outputs/q3/` A0–A5 和旧 checkpoint | v1 结果溯源 | 历史 | 否 |
| Q3 B2/B3/B4/B5 | v2 位置、池化、正则、诊断 Router 消融 | 消融 | B1 预测主干最终采用 |

具体文件逐项见 [仓库盘点](../REPO_INVENTORY.md) 和 [实验注册表](../paper/EXPERIMENT_REGISTRY.csv)。
