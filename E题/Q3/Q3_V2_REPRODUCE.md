# Q3 FER-MSA v2 运行与数据处理说明

## 已提交的最终材料

- `outputs/q3_v2/final_model.pt`：最终 B1 预测主干与仅供诊断的 Router（路由器）参数；只保存模型权重和必要的轮次、验证指标、推理温度，不包含服务器路径或优化器状态。
- `configs/q3_v2_final.yaml`：与最终权重对应的结构和解释配置。其他 `q3_v2_*.yaml` 对应消融实验。
- `outputs/q3_v2/train_feature_baselines.npz`、`train_feature_position_baselines.npz`：仅从附件 2 的 train（训练集）计算的全局和逐位置基线。
- `outputs/q3_v2/ablation_v2.csv`、`faithfulness/`、`lomo_stability/`：validation（验证集）消融、解释忠实性和模态基线稳定性结果。
- `outputs/q3_v2/attachment4/`：对齐版附件 4 的 20 行预测、120 行局部证据、清单 JSON、20 张解释卡与 40 张视觉关键帧。

## 输入与环境

在 `E题/Q3` 目录运行。先把 `configs/q3_v2_final.yaml` 的三个 `SET_PATH_TO_...` 占位符改为附件 2 `aligned_50.pkl`、附件 4 对齐版特征目录和对应视频目录的实际路径。只使用 `aligned_50` 版本；附件 4 的秒数与文本片段由模型网格近似回投，不是真实时间戳。

服务器完成本轮实验时使用 Python 3.11.16、PyTorch 2.12.1+cu130、NumPy 2.4.6、pandas 3.0.6、SciPy 1.17.1、scikit-learn 1.9.1、PyYAML 6.0.3、Matplotlib 3.11.2、OpenCV 5.0.0。依赖范围见 `requirements.txt`；其他 CUDA（显卡计算平台）环境应安装与自身驱动兼容的 PyTorch。训练使用 seed 42、batch 32、学习率 0.0002、dropout 0.15、最长 60 轮和耐心值 10。

## 从已提交权重运行最终推理

```bash
cd E题/Q3
python -m pip install -r requirements.txt
python -m scripts.check_attachment4_inventory --config configs/q3_v2_final.yaml --output outputs/q3_v2/attachment4/attachment4_inventory.json
python infer_attachment4.py --config configs/q3_v2_final.yaml --checkpoint outputs/q3_v2/final_model.pt --output-dir outputs/q3_v2/attachment4
```

若需重新计算遮挡与 LOMO（逐模态删除）基线，只读取 train：

```bash
python -m scripts.prepare_q3_v2_baselines --config configs/q3_v2_final.yaml
```

若需重新计算 validation 的 LOMO 基线稳定性：

```bash
python -m scripts.lomo_baseline_stability --config configs/q3_v2_final.yaml --checkpoint outputs/q3_v2/final_model.pt --output-dir outputs/q3_v2/lomo_stability
```

## 训练与模型选择

B0 使用 `--variant A0`，B1/B2/B3 使用 `--variant A1`，B4 使用 `--variant A2`；这些参数名沿用旧训练接口，**结果表中的 B0–B5 定义以 `Q3_FAITHFULNESS_V2_REPORT.md` 为准**。B5 在 B4 冻结后仅拟合诊断 Router，预测参数不更新。示例：

```bash
python train.py --config configs/q3_v2_base.yaml --variant A1 --output-dir outputs/q3_v2/01_a1_reproduction
python train.py --config configs/q3_v2_position.yaml --variant A1 --output-dir outputs/q3_v2/02_position
python train.py --config configs/q3_v2_gate_pooling.yaml --variant A1 --output-dir outputs/q3_v2/03_gate_pooling
python train.py --config configs/q3_v2_gate_regularized.yaml --variant A2 --output-dir outputs/q3_v2/04_gate_regularized
```

四个单指标 checkpoint（权重保存点）分别为 `best_accuracy.pt`、`best_macro_f1.pt`、`best_mae.pt` 和 `best_pearson.pt`。本轮各组比较使用相同种子与训练设置，早停保持旧训练器的固定实现；最终选模明确依据 validation 的 Macro-F1 与解释忠实性。最终 B1 第 4 轮同时达到最佳 Macro-F1、Accuracy 与 Pearson；`final_model.pt` 在此基础上只加入 train 拟合的诊断 Router。所有新 `.pt` 中仅最终模型提交，其他 checkpoint 留在服务器。test（测试集）已封存，本轮未重跑、未调参。
