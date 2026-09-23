# 问题二：SRF-MSA

SRF-MSA（Span-aware Reconstruction and Reliability Fusion for Multimodal Sentiment Analysis，连续缺失区间感知的重建与可靠性融合情感模型）使用赛题附件 2 的 `aligned_50.pkl` 训练和验证，并对附件 3 的 30 个对齐样本推理。训练参数只来自 `train`，结构与训练轮次只根据 `valid` 选择；`test` 在最终模型冻结后评估一次。附件 3 无标签，只输出预测。

## 数据路径与环境

编辑 `configs/q2_aligned.yaml` 中的 `paths.aligned_pkl`、`paths.attachment3_dir` 和 `paths.output_dir`。服务器默认路径已填入。运行环境使用服务器现有的 `/home/hanjinwei/miniconda3/envs/math/bin/python`，依赖版本范围记录于 Git 中的 `requirements.txt`。如确需补装 Python 包，使用国内源：

```bash
/home/hanjinwei/miniconda3/envs/math/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 运行

在本目录执行：

```bash
python train.py --config configs/q2_aligned.yaml
python evaluate.py --config configs/q2_aligned.yaml --checkpoint outputs/q2/best_joint.pt --split valid
python evaluate.py --config configs/q2_aligned.yaml --checkpoint outputs/q2/best_joint.pt --split test
python robustness_sweep.py --config configs/q2_aligned.yaml --checkpoint outputs/q2/best_joint.pt
python ablation.py --config configs/q2_aligned.yaml
python infer_attachment3.py --config configs/q2_aligned.yaml --checkpoint outputs/q2/best_joint.pt
python export_q2_report_tables.py --config configs/q2_aligned.yaml
```

`run_q2.sh` 按上述次序运行，可在服务器 `screen` 会话中启动。`train.py` 先用训练集预热文本桥接器 3 轮，再训练完整模型；消融组 A0 至 A4 分别独立训练并共用同一份预热权重。模型选择使用固定种子、30% 连续缺失的 `valid` 指标，综合分数权重见配置；附件 2 的 `test` 不用于模型选择。

## 模型与输出

```text
text_bert -> Text Feature Bridge -> Span Geometry -> Temporal Reconstruction --+
audio     -> Audio Adapter       -> Span Geometry -> Temporal Reconstruction --+-> Reliability Fusion -> Temporal Encoder -> Attention Pooling -> 分类/回归
vision    -> Vision Adapter      -> Span Geometry -> Temporal Reconstruction --+
```

`outputs/q2/` 含 `best_joint.pt`、`best_cls.pt`、`best_reg.pt`、训练曲线 CSV、验证及测试指标 JSON、四张缺失规律表、`ablation_table.csv`、附件 3 全量预测 CSV、论文图和结果摘要。`attachment3_predictions.csv` 包含三分类概率与 `[-3,3]` 情感强度，`sample_id` 使用原文件名 stem。

附件 3 的模态缺失掩码由有效区间内全零候选推断，属于 **zero-derived missing candidates（零值推断缺失候选）**，不是赛题提供的真实缺失掩码。

## 相关工作

- LNLN: *Towards Robust Multimodal Sentiment Analysis with Incomplete Data*, NeurIPS 2024。
- P-RMF: *Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data*, ACL 2025。
- CMAD: *Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities*, ICCV 2025。

本实现独立构造连续区间增强、区间几何编码、时间重建和逐时刻可靠性融合；引用上述文献时不将通用重建或门控思想称为首创。
