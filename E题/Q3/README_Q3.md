# Q3 FER-MSA 工程

> **Q3 v2 最终选择：** 基于 validation 的 B1 复现模型；Router 仅作诊断，证据预算 0.25。新代码、结果和复现边界见 [Q3_FAITHFULNESS_V2_REPORT.md](Q3_FAITHFULNESS_V2_REPORT.md)。下文记录 v1 历史实验，A5 不再默认是最终模型。

FER-MSA（Faithful Evidence Routing for Multimodal Sentiment Analysis，忠实证据路由多模态情感模型）按执行规范完成三分类、`[-3,3]` 情感强度回归、模态作用程度和局部证据输出。实验在服务器 GPU 上完成。

## 数据与环境

先将 `configs/q3_aligned.yaml` 中的路径占位符设置为附件 2 `aligned_50.pkl`、附件 4 对齐版特征和视频在运行环境中的实际路径。训练、验证、测试和附件 4 推理只用 aligned_50（50 格对齐序列），保持三模态共用位置。只在附件 2 train split（训练划分）计算遮挡均值；附件 4 不参与训练或调参。

先准备 Python 环境，并将 `configs/q3_aligned.yaml` 中三个 `SET_PATH_TO_...` 项指向本机附件 2 和附件 4。依赖声明见 `requirements.txt`。如需重新安装非 PyTorch 包，可使用国内镜像；PyTorch（张量计算框架）需安装与服务器 CUDA（显卡计算平台）匹配的版本。

```bash
cd E题/Q3
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

服务器 GPU（图形处理器）内存足够时使用 batch 32；如发生 OOM（显存不足），按执行规范先降到 16/8，再将 temporal layers（时间编码层）降为 1。

首轮 A5 训练发现 gate（证据门）在有效位置趋近 0，未通过文档 Phase 3 的分化标准。按文档允许的坍缩排查路径，将反事实损失前三轮从 0 渐进加权，并把证据预算损失权重从默认 0.05 提高到 1.0；其余模型结构与默认参数不变。最终报告使用修正后重新训练的结果。

## 执行命令

以下命令从克隆仓库根目录执行。`CUDA_VISIBLE_DEVICES=1` 指定第二张显卡，可按服务器占用情况调整。

```bash
cd E题/Q3
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/q3_aligned.yaml
CUDA_VISIBLE_DEVICES=1 python evaluate.py --config configs/q3_aligned.yaml --checkpoint outputs/q3/best_joint.pt --split valid
CUDA_VISIBLE_DEVICES=1 python faithfulness_eval.py --config configs/q3_aligned.yaml --checkpoint outputs/q3/best_joint.pt
CUDA_VISIBLE_DEVICES=1 python ablation.py --config configs/q3_aligned.yaml
CUDA_VISIBLE_DEVICES=1 python evaluate.py --config configs/q3_aligned.yaml --checkpoint outputs/q3/best_joint.pt --split test
CUDA_VISIBLE_DEVICES=1 python infer_attachment4.py --config configs/q3_aligned.yaml --checkpoint outputs/q3/best_joint.pt
CUDA_VISIBLE_DEVICES=1 python export_q3_report_tables.py --config configs/q3_aligned.yaml
```

`ablation.py` 生成 A0–A5 六组消融，已有 checkpoint（模型权重保存点）会复用。test split（测试划分）在主模型冻结后评估一次。

## 主要输出

- `outputs/q3/best_joint.pt`、`best_cls.pt`、`best_reg.pt`：按验证集综合分数、分类、回归选择的模型。
- `train_feature_baselines.npz`：仅由 train 有效位置计算的三模态均值。
- `valid_metrics.json`、`test_metrics.json`、`valid_predictions.csv`、`test_predictions.csv`：标准预测指标和逐样本结果。
- `valid_faithfulness.json`、`valid_faithfulness_random_control.csv`、`valid_deletion_curve.csv`：解释忠实性、随机删除对照和删除曲线。
- `ablation_table.csv`：A0–A5 的性能与解释指标；A0 无 evidence gate（证据门），解释项为 NA（不适用）。
- `attachment4_predictions_explanations.csv`：每个附件 4 样本一行预测、LOMO（Leave-One-Modality-Out，逐模态删除）作用程度和主要证据。
- `attachment4_explanations_long.csv`：每段证据一行，含精确 grid（输入位置）区间、近似秒级区间和遮挡验证分数。
- `attachment4_mapping_notes.csv`：每样本映射方法与视频时长。
- `explanation_cards/` 和 `evidence_frames/`：解释卡和视觉关键帧。
- `fig_*.png`、`valid_error_attribution.csv`、`modality_importance_by_class.csv`、`q3_results_summary.md`：论文图表与结果汇总。

## 解释字段

`router_text/audio/vision` 是模型内部路由权重，只用于诊断。正式 `importance_text/audio/vision` 使用 LOMO（逐模态删除）后分类概率与回归输出变化的加权归一化值；三者和为 1。`main_modality` 为最大 LOMO 作用程度模态。局部候选由 gate（证据门）给出，再用 local occlusion（局部遮挡）验证。`grid_start` / `grid_end_exclusive` 是模型 50 格输入上的精确索引。

附件 4 不提供官方逐词时间戳或 50 格到视频秒数的映射，因此 `start_sec_approx`、`end_sec_approx` 和 `text_fragment_approx` 是按有效长度、视频时长和文本长度做的**近似回投**。它们用于人工回看，不属于真实词级时间标注；附件 4 无标签，也不报告预测正确率或定位真值准确率。

若视频容器报告的帧数与实际可解码帧数不一致，回投使用可解码帧数与 FPS（每秒帧数）计算的时长，并在 `attachment4_mapping_notes.csv` 记录差异。关键帧随机定位失败时，尝试顺序解码到对应帧并记录回退标记。

## 相关工作

论文正文应引用 [EMOE: Modality-Specific Enhanced Dynamic Emotion Experts](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)（CVPR 2025）、[Fast Retrieval and Slow Reasoning for Explainable Multimodal Sentiment Analysis](https://aclanthology.org/2026.findings-acl.1519/)（Findings of ACL 2026）、[Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation](https://aclanthology.org/2026.acl-long.2012/)（ACL 2026）。这些工作是相关研究，FER-MSA 是按本题数据和解释目标独立实现的模型。
