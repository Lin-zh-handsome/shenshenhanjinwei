# LNLN 独立复现基线

官方来源：[Haoyu-ha/LNLN](https://github.com/Haoyu-ha/LNLN)，固定 Git 提交 `80a052c62b7b3b7a6cb6956846e9d9c121f56eda`。官方论文：Zhang、Wang、Yu，*Towards Robust Multimodal Sentiment Analysis with Incomplete Data*，NeurIPS 2024。此目录只包含本赛题的数据和评估适配脚本；LNLN 的 DMC（主导模态校正）与 DMML（基于主导模态的多模态学习）仍从官方仓库导入，不进入 SRF-MSA 主模型。

在服务器上准备官方代码：

```bash
git clone https://github.com/Haoyu-ha/LNLN.git /home/hanjinwei/math/baselines/LNLN
git -C /home/hanjinwei/math/baselines/LNLN checkout 80a052c62b7b3b7a6cb6956846e9d9c121f56eda
```

`config.yaml` 设置赛题附件 2 的 `aligned_50.pkl` 和现有 `bert-base-uncased` 权重。BERT 是通用预训练语言模型，不能用其他情感数据集训练。若缺依赖，使用国内 PyPI 源：

```bash
/home/hanjinwei/miniconda3/envs/math/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r baseline_lnln/requirements.txt
```

在 Q2 根目录运行：

```bash
python baseline_lnln/train_baseline.py --config baseline_lnln/config.yaml
python baseline_lnln/evaluate_baseline.py --config baseline_lnln/config.yaml --split valid
python baseline_lnln/evaluate_baseline.py --config baseline_lnln/config.yaml --split test
```

以上命令也封装在 `baseline_lnln/run_baseline.sh`。该脚本遇到现存 checkpoint 会停止，避免无意复用或覆盖旧实验。

适配说明：官方 MOSEI 配置使用 `unaligned_50.pkl` 的音视频 500 步；本赛题要求统一的 `aligned_50.pkl` 50 步，所以仅把输入长度设为 `[50,50,50]`。官方模型只有情感强度回归头；本基线保持这一结构，在验证集上选择中性区间阈值后才派生三分类，报告时标明这种类别输出与 SRF-MSA 的直接分类头不同。训练掩码默认 `official_random`，按每批半数样本不缺失、其余样本随机选择独立缺失率来模拟官方随机点缺失协议；这不是逐行复刻作者数据读取器。`q2_span` 是额外的连续区间训练条件。两者都在同一套固定 30% 连续缺失 valid 上评价，不混淆训练协议。

官方脚本每轮评估 test 并按 test 保存模型；本适配脚本只用 train 更新参数，只用 valid 选择 checkpoint（模型权重）和中性阈值，最终模型冻结后才运行一次 test。原始官方报告指标和本赛题 3 分类指标定义不同，不能直接把论文数字与本项目表格相减。所有改动均在适配脚本中，官方代码工作区保持原样。
