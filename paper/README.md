# 论文材料入口

先读 [写作摘要](PAPER_WRITING_BRIEF.md)，按 [论文索引](PAPER_INDEX.md)找代码与证据，再用 [结论证据矩阵](CLAIM_EVIDENCE_MATRIX.md)判断措辞。原始实验仍在 `E题/Q*/outputs/`；此目录只放索引、摘要、轻量表格与精选图。

| 需求 | 入口 |
|---|---|
| 最终结果 | [结果总表](FINAL_RESULTS.csv)、[单一来源](SOURCE_OF_TRUTH.md) |
| 所有实验 | [实验注册表](EXPERIMENT_REGISTRY.csv)、[仓库盘点](../REPO_INVENTORY.md) |
| 论文表图 | [tables](tables/)、[figures](figures/)、[图数据](generated/) |
| 方法/符号 | [模型卡](MODEL_SUMMARY.md)、[符号表](SYMBOLS.md) |
| 复现/提交 | [复现](REPRODUCTION.md)、[提交清单](SUBMISSION_MANIFEST.md) |
| 避免误写 | [注意事项](PAPER_CAUTION.md)、[数值冲突报告](RESULT_INCONSISTENCIES.md) |

重新生成轻量结果：在仓库根目录执行 `python paper/scripts/build_all.py`。该命令只读取已存在的 CSV/JSON 和文件元数据，不运行模型。
