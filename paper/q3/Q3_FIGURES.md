# Q3 论文图

| 用途 | paper 路径 | Source |
|---|---|---|
| 整体模态贡献 | `paper/figures/q3_modality_importance.png` | `paper/generated/q3_modality_importance_plot.csv`，最终 LOMO 逐样本 CSV |
| 典型证据卡 | `paper/figures/q3_evidence_example.png` | `E题/Q3/outputs/q3_v2/attachment4/explanation_cards/01.png` |
| Faithfulness（忠实性） | `paper/figures/q3_faithfulness.png` | `paper/generated/q3_faithfulness_plot.csv`，最终 JSON |
| Temporal self-attention（时序自注意力） | [三模态四头平均图](../figures/q3_attention_07_modalities.png)、[文本四头图](../figures/q3_attention_07_text_heads.png) | 最终 B1 权重、附件四对齐版 07 号样本；[导出脚本](../scripts/export_q3_attention.py)、`paper/generated/q3_attention_07_*_mean.csv` |

其他解释卡见 [索引](Q3_EXPLANATION_CARD_INDEX.csv)。不要把卡片中的近似秒数当真实时间标注。

自注意力热图显示编码器内部时间步之间的信息交互；解释卡的色带显示门值和局部遮挡结合后的证据重要性。论文中的正式模态贡献仍以 LOMO（逐模态删除）为准。
