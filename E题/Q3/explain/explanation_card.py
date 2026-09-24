from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


MODALITIES = ("text", "audio", "vision")


def save_explanation_card(path, sample_id, prediction, importance, evidence, spans, visual_frame=None):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(5, 3, width_ratios=[3, 1.2, 2], height_ratios=[0.8, 1, 1, 1, 0.65], hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, :2])
    ax.axis("off")
    title = (f"Sample {sample_id} | {prediction['pred_label']} | intensity {prediction['pred_intensity']:.3f}\n"
             f"P(Neg/Neu/Pos) = {prediction['prob_negative']:.3f} / {prediction['prob_neutral']:.3f} / {prediction['prob_positive']:.3f}")
    ax.text(0, 0.7, title, va="center", fontsize=12)
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(MODALITIES, importance, color=["#4477AA", "#EE6677", "#228833"])
    ax.set_ylim(0, 1)
    ax.set_title(f"LOMO importance | main: {prediction['main_modality']}", fontsize=10)
    for row, modality in enumerate(MODALITIES, 1):
        ax = fig.add_subplot(gs[row, 0])
        data = np.asarray(evidence[modality]["score"])[None, :]
        ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, extent=(0, 50, 0, 1))
        ax.set_yticks([])
        ax.set_xlim(0, 50)
        ax.set_ylabel(modality.title())
        if row == 3:
            ax.set_xlabel("Aligned grid position (exact)")
        note_ax = fig.add_subplot(gs[row, 1])
        note_ax.axis("off")
        lines = []
        for span in spans[modality]:
            label = f"#{span['rank']} [{span['grid_start']},{span['grid_end_exclusive']})"
            if span.get("start_sec_approx") is not None:
                label += f" ~{span['start_sec_approx']:.2f}-{span['end_sec_approx']:.2f}s"
            lines.append(label)
        note_ax.text(0, 0.5, "\n".join(lines), transform=note_ax.transAxes, va="center", fontsize=9)
    if visual_frame and Path(visual_frame).is_file():
        ax = fig.add_subplot(gs[1:4, 2])
        ax.imshow(mpimg.imread(visual_frame))
        ax.axis("off")
        ax.set_title("Main visual evidence frame")
    text_spans = spans.get("text", [])
    fragment = text_spans[0].get("text_fragment_approx", "") if text_spans else ""
    footer = fig.add_subplot(gs[4, :])
    footer.axis("off")
    footer.text(0, 0.75, f"Approximate text: {fragment[:120]}", fontsize=9, transform=footer.transAxes)
    footer.text(0, 0.20, "grid position exact; video/text projection approximate due to unavailable official timestamps", fontsize=8, transform=footer.transAxes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
