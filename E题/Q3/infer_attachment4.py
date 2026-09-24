import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from data.attachment4_dataset import Attachment4AlignedDataset
from data.feature_baseline import load_feature_means
from explain.back_projection import MAPPING_NOTE, TEXT_METHOD, TIME_METHOD, project_span, regex_tokenize, save_keyframe, video_metadata
from explain.evidence_segments import evidence_spans
from explain.explanation_card import save_explanation_card
from explain.local_occlusion import local_evidence_scores
from explain.modality_ablation import MODALITIES, modality_importance
from utils.io import get_device, load_checkpoint, load_config


LABELS = ("Negative", "Neutral", "Positive")


@torch.no_grad()
def infer_attachment4(cfg, checkpoint):
    output = Path(cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg)
    model, _ = load_checkpoint(checkpoint, cfg, device)
    means = load_feature_means(output / "train_feature_baselines.npz")
    dataset = Attachment4AlignedDataset(cfg["paths"]["attachment4_aligned_dir"], cfg["paths"]["attachment4_video_dir"])
    summary_rows, span_rows, mapping_rows = [], [], []
    for sample in tqdm(dataset, desc="attachment4 inference"):
        sample_id = sample["sample_id"]
        batch = {k: sample[k].unsqueeze(0).to(device) for k in (*MODALITIES, "valid_mask")}
        valid_len = int(batch["valid_mask"][0].sum())
        metadata = video_metadata(sample["video_path"])
        mapping_warnings = [metadata["warning"]] if metadata["warning"] else []
        lomo = modality_importance(model, batch, means, cfg)
        full = lomo["full"]
        local = local_evidence_scores(model, batch, means, cfg, full)
        importance = lomo["importance"][0].float().cpu().numpy()
        router = full["router_alpha"][0].float().cpu().numpy()
        probabilities = full["cls_prob"][0].float().cpu().numpy()
        class_id = int(probabilities.argmax())
        main_modality = MODALITIES[int(importance.argmax())]
        spans = {}
        for i, modality in enumerate(MODALITIES):
            spans[modality] = evidence_spans(modality, local[modality], valid_len, cfg)
            if not spans[modality]:
                raise ValueError(f"No evidence span for {sample_id}/{modality}")
            for span in spans[modality]:
                span.update(project_span(span, valid_len, sample["raw_text"], metadata["duration"]))
                keyframe = ""
                if modality == "vision":
                    keyframe, fallback = save_keyframe(sample["video_path"], span, metadata, output / "evidence_frames" / f"{sample_id}_vision_rank{span['rank']}.jpg")
                    if fallback:
                        mapping_warnings.append(f"keyframe_seek_fallback_rank{span['rank']}")
                span["keyframe_path"] = keyframe
                span_rows.append({
                    "sample_id": sample_id, "modality": modality, "modality_importance": float(importance[i]),
                    "rank": span["rank"], "grid_start": span["grid_start"],
                    "grid_end_exclusive": span["grid_end_exclusive"], "span_length": span["length"],
                    "mean_gate": span["mean_gate"], "mean_occlusion": span["mean_occlusion"],
                    "evidence_score": span["evidence_score"],
                    "start_sec_approx": span["start_sec_approx"], "end_sec_approx": span["end_sec_approx"],
                    "audio_start_sec_approx": span["start_sec_approx"] if modality == "audio" else "",
                    "audio_end_sec_approx": span["end_sec_approx"] if modality == "audio" else "",
                    "text_fragment_approx": span["text_fragment_approx"] if modality == "text" else "",
                    "keyframe_path": keyframe, "mapping_method": TIME_METHOD,
                })
        main = spans[main_modality][0]
        prediction = {
            "sample_id": sample_id, "pred_class_id": class_id, "pred_label": LABELS[class_id],
            "prob_negative": float(probabilities[0]), "prob_neutral": float(probabilities[1]),
            "prob_positive": float(probabilities[2]), "pred_intensity": float(full["reg_pred"][0]),
            "main_modality": main_modality,
            **{f"importance_{m}": float(importance[i]) for i, m in enumerate(MODALITIES)},
            **{f"router_{m}": float(router[i]) for i, m in enumerate(MODALITIES)},
            "main_grid_start": main["grid_start"], "main_grid_end_exclusive": main["grid_end_exclusive"],
            "main_start_sec_approx": main["start_sec_approx"], "main_end_sec_approx": main["end_sec_approx"],
            "main_text_fragment_approx": spans["text"][0]["text_fragment_approx"],
            "main_visual_keyframe": spans["vision"][0]["keyframe_path"],
            "audio_start_sec_approx": spans["audio"][0]["start_sec_approx"],
            "audio_end_sec_approx": spans["audio"][0]["end_sec_approx"],
            "mapping_note": MAPPING_NOTE,
        }
        signed_cls = lomo["cls_prob_delta_signed"][0].float().cpu().numpy()
        signed_reg = lomo["reg_delta_signed"][0].float().cpu().numpy()
        for i, modality in enumerate(MODALITIES):
            prediction[f"cls_prob_delta_signed_{modality}"] = float(signed_cls[i])
            prediction[f"reg_delta_signed_{modality}"] = float(signed_reg[i])
        summary_rows.append(prediction)
        mapping_rows.append({"sample_id": sample_id, "video_duration_sec": metadata["duration"],
                             "valid_len": valid_len, "raw_text_word_count": len(regex_tokenize(sample["raw_text"])),
                             "time_mapping_method": TIME_METHOD, "text_mapping_method": TEXT_METHOD,
                             "warning": ";".join(["official token and position timestamps unavailable", *mapping_warnings])})
        save_explanation_card(output / "explanation_cards" / f"{sample_id}.png", sample_id, prediction,
                              importance, local, spans, spans["vision"][0]["keyframe_path"])
    pd.DataFrame(summary_rows).to_csv(output / "attachment4_predictions_explanations.csv", index=False)
    pd.DataFrame(span_rows).to_csv(output / "attachment4_explanations_long.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(output / "attachment4_mapping_notes.csv", index=False)
    print(f"Attachment 4: {len(summary_rows)} samples, {len(span_rows)} spans", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q3_aligned.yaml")
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    infer_attachment4(load_config(args.config), args.checkpoint)
