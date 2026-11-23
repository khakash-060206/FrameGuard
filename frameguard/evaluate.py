"""Compare spatial averaging and temporal predictions on identical windows."""

import json
from pathlib import Path

import numpy as np
import torch

from .data import Samples
from .metrics import classification_metrics, select_threshold, diagnostic_curves
from .model import load_checkpoint
from .train import video_predictions


def spatial_window_predictions(dataset):
    grouped = {}
    for video, selected in dataset.samples:
        grouped.setdefault(video, []).append(selected)
    result = []
    for video, windows in grouped.items():
        row = dataset.videos[video]
        with np.load(dataset.folder / row["file"], allow_pickle=False) as data:
            scores = torch.from_numpy(data["spatial_logits"]).sigmoid().numpy()
            score = float(np.mean([scores[indices].mean() for indices in windows]))
        result.append({"dataset": row["dataset"], "video_id": row["video_id"], "label": row["label"], "score": score})
    return result


FFPP = "FaceForensics++"


def assign_domains(predictions, training_groups):
    """Tag each prediction with the result groups it counts towards.

    Each dataset; "All datasets" when several are present; and, when FF++ has several manipulation
    methods, one group per method (that method's fakes plus all FF++ real videos). Fakes from a method
    absent from training/validation count only towards their "(unseen method)" group, never the totals.
    """
    trained = {row["video_id"].split("/")[0] for row in training_groups if row["dataset"] == FFPP}
    method = lambda p: p["video_id"].split("/")[0]
    methods = sorted({method(p) for p in predictions if p["dataset"] == FFPP and p["label"] == 1})
    name = lambda m: f"FF++ {m}" + ("" if m in trained else " (unseen method)")
    several = len({p["dataset"] for p in predictions}) > 1
    for p in predictions:
        ffpp_method = method(p) if p["dataset"] == FFPP else None
        unseen = p["label"] == 1 and ffpp_method is not None and ffpp_method not in trained
        domains = [] if unseen else [p["dataset"]] + (["All datasets"] if several else [])
        if ffpp_method and len(methods) > 1:
            domains += [name(m) for m in methods] if p["label"] == 0 else [name(ffpp_method)]
        p["domains"] = domains
    return predictions


def evaluate(folder, checkpoint, output, split="test", device="cpu", batch_size=8):
    if split not in {"test", "external"}:
        raise ValueError("Final evaluation accepts test or external only")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new report path to preserve previous evaluation")
    model, metadata = load_checkpoint(checkpoint, device)
    if metadata["stage"] != "temporal":
        raise ValueError("Paired evaluation requires a trained temporal checkpoint")
    dataset = Samples(folder, split, temporal=True)
    if metadata["spatial_checkpoint_sha256"] != dataset.index["checkpoint_sha256"]:
        raise ValueError("Evaluation cache uses a different spatial backbone")
    def validation_groups(rows):
        return sorted((row["dataset"], row["video_id"], row["source_ids"], row["identity_ids"])
                      for row in rows if row["split"] == "val")
    if validation_groups(metadata["training_groups"]) != validation_groups(dataset.index["videos"]):
        raise ValueError("Use the original validation cohort for baseline threshold selection")
    used_groups = set()
    for row in metadata["training_groups"]:
        for field in ("source_ids", "identity_ids"):
            used_groups.update((row["dataset"], field, value) for value in row[field].split("|") if value)
    for row in dataset.videos:
        for field in ("source_ids", "identity_ids"):
            if any((row["dataset"], field, value) in used_groups for value in row[field].split("|") if value):
                raise ValueError(f"Evaluation overlaps training/validation: {row['video_id']}")
    # Spatial averaging over windows differs from averaging every frame; tune its threshold on validation windows only.
    validation = Samples(folder, "val", temporal=True)
    spatial_val = spatial_window_predictions(validation)
    spatial_threshold = select_threshold([p["label"] for p in spatial_val], [p["score"] for p in spatial_val])
    spatial = spatial_window_predictions(dataset)
    temporal, temporal_loss = video_predictions(model, dataset, batch_size, device)
    metrics, curves = {}, {}
    for name, predictions, threshold in (("spatial", spatial, spatial_threshold),
                                          ("temporal", temporal, metadata["threshold_temporal"])):
        metrics[name] = {}
        curves[name] = {}
        assign_domains(predictions, metadata["training_groups"])
        for domain in sorted({d for p in predictions for d in p["domains"]}):
            selected = [p for p in predictions if domain in p["domains"]]
            if len({p["label"] for p in selected}) < 2:
                continue  # ROC-AUC needs both classes
            metrics[name][domain] = classification_metrics([p["label"] for p in selected], [p["score"] for p in selected], threshold)
            curves[name][domain] = diagnostic_curves([p["label"] for p in selected], [p["score"] for p in selected])
    eligible = {p["video_id"] + "@" + p["dataset"] for p in temporal}
    excluded = [{"dataset": row["dataset"], "video_id": row["video_id"], "reason": "No continuous temporal window"}
                for row in dataset.videos if row["video_id"] + "@" + row["dataset"] not in eligible]
    excluded += [row for row in dataset.index["rejected"] if row["split"] == split]
    report = {"split": split, "unit": "video", "calibrated": False, "metrics": metrics, "curves": curves,
              "temporal_window_loss": temporal_loss,
              "spatial_predictions": spatial, "temporal_predictions": temporal, "unassessed": excluded,
              "manifest_sha256": dataset.index["manifest_sha256"], "checkpoint": str(checkpoint),
              "note": "Only identical continuous windows are compared. This does not include an external detector."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
