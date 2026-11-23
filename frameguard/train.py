"""Staged CNN and LSTM training; no real-data claim is made by software tests."""

import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Samples
from .manifest import file_sha256
from .metrics import classification_metrics, select_threshold
from .model import FrameGuard, load_checkpoint, save_checkpoint


def video_predictions(model, dataset, batch_size, device):
    scores = {}
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for values, labels, videos in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = model.temporal(values.to(device)) if dataset.temporal else model(values.to(device))
            total_loss += nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum").item()
            for video, score in zip(videos.tolist(), logits.sigmoid().cpu().tolist()):
                scores.setdefault(video, []).append(score)
    predictions = [{"dataset": dataset.videos[v]["dataset"], "video_id": dataset.videos[v]["video_id"],
             "label": dataset.videos[v]["label"], "score": float(np.mean(values)), "samples": len(values)}
            for v, values in sorted(scores.items())]
    return predictions, total_loss / len(dataset)


def train(folder, output, stage, checkpoint=None, epochs=5, batch_size=8, device="cpu", seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    temporal = stage == "temporal"
    training, validation = Samples(folder, "train", temporal), Samples(folder, "val", temporal)
    if len({validation.videos[v]["label"] for v, _ in validation.samples}) != 2:
        raise ValueError("Validation needs assessable videos from both classes")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new checkpoint path to preserve existing experiments")
    if temporal:
        if not checkpoint:
            raise ValueError("Temporal training requires --checkpoint with the spatial model")
        if training.index["checkpoint_sha256"] != file_sha256(checkpoint):
            raise ValueError("Cached features do not match the supplied spatial checkpoint")
        model, metadata = load_checkpoint(checkpoint, device)
        if metadata["stage"] != "spatial":
            raise ValueError("Temporal training must start from a spatial checkpoint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.lstm, model.sequence_head):
            module.requires_grad_(True)
    else:
        if checkpoint:
            raise ValueError("Spatial training starts from ImageNet, not a deepfake checkpoint")
        model, metadata = FrameGuard(pretrained=True).to(device), {}
        model.lstm.requires_grad_(False)
        model.sequence_head.requires_grad_(False)
        model.features.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(training, batch_size=batch_size, sampler=training.balanced_sampler(), num_workers=0)
    history, best_auc = [], -1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        if not temporal and epoch == 1:
            model.features.requires_grad_(True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.train()
        if temporal or epoch == 0:
            model.features.eval()
        total, correct = 0.0, 0
        for step, (values, labels, _) in enumerate(loader, start=1):
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.temporal(values) if temporal else model(values)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training produced nonfinite loss; no checkpoint written for this epoch")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
            correct += ((logits.detach() >= 0) == (labels >= 0.5)).sum().item()
            if step == 1 or step % 100 == 0:
                print(f"{stage} epoch {epoch + 1}/{epochs}, batch {step}/{len(loader)}, loss {loss.item():.4f}", flush=True)
        predictions, val_loss = video_predictions(model, validation, batch_size, device)
        labels, scores = [p["label"] for p in predictions], [p["score"] for p in predictions]
        threshold = select_threshold(labels, scores)
        metrics = classification_metrics(labels, scores, threshold)
        record = {"epoch": epoch + 1, "stage": stage, "training_loss": total / len(training),
                  "training_accuracy": correct / len(training), "validation_loss": val_loss, "validation": metrics,
                  "accuracy_note": "Training: sampled frames/windows at 0.5. Validation: videos at validation-selected threshold."}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            current = {**metadata, "stage": stage, "epoch": epoch + 1, "seed": seed,
                       "threshold_temporal" if temporal else "threshold_spatial": threshold,
                       "validation": metrics, "manifest_sha256": training.index["manifest_sha256"],
                       "training_groups": [{k: row[k] for k in ("dataset", "video_id", "source_ids", "identity_ids", "split")}
                                           for row in training.index["videos"] if row["split"] in {"train", "val"}],
                       "initialisation": "ImageNet EfficientNet-B0", "batch_size": batch_size,
                       "sampler": "class-balanced then video-balanced replacement", "device": device,
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "augmentation": "none", "calibrated": False}
            if temporal:
                current["spatial_checkpoint_sha256"] = file_sha256(checkpoint)
            save_checkpoint(output, model, current)
        history_path = output.with_suffix(".history.json")
        temporary = history_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(history_path)

