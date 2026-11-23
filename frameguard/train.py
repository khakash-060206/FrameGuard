"""Staged CNN and LSTM training; no real-data claim is made by software tests."""

from contextlib import nullcontext
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


def use_amp(device, amp):
    """Mixed precision only on CUDA; `amp=None` means automatic."""
    cuda = str(device).startswith("cuda")
    if amp and not cuda:
        raise ValueError("Mixed precision (--amp on) requires a CUDA device")
    return cuda if amp is None else bool(amp)


def autocast(amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()


def loader_options(device, workers):
    options = {"num_workers": workers, "pin_memory": str(device).startswith("cuda")}
    if workers > 0:
        options["persistent_workers"] = True
    return options


def video_predictions(model, dataset, batch_size, device, workers=0, amp=False):
    scores = {}
    total_loss = 0.0
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_options(device, workers))
    with torch.inference_mode(), autocast(amp):
        for values, labels, videos in loader:
            values = values.to(device, non_blocking=True)
            logits = (model.temporal(values) if dataset.temporal else model(values)).float()
            total_loss += nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum").item()
            for video, score in zip(videos.tolist(), logits.sigmoid().cpu().tolist()):
                scores.setdefault(video, []).append(score)
    predictions = [{"dataset": dataset.videos[v]["dataset"], "video_id": dataset.videos[v]["video_id"],
             "label": dataset.videos[v]["label"], "score": float(np.mean(values)), "samples": len(values)}
            for v, values in sorted(scores.items())]
    return predictions, total_loss / len(dataset)


def train(folder, output, stage, checkpoint=None, epochs=5, batch_size=8, device="cpu", seed=42,
          frames_per_video=None, workers=0, amp=None):
    """Train one stage.

    frames_per_video: spatial stage only. Draw this many crops per training video per epoch
        (class- and video-balanced, as before) instead of one pass over every crop. None keeps
        the original behaviour.
    workers: DataLoader worker processes.
    amp: float16 mixed precision on CUDA. None = on for CUDA, off for CPU.
    """
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs and batch_size must be positive; workers cannot be negative")
    if frames_per_video is not None and frames_per_video < 1:
        raise ValueError("frames_per_video must be positive")
    amp = use_amp(device, amp)
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
    videos_in_training = len({video for video, _ in training.samples})
    draws = frames_per_video * videos_in_training if frames_per_video and not temporal else None
    loader = DataLoader(training, batch_size=batch_size, sampler=training.balanced_sampler(draws),
                        **loader_options(device, workers))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_auc = [], -1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        if not temporal and epoch == 1:
            model.features.requires_grad_(True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.train()
        if temporal or epoch == 0:
            model.features.eval()
        total, correct, seen = 0.0, 0, 0
        for step, (values, labels, _) in enumerate(loader, start=1):
            values, labels = values.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(amp):
                logits = model.temporal(values) if temporal else model(values)
            logits = logits.float()
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Training produced nonfinite loss; no checkpoint written for this epoch")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            seen += len(labels)
            total += loss.item() * len(labels)
            correct += ((logits.detach() >= 0) == (labels >= 0.5)).sum().item()
            if step == 1 or step % 100 == 0:
                print(f"{stage} epoch {epoch + 1}/{epochs}, batch {step}/{len(loader)}, loss {loss.item():.4f}", flush=True)
        predictions, val_loss = video_predictions(model, validation, batch_size, device, workers, amp)
        labels, scores = [p["label"] for p in predictions], [p["score"] for p in predictions]
        threshold = select_threshold(labels, scores)
        metrics = classification_metrics(labels, scores, threshold)
        record = {"epoch": epoch + 1, "stage": stage, "training_loss": total / seen,
                  "training_accuracy": correct / seen, "training_samples": seen, "validation_loss": val_loss, "validation": metrics,
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
                       "frames_per_video": None if temporal else frames_per_video, "samples_per_epoch": seen,
                       "mixed_precision": amp,
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "augmentation": "none", "calibrated": False}
            if temporal:
                current["spatial_checkpoint_sha256"] = file_sha256(checkpoint)
            save_checkpoint(output, model, current)
        history_path = output.with_suffix(".history.json")
        temporary = history_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(history_path)

