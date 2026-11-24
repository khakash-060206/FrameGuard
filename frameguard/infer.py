"""UI-independent image, recorded-video, and live-frame inference."""

import json
import math
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
import torch

from .media import FaceDetector, SceneCuts, normalise, read_image, video_frames
from .model import load_checkpoint
from .temporal import SAMPLE_FPS, Tracker


class Predictor:
    def __init__(self, checkpoint, detector_path, device="cpu"):
        self.model, self.metadata = load_checkpoint(checkpoint, device)
        self.detector = FaceDetector(detector_path)
        self.device = device
        self.tracker, self.cuts = Tracker(), SceneCuts()

    def close(self):
        self.detector.close()

    def assess(self, frame, timestamp=None):
        started = time.perf_counter()
        faces = self.detector.detect(frame)
        temporal = timestamp is not None and self.metadata["stage"] == "temporal"
        ids = self.tracker.update([box for box, _ in faces], timestamp, self.cuts.update(frame)) if timestamp is not None else list(range(len(faces)))
        results = []
        if faces:
            with torch.inference_mode():
                features = self.model.encode(normalise(np.stack([crop for _, crop in faces])).to(self.device))
                spatial = self.model.image_head(features).squeeze(-1).sigmoid().cpu().tolist()
                for index, ((box, _), key, score) in enumerate(zip(faces, ids, spatial)):
                    mode, start = "spatial", None
                    state = "collecting_frames" if temporal else "spatial_only"
                    threshold = self.metadata["threshold_spatial"]
                    if temporal:
                        window = self.tracker.tracks[key].append(timestamp, features[index].detach())
                        if window:
                            sequence = torch.stack([value for _, value in window])[None]
                            score = self.model.temporal(sequence).sigmoid().item()
                            mode, state, start = "temporal", "window_prediction", window[0][0]
                            threshold = self.metadata["threshold_temporal"]
                        elif self.tracker.tracks[key].count >= 16:
                            state = "between_window_updates"
                    results.append({"track_id": key, "box": list(box), "mode": mode, "state": state,
                                    "spatial_score": spatial[index], "score": float(score), "calibrated": False,
                                    "label": "manipulated" if score >= threshold else "real", "threshold": threshold,
                                    "window_start": start, "window_end": timestamp if mode == "temporal" else None})
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return {"timestamp": timestamp, "status": "assessed" if results else "unable_to_assess", "faces": results,
                "processing_ms": (time.perf_counter() - started) * 1000}

    def heatmaps(self, frame, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        for index, (_, crop) in enumerate(self.detector.detect(frame)):
            with torch.enable_grad():
                value = normalise(crop[None]).to(self.device).requires_grad_(True)
                activations = self.model.features(value)
                logit = self.model.image_head(self.model.pool(activations).flatten(1)).sum()
                gradients = torch.autograd.grad(logit, activations)[0]
                cam = (gradients.mean((2, 3), keepdim=True) * activations).sum(1).relu()[0]
                cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam /= max(float(cam.max()), 1e-8)
            coloured = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), 0.6, coloured, 0.4, 0)
            ok, encoded = cv2.imencode(".png", overlay)
            if not ok:
                raise ValueError("Unable to encode Grad-CAM output")
            encoded.tofile(folder / f"face_{index:02d}.png")


def webcam_frames(camera=0, duration=30):
    """A one-frame queue prevents an unbounded Python-side live backlog."""
    if duration <= 0:
        raise ValueError("Webcam duration must be positive")
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot open webcam")
    latest, stop = queue.Queue(maxsize=1), threading.Event()
    started = time.perf_counter()

    def capture_loop():
        try:
            while not stop.is_set() and time.perf_counter() - started < duration:
                ok, frame = capture.read()
                if not ok:
                    break
                try:
                    latest.get_nowait()
                except queue.Empty:
                    pass
                latest.put_nowait((time.perf_counter() - started, frame))
        finally:
            stop.set()

    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    next_time = 0.0
    try:
        while not stop.is_set() or not latest.empty():
            try:
                timestamp, frame = latest.get(timeout=0.2)
            except queue.Empty:
                continue
            if timestamp >= next_time:
                yield timestamp, frame
                next_time = timestamp + 1 / SAMPLE_FPS
    finally:
        stop.set()
        thread.join(timeout=2)
        capture.release()


# Scores within this many log-odds of the threshold are reported as "uncertain". Fixed a priori (not tuned on
# test data); on the Run 2 test set it flags 6.3% of videos, whose accuracy is 59.5% versus 97.1% for the rest.
UNCERTAIN_MARGIN = 1.5


def uncertain_band(threshold, margin=UNCERTAIN_MARGIN):
    """Score interval (low, high) treated as too close to the threshold to call."""
    centre = math.log(threshold / (1 - threshold))
    return tuple(1 / (1 + math.exp(-(centre + offset))) for offset in (-margin, margin))


def verdict(records, margin=UNCERTAIN_MARGIN):
    """Media-level verdict scored like evaluation: mean of every temporal window score against the
    validation-selected temporal threshold. Images and clips too short for a window fall back to the
    mean spatial frame score against the spatial threshold. Scores inside the uncertainty band
    around the threshold give "uncertain" rather than a forced real/manipulated call."""
    faces = [face for record in records for face in record["faces"]]
    if not faces:
        return {"label": "unable_to_assess", "mode": None, "score": None, "threshold": None, "units": 0, "max_faces": 0}
    windows = [face for face in faces if face["mode"] == "temporal"]
    used = windows or faces
    score, threshold = float(np.mean([face["score"] for face in used])), used[0]["threshold"]
    low, high = uncertain_band(threshold, margin)
    label = "uncertain" if low < score < high else "manipulated" if score >= threshold else "real"
    return {"label": label, "mode": "temporal" if windows else "spatial",
            "score": score, "threshold": threshold, "uncertain_range": [low, high], "units": len(used),
            "max_faces": max(len(record["faces"]) for record in records), "calibrated": False}


def predict(source, checkpoint, detector_path, output, device="cpu", heatmaps=None, webcam=False, duration=30,
            max_samples=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output path to preserve previous predictions")
    predictor = Predictor(checkpoint, detector_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            if not webcam and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                frame = read_image(source)
                stream.write(json.dumps(predictor.assess(frame), allow_nan=False) + "\n")
                if heatmaps:
                    predictor.heatmaps(frame, heatmaps)
            else:
                if heatmaps:
                    raise ValueError("CLI heatmap export currently supports images only")
                frames = webcam_frames(int(source), duration) if webcam else video_frames(source, max_samples)
                try:
                    for timestamp, frame in frames:
                        stream.write(json.dumps(predictor.assess(frame, timestamp), allow_nan=False) + "\n")
                        stream.flush()
                finally:
                    frames.close()
    finally:
        predictor.close()

