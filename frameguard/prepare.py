"""Decode a validated video manifest into explicitly labelled face tracks."""

import json
from pathlib import Path

import numpy as np

from .manifest import file_sha256, read_manifest
from .media import FaceDetector, SceneCuts, video_frames
from .model import ModelConfig
from .temporal import Tracker
from dataclasses import asdict


def prepare(manifest, output, detector_path, max_samples=256):
    if max_samples < 16:
        raise ValueError("max_samples must be at least 16")
    rows = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    detector = FaceDetector(detector_path)
    prepared, rejected = [], []
    try:
        for index, row in enumerate(rows):
            tracker, cuts = Tracker(), SceneCuts()
            crops, times, track_ids = [], [], []
            no_faces = 0
            try:
                for timestamp, frame in video_frames(row["path"], max_samples):
                    faces = detector.detect(frame)
                    if len(faces) > 1:
                        raise ValueError("Multiple usable faces: target identity is ambiguous; video excluded")
                    ids = tracker.update([box for box, _ in faces], timestamp, cuts.update(frame))
                    if not faces:
                        no_faces += 1
                    for key, (_, crop) in zip(ids, faces):
                        crops.append(crop)
                        times.append(timestamp)
                        track_ids.append(key)
                if not crops:
                    raise ValueError("No usable faces")
                filename = f"{index:06d}.npz"
                crops_file = f"{index:06d}.crops.npy"
                # Memory mapping avoids decompressing an entire video for each randomly sampled frame.
                np.save(output / crops_file, np.stack(crops), allow_pickle=False)
                np.savez_compressed(output / filename, timestamps=np.array(times), tracks=np.array(track_ids))
                prepared.append({**row, "file": filename, "crops_file": crops_file, "frames": len(crops), "no_face_samples": no_faces})
            except (ValueError, OSError) as error:
                rejected.append({**row, "reason": str(error)})
            print(f"Prepared {index + 1}/{len(rows)}: accepted={len(prepared)} rejected={len(rejected)}", flush=True)
    finally:
        detector.close()
        report = {"format": 1, "manifest_sha256": file_sha256(manifest), "config": asdict(ModelConfig()),
                  "max_samples": max_samples, "videos": prepared, "rejected": rejected}
        (output / "index.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not prepared:
        raise ValueError(f"No usable videos; inspect {output / 'index.json'}")

