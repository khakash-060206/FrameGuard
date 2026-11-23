"""Decode a validated video manifest into explicitly labelled face tracks."""

from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .manifest import file_sha256, read_manifest
from .media import FaceDetector, SceneCuts, video_frames
from .model import ModelConfig
from .temporal import Tracker

_WORKER = {}


def _start_worker(detector_path, output, max_samples, pooled=False):
    if pooled:
        import cv2
        cv2.setNumThreads(1)  # one decode thread per process; parallelism comes from the process pool
    _WORKER.update(detector=FaceDetector(detector_path), output=Path(output), max_samples=max_samples)


def _prepare_one(item):
    """Returns (accepted: bool, row) for one manifest row; runs inside a worker or the main process."""
    index, row = item
    detector, output, max_samples = _WORKER["detector"], _WORKER["output"], _WORKER["max_samples"]
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
        return True, {**row, "file": filename, "crops_file": crops_file, "frames": len(crops), "no_face_samples": no_faces}
    except (ValueError, OSError) as error:
        return False, {**row, "reason": str(error)}


def prepare(manifest, output, detector_path, max_samples=256, workers=1):
    if max_samples < 16:
        raise ValueError("max_samples must be at least 16")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    rows = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prepared, rejected = [], []
    started = time.perf_counter()
    pool = None
    try:
        if workers == 1:
            _start_worker(detector_path, output, max_samples)
            results = map(_prepare_one, enumerate(rows))
        else:
            pool = multiprocessing.get_context("spawn").Pool(workers, _start_worker, (detector_path, str(output), max_samples, True))
            results = pool.imap(_prepare_one, enumerate(rows), chunksize=4)  # ordered, so the index stays deterministic
        for done, (accepted, entry) in enumerate(results, start=1):
            (prepared if accepted else rejected).append(entry)
            if done % 25 == 0 or done == len(rows):
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (len(rows) - done)
                print(f"Prepared {done}/{len(rows)}: accepted={len(prepared)} rejected={len(rejected)} | "
                      f"{done / elapsed:.2f} videos/s | elapsed {elapsed / 60:.0f} min | ETA {eta / 60:.0f} min", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
        elif "detector" in _WORKER:
            _WORKER.pop("detector").close()
        report = {"format": 1, "manifest_sha256": file_sha256(manifest), "config": asdict(ModelConfig()),
                  "max_samples": max_samples, "videos": prepared, "rejected": rejected}
        (output / "index.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not prepared:
        raise ValueError(f"No usable videos; inspect {output / 'index.json'}")
