"""Shared decoding and face preprocessing for training and inference."""

from pathlib import Path
import math

import cv2
import numpy as np
import torch

from .temporal import SAMPLE_FPS

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def normalise(crops):
    if crops.dtype != np.uint8 or crops.ndim != 4 or crops.shape[1:] != (224, 224, 3):
        raise ValueError("Expected uint8 RGB crops [batch, 224, 224, 3]")
    tensor = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float() / 255
    return (tensor - tensor.new_tensor(MEAN)[None, :, None, None]) / tensor.new_tensor(STD)[None, :, None, None]


class FaceDetector:
    def __init__(self, model_path):
        if not Path(model_path).is_file():
            raise ValueError(f"Missing MediaPipe face detection asset: {model_path}. See docs/RUNBOOK.md.")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)), min_detection_confidence=0.5))

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)).detections
        height, width = rgb.shape[:2]
        result = []
        for detection in detections:
            box = detection.bounding_box
            if min(box.width, box.height) < 32:
                continue
            x1 = max(0, int(box.origin_x - box.width * 0.2))
            y1 = max(0, int(box.origin_y - box.height * 0.2))
            x2 = min(width, int(box.origin_x + box.width * 1.2))
            y2 = min(height, int(box.origin_y + box.height * 1.2))
            if x2 - x1 >= 32 and y2 - y1 >= 32:
                crop = cv2.resize(rgb[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_AREA)
                result.append(((x1, y1, x2, y2), crop))
        return result

    def close(self):
        self.detector.close()


def read_image(path):
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Supported images: JPG, JPEG, PNG")
    if not path.is_file():
        raise ValueError(f"Missing image: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def video_frames(path, max_samples=None):
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an existing MP4: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps < SAMPLE_FPS:
            raise ValueError("Video needs valid frame-rate metadata and at least 8 FPS")
        index, emitted, next_time, previous = 0, 0, 0.0, -1.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or timestamp <= previous:
                # shortcut: missing timestamps use nominal FPS; validate variable-frame-rate files separately.
                timestamp = index / fps
            if timestamp <= previous:
                raise ValueError("Nonmonotonic video timestamps")
            previous = timestamp
            index += 1
            if timestamp + 1e-6 < next_time:
                continue
            yield timestamp, frame
            emitted += 1
            next_time = (math.floor(timestamp * SAMPLE_FPS + 1e-6) + 1) / SAMPLE_FPS
            if max_samples is not None and emitted >= max_samples:
                break
        if emitted == 0:
            raise ValueError(f"Video contains no decodable frames: {path}")
    finally:
        capture.release()


class SceneCuts:
    def __init__(self):
        self.previous = None

    def update(self, frame):
        current = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255
        # shortcut: a conservative brightness-change heuristic; validate cuts/tracking on real clips before deployment.
        cut = self.previous is not None and float(np.abs(current - self.previous).mean()) > 0.3
        self.previous = current
        return cut

