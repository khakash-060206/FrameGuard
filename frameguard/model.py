"""Our ImageNet-initialised CNN and newly trained temporal head."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    feature_size: int = 1280
    hidden_size: int = 128
    window: int = 16
    sample_fps: int = 8
    stride: int = 4
    crop_margin: float = 0.2


class FrameGuard(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.config = ModelConfig()
        network = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.features = network.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.image_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))
        self.lstm = nn.LSTM(1280, 128, batch_first=True)
        self.sequence_head = nn.Linear(128, 1)

    def encode(self, images):
        return self.pool(self.features(images)).flatten(1)

    def forward(self, images):
        return self.image_head(self.encode(images)).squeeze(-1)

    def temporal(self, features):
        if features.ndim != 3 or features.shape[1:] != (16, 1280):
            raise ValueError("Expected features with shape [batch, 16, 1280]")
        output, _ = self.lstm(features)
        return self.sequence_head(output[:, -1]).squeeze(-1)


def save_checkpoint(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save({"format": 1, "config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != 1 or payload.get("config") != asdict(ModelConfig()):
        raise ValueError("Unsupported FrameGuard checkpoint configuration")
    metadata = payload.get("metadata", {})
    if metadata.get("stage") not in {"spatial", "temporal"}:
        raise ValueError("Checkpoint is not a trained FrameGuard spatial/temporal checkpoint")
    required = ["threshold_spatial"] + (["threshold_temporal"] if metadata["stage"] == "temporal" else [])
    for key in required:
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Checkpoint lacks a valid validation threshold: {key}")
    model = FrameGuard()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), metadata
