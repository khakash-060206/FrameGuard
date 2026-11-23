"""Video-balanced training data and checkpoint-specific feature extraction."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .manifest import file_sha256
from .media import normalise
from .model import ModelConfig, load_checkpoint
from .temporal import window_starts


def read_index(folder, kind="crops"):
    folder = Path(folder)
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    if index.get("format") != 1 or index.get("config") != asdict(ModelConfig()):
        raise ValueError("Incompatible prepared data configuration")
    if kind == "features" and not index.get("checkpoint_sha256"):
        raise ValueError("Temporal training requires a checkpoint-specific feature cache")
    for row in index["videos"]:
        file = (folder / row["file"]).resolve()
        if not file.is_relative_to(folder.resolve()) or not file.is_file():
            raise ValueError(f"Invalid prepared data path: {file}")
        if kind == "crops" and row.get("crops_file"):
            crop_file = (folder / row["crops_file"]).resolve()
            if not crop_file.is_relative_to(folder.resolve()) or not crop_file.is_file():
                raise ValueError(f"Invalid crop path: {crop_file}")
    return index


class Samples(Dataset):
    def __init__(self, folder, split, temporal=False):
        self.folder, self.temporal = Path(folder), temporal
        self.index = read_index(folder, "features" if temporal else "crops")
        self.videos = [row for row in self.index["videos"] if row["split"] == split]
        self.samples = []
        for video, row in enumerate(self.videos):
            with np.load(self.folder / row["file"], allow_pickle=False) as data:
                if temporal:
                    for track in np.unique(data["tracks"]):
                        indices = np.flatnonzero(data["tracks"] == track)
                        for start in window_starts(data["timestamps"][indices]):
                            self.samples.append((video, indices[start:start + 16]))
                else:
                    count = row["frames"] if row.get("crops_file") else len(data["crops"])
                    self.samples.extend((video, i) for i in range(count))
        self.cached_video, self.cached_data = None, None
        if not self.samples:
            raise ValueError(f"No {'continuous windows' if temporal else 'face crops'} for split {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, selected = self.samples[index]
        if video != self.cached_video:
            row = self.videos[video]
            if not self.temporal and row.get("crops_file"):
                self.cached_data = np.load(self.folder / row["crops_file"], mmap_mode="r", allow_pickle=False)
            else:
                with np.load(self.folder / row["file"], allow_pickle=False) as data:
                    self.cached_data = data["features" if self.temporal else "crops"].copy()
            self.cached_video = video
        if self.temporal:
            value = torch.from_numpy(self.cached_data[selected].copy()).float()
        else:
            value = normalise(self.cached_data[selected][None].copy())[0]
        return value, torch.tensor(self.videos[video]["label"], dtype=torch.float32), video

    def balanced_sampler(self):
        counts = Counter(video for video, _ in self.samples)
        classes = Counter(self.videos[v]["label"] for v in counts)
        if set(classes) != {0, 1}:
            raise ValueError("Training requires videos from both classes")
        weights = [1 / (counts[v] * classes[self.videos[v]["label"]]) for v, _ in self.samples]
        return WeightedRandomSampler(weights, len(weights), replacement=True)


def cache_features(folder, checkpoint, output, device="cpu"):
    folder, output = Path(folder), Path(output)
    index = read_index(folder)
    model, metadata = load_checkpoint(checkpoint, device)
    if metadata["stage"] != "spatial":
        raise ValueError("Feature cache must use the spatial baseline checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    for row in index["videos"]:
        with np.load(folder / row["file"], allow_pickle=False) as data:
            features, logits = [], []
            crops = np.load(folder / row["crops_file"], mmap_mode="r", allow_pickle=False) if row.get("crops_file") else data["crops"]
            with torch.inference_mode():
                for start in range(0, len(crops), 16):
                    encoded = model.encode(normalise(crops[start:start + 16].copy()).to(device))
                    features.append(encoded.cpu().numpy())
                    logits.append(model.image_head(encoded).squeeze(-1).cpu().numpy())
            np.savez_compressed(output / row["file"], features=np.concatenate(features),
                                spatial_logits=np.concatenate(logits), timestamps=data["timestamps"], tracks=data["tracks"])
        completed.append(row)
        print(f"Cached {len(completed)}/{len(index['videos'])}", flush=True)
    index.update(videos=completed, checkpoint_sha256=file_sha256(checkpoint), augmentation="none")
    (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
