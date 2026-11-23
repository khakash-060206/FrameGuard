"""Manifest validation stays dependency-free so data checks work before ML setup."""

import csv
import hashlib
from pathlib import Path

FIELDS = ("path", "dataset", "video_id", "source_ids", "identity_ids", "label", "split")
SPLITS = {"train", "val", "test", "external"}


def read_manifest(path, check_files=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {', '.join(FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no videos")
    seen_paths, seen_videos, groups = set(), set(), {}
    for number, row in enumerate(rows, start=2):
        if any(row.get(key) is None for key in FIELDS):
            raise ValueError(f"Incomplete manifest row {number}")
        for key in FIELDS:
            row[key] = row[key].strip()
        if any(not row[key] for key in FIELDS if key != "identity_ids"):
            raise ValueError(f"Missing required value on row {number}")
        if row["label"] not in {"0", "1"} or row["split"] not in SPLITS:
            raise ValueError(f"Invalid label or split on row {number}")
        row["label"] = int(row["label"])
        media = (path.parent / row["path"]).resolve()
        if media.suffix.lower() != ".mp4":
            raise ValueError(f"Training manifest expects MP4 videos: {media}")
        if check_files and not media.is_file():
            raise ValueError(f"Missing video: {media}")
        video_key = (row["dataset"], row["video_id"])
        if media in seen_paths or video_key in seen_videos:
            raise ValueError(f"Duplicate video: {row['video_id']}")
        seen_paths.add(media)
        seen_videos.add(video_key)
        row["path"] = str(media)
        for field in ("source_ids", "identity_ids"):
            ids = row[field].split("|") if row[field] else []
            if any(not item.strip() for item in ids):
                raise ValueError(f"Empty identifier in {field}, row {number}")
            for item in ids:
                key = (row["dataset"], field, item.strip())
                previous = groups.setdefault(key, row["split"])
                if previous != row["split"]:
                    raise ValueError(f"Split leakage for {key}: {previous} / {row['split']}")
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

