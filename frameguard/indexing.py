"""Index official dataset layouts; no dataset downloads or fabricated labels."""

import csv
import json
from pathlib import Path

from .manifest import FIELDS, read_manifest


def write_manifest(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Manifest already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        validated = read_manifest(temporary)
        temporary.rename(output)
        return validated
    finally:
        temporary.unlink(missing_ok=True)


def merge_manifests(manifests, output):
    rows = []
    for path in manifests:
        rows.extend(read_manifest(path))
    return write_manifest(rows, output)


def index_ffpp(root, splits, output, compression="c23", methods=("Deepfakes",)):
    root, splits = Path(root).resolve(), Path(splits)
    partitions = {}
    for split, filename in (("train", "train.json"), ("val", "val.json"), ("test", "test.json")):
        pairs = json.loads((splits / filename).read_text(encoding="utf-8"))
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Expected official FF++ split pairs")
            for source in pair:
                source = str(source).zfill(3)
                if partitions.setdefault(source, split) != split:
                    raise ValueError(f"Source {source} appears in multiple official splits")
    rows = []
    for method, directory, label in [("original", root / "original_sequences/youtube" / compression / "videos", 0)] + [
        (method, root / "manipulated_sequences" / method / compression / "videos", 1) for method in methods]:
        if not directory.is_dir():
            raise ValueError(f"Missing dataset directory: {directory}")
        for path in sorted(directory.glob("*.mp4")):
            sources = path.stem.split("_")
            if len(sources) != (2 if label else 1) or any(source not in partitions for source in sources):
                raise ValueError(f"Cannot map source identifiers: {path}")
            assigned = {partitions[source] for source in sources}
            if len(assigned) != 1:
                raise ValueError(f"Manipulated video crosses split boundaries: {path}")
            rows.append({"path": str(path), "dataset": "FaceForensics++", "video_id": f"{method}/{path.stem}",
                         "source_ids": "|".join(sources), "identity_ids": "", "label": label, "split": assigned.pop()})
    return write_manifest(rows, output)


def index_celeb(root, output):
    root = Path(root).resolve()
    rows = []
    # The official list's first column uses 1=real; our manifest deliberately uses 1=manipulated.
    for line in (root / "List_of_testing_videos.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, relative = line.split(maxsplit=1)
        if label not in {"0", "1"}:
            raise ValueError("Invalid Celeb-DF test-list label")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dataset entry points outside its root")
        identities = [part for part in path.stem.split("_") if part.startswith("id") and part[2:].isdigit()]
        rows.append({"path": str(path), "dataset": "Celeb-DF-v2", "video_id": relative.replace("\\", "/"),
                     "source_ids": path.stem, "identity_ids": "|".join(identities), "label": 1 - int(label), "split": "external"})
    return write_manifest(rows, output)

