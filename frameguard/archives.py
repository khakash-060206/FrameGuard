"""Import the supplied archive layouts without mixing related source videos."""

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import zipfile

from .indexing import write_manifest


def grouped_splits(sources, pairs, seed=42):
    parent = {source: source for source in sources}

    def find(value):
        if value not in parent:
            raise ValueError(f"Manipulation refers to missing original {value}")
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for first, second in pairs:
        parent[find(first)] = find(second)
    groups = {}
    for source in sorted(sources):
        groups.setdefault(find(source), []).append(source)
    ordered = sorted(groups.values())
    if len(ordered) < 7:
        raise ValueError("Need at least seven independent source groups")
    random.Random(seed).shuffle(ordered)
    train_end, val_end = int(len(ordered) * 0.7), int(len(ordered) * 0.85)
    return {source: "train" if index < train_end else "val" if index < val_end else "test"
            for index, group in enumerate(ordered) for source in group}


def extract_member(archive, name, root):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive entry escapes extraction directory")
    expected = archive.getinfo(name).file_size
    if target.exists():
        raise ValueError(f"Extraction target exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        with archive.open(name) as source, temporary.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        if temporary.stat().st_size != expected:
            raise ValueError(f"Incomplete extraction: {name}")
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class ZipSource:
    """Dataset supplied as a ZIP archive; selected videos are extracted with CRC checks."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(self.path)
        self.names = set(self.archive.namelist())

    def size(self, name):
        return self.archive.getinfo(name).file_size

    def read_text(self, name):
        return self.archive.read(name).decode("utf-8-sig")

    def video(self, name, root):
        return extract_member(self.archive, name, root)

    def describe(self):
        return {"name": self.path.name, "type": "zip", "bytes": self.path.stat().st_size}

    def close(self):
        self.archive.close()


class FolderSource:
    """Dataset supplied as an already-extracted folder (e.g. a Kaggle input); videos are used in place."""

    def __init__(self, path, marker):
        self.root = find_root(Path(path), marker)
        self.names = {file.relative_to(self.root).as_posix() for file in self.root.rglob("*") if file.is_file()}

    def size(self, name):
        return (self.root / name).stat().st_size

    def read_text(self, name):
        return (self.root / name).read_text(encoding="utf-8-sig")

    def video(self, name, root):
        return (self.root / Path(*PurePosixPath(name).parts)).resolve()

    def describe(self):
        return {"name": str(self.root), "type": "folder"}

    def close(self):
        pass


def find_root(path, marker, depth=4):
    """Return the folder that directly contains `marker`, searching a few levels below `path`."""
    if not path.is_dir():
        raise ValueError(f"Dataset folder not found: {path}")
    level = [path]
    for _ in range(depth + 1):
        for folder in level:
            if (folder / marker).exists():
                return folder
        level = sorted(child for folder in level for child in folder.iterdir() if child.is_dir())
    raise ValueError(f"Could not find {marker} inside {path}")


def open_source(path, marker):
    path = Path(path)
    return FolderSource(path, marker) if path.is_dir() else ZipSource(path)


def import_archives(ffpp, celeb, output, seed=42):
    """Build the leakage-checked manifest from FF++ C23 and Celeb-DF v2, each given as a ZIP or a folder.

    ZIP inputs are extracted into `output`. Folder inputs are referenced in place, so nothing is copied.
    Both forms produce the same video IDs and the same seeded split.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing data will not be overwritten")
    first, second = open_source(ffpp, "FaceForensics++_C23"), None
    try:
        second = open_source(celeb, "List_of_testing_videos.txt")
        original = sorted(n for n in first.names if n.startswith("FaceForensics++_C23/original/") and n.endswith(".mp4"))
        fake = sorted(n for n in first.names if n.startswith("FaceForensics++_C23/Deepfakes/") and n.endswith(".mp4"))
        if not original or not fake:
            raise ValueError("Expected FF++ data with FaceForensics++_C23/original/ and Deepfakes/ folders")
        pairs = [Path(n).stem.split("_") for n in fake]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Unrecognised Deepfakes filename")
        splits = grouped_splits([Path(n).stem for n in original], pairs, seed)
        external = []
        for line in second.read_text("List_of_testing_videos.txt").splitlines():
            if line.strip():
                label, name = line.split(maxsplit=1)
                if label not in {"0", "1"} or name not in second.names:
                    raise ValueError("Invalid/missing Celeb-DF test-list video")
                external.append((1 - int(label), name))
        required = 0
        if isinstance(first, ZipSource):
            required += sum(first.size(n) for n in original + fake)
        if isinstance(second, ZipSource):
            required += sum(second.size(n) for _, n in external)
        free_space_at = output.parent if output.parent.exists() else Path.cwd()
        if required and shutil.disk_usage(free_space_at).free < required + 1024**3:
            raise ValueError("Insufficient disk space for selected videos")
        output.mkdir(parents=True)
        rows = []
        for index, name in enumerate(original + fake):
            path = first.video(name, output / "ffpp")
            sources = path.stem.split("_")
            rows.append(dict(path=str(path), dataset="FaceForensics++", video_id=f"{path.parent.name}/{path.stem}",
                             source_ids="|".join(sources), identity_ids="", label=int(path.parent.name == "Deepfakes"),
                             split=splits[sources[0]]))
            if (index + 1) % 100 == 0:
                print(f"FF++ indexed {index + 1}/{len(original) + len(fake)}", flush=True)
        for index, (label, name) in enumerate(external):
            path = second.video(name, output / "celeb")
            identities = [part for part in path.stem.split("_") if part.startswith("id") and part[2:].isdigit()]
            rows.append(dict(path=str(path), dataset="Celeb-DF-v2", video_id=name, source_ids=path.stem,
                             identity_ids="|".join(identities), label=label, split="external"))
            if (index + 1) % 100 == 0:
                print(f"Celeb-DF indexed {index + 1}/{len(external)}", flush=True)
        write_manifest(rows, output / "manifest.csv")
        inventory = {"seed": seed, "protocol": "Project 70/15/15 connected-source-group split, NOT official FF++ benchmark splits",
                     "external_protocol": "Celeb-DF supplied 518-video list (or actual list size below)",
                     "archives": [first.describe(), second.describe()],
                     "selected_videos": len(rows), "extracted_bytes": required,
                     "splits": {split: {"real": sum(r["split"] == split and r["label"] == 0 for r in rows),
                                         "manipulated": sum(r["split"] == split and r["label"] == 1 for r in rows)}
                                for split in ("train", "val", "test", "external")},
                     "datasets": dict(Counter(r["dataset"] for r in rows)),
                     "verification": "ZIP members extracted with CRC checks; folder inputs used in place; video decoding not yet tested",
                     "identity_limit": "FF++ source-disjoint only; identity metadata unavailable"}
        (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(json.dumps(inventory, indent=2))
    finally:
        first.close()
        if second is not None:
            second.close()
