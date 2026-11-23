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


def import_archives(ffpp, celeb, output, seed=42):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing data will not be overwritten")
    with zipfile.ZipFile(ffpp) as first, zipfile.ZipFile(celeb) as second:
        original = sorted(n for n in first.namelist() if n.startswith("FaceForensics++_C23/original/") and n.endswith(".mp4"))
        fake = sorted(n for n in first.namelist() if n.startswith("FaceForensics++_C23/Deepfakes/") and n.endswith(".mp4"))
        if not original or not fake:
            raise ValueError("Expected FF++ archive with original/ and Deepfakes/ folders")
        pairs = [Path(n).stem.split("_") for n in fake]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Unrecognised Deepfakes filename")
        splits = grouped_splits([Path(n).stem for n in original], pairs, seed)
        external = []
        for line in second.read("List_of_testing_videos.txt").decode("utf-8-sig").splitlines():
            if line.strip():
                label, name = line.split(maxsplit=1)
                if label not in {"0", "1"} or name not in second.namelist():
                    raise ValueError("Invalid/missing Celeb-DF test-list video")
                external.append((1 - int(label), name))
        required = sum(first.getinfo(n).file_size for n in original + fake) + sum(second.getinfo(n).file_size for _, n in external)
        if shutil.disk_usage(output.parent).free < required + 1024**3:
            raise ValueError("Insufficient disk space for selected videos")
        output.mkdir(parents=True)
        rows = []
        for index, name in enumerate(original + fake):
            path = extract_member(first, name, output / "ffpp")
            sources = path.stem.split("_")
            rows.append(dict(path=str(path), dataset="FaceForensics++", video_id=f"{path.parent.name}/{path.stem}",
                             source_ids="|".join(sources), identity_ids="", label=int(path.parent.name == "Deepfakes"),
                             split=splits[sources[0]]))
            if (index + 1) % 100 == 0:
                print(f"FF++ extracted {index + 1}/{len(original) + len(fake)}", flush=True)
        for index, (label, name) in enumerate(external):
            path = extract_member(second, name, output / "celeb")
            identities = [part for part in path.stem.split("_") if part.startswith("id") and part[2:].isdigit()]
            rows.append(dict(path=str(path), dataset="Celeb-DF-v2", video_id=name, source_ids=path.stem,
                             identity_ids="|".join(identities), label=label, split="external"))
            if (index + 1) % 100 == 0:
                print(f"Celeb-DF extracted {index + 1}/{len(external)}", flush=True)
        write_manifest(rows, output / "manifest.csv")
        inventory = {"seed": seed, "protocol": "Project 70/15/15 connected-source-group split, NOT official FF++ benchmark splits",
                     "external_protocol": "Celeb-DF supplied 518-video list (or actual list size below)",
                     "archives": [{"name": Path(p).name, "bytes": Path(p).stat().st_size} for p in (ffpp, celeb)],
                     "selected_videos": len(rows), "extracted_bytes": required,
                     "splits": {split: {"real": sum(r["split"] == split and r["label"] == 0 for r in rows),
                                         "manipulated": sum(r["split"] == split and r["label"] == 1 for r in rows)}
                                for split in ("train", "val", "test", "external")},
                     "datasets": dict(Counter(r["dataset"] for r in rows)),
                     "verification": "Selected ZIP members read fully with CRC checks; video decoding not yet tested",
                     "identity_limit": "FF++ source-disjoint only; identity metadata unavailable"}
        (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(json.dumps(inventory, indent=2))
