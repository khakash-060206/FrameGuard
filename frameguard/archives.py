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


def weighted_group_splits(groups, seed=42, fractions=(("train", 0.7), ("val", 0.15), ("test", 0.15))):
    """Assign whole groups to splits so each split approaches its target share of every class.

    groups: {group: count} or {group: (real_count, fake_count)}. Largest groups are placed first (seeded
    order among equal sizes), each into the split whose per-class shortfall it fills best.
    """
    if len(groups) < len(fractions):
        raise ValueError("Need at least one independent group per split")
    counts = {key: value if isinstance(value, tuple) else (value,) for key, value in groups.items()}
    totals = [sum(column) for column in zip(*counts.values())]
    keys = sorted(counts)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: -sum(counts[key]))
    assigned, result = {name: [0] * len(totals) for name, _ in fractions}, {}

    def shortfall(item, key):
        name, share = item
        return sum(c * (share * t - a) / t for c, t, a in zip(counts[key], totals, assigned[name]) if t)

    for key in keys:
        name = max(fractions, key=lambda item: shortfall(item, key))[0]
        result[key] = name
        assigned[name] = [a + c for a, c in zip(assigned[name], counts[key])]
    return result


def connected_groups(members):
    """members: {video: [ids]}; returns {video: group root} where videos sharing any id are connected."""
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for ids in members.values():
        for first, second in zip(ids, ids[1:]):
            parent[find(first)] = find(second)
        for value in ids:
            find(value)
    return {video: find(ids[0]) for video, ids in members.items()}


FFPP_TRAINED_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def import_mixed(ffpp, celeb, dfdc, output, seed=42, holdout="FaceShifter", celeb_fakes=2500, dfdc_reals=3000):
    """One pooled FF++ + Celeb-DF v2 + DFDC manifest with a 70/15/15 split per dataset by connected groups.

    Groups: FF++ source videos (fakes join their two sources); Celeb-DF identities (fakes join both
    identities; YouTube-real videos stand alone); DFDC original videos (fakes join their original).
    The `holdout` FF++ method never enters train/val; only its fakes from test groups are kept, so it
    measures an unseen manipulation method. Folders are used in place; ZIPs are extracted into `output`.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing data will not be overwritten")
    rng = random.Random(seed)
    first, second = open_source(ffpp, "FaceForensics++_C23"), None
    try:
        second = open_source(celeb, "List_of_testing_videos.txt")
        selected = []  # (source, name, dataset, video_id, ids, identity_ids, label, method)

        prefix = "FaceForensics++_C23/"
        listing = lambda folder: sorted(n for n in first.names if n.startswith(prefix + folder + "/") and n.endswith(".mp4"))
        originals = listing("original")
        if not originals:
            raise ValueError("Expected FF++ folders FaceForensics++_C23/original/ and manipulation methods")
        for name in originals:
            selected.append((first, name, "FaceForensics++", f"original/{Path(name).stem}", [Path(name).stem], "", 0, None))
        for method in FFPP_TRAINED_METHODS + ((holdout,) if holdout else ()):
            names = listing(method)
            if not names:
                raise ValueError(f"FF++ method folder missing or empty: {method}")
            for name in names:
                pair = Path(name).stem.split("_")
                if len(pair) != 2:
                    raise ValueError(f"Unrecognised FF++ filename: {name}")
                selected.append((first, name, "FaceForensics++", f"{method}/{Path(name).stem}", pair, "", 1, method))

        celeb_real = sorted(n for n in second.names if n.split("/")[0] in {"Celeb-real", "YouTube-real"} and n.endswith(".mp4"))
        synthesis = sorted(n for n in second.names if n.startswith("Celeb-synthesis/") and n.endswith(".mp4"))
        if not celeb_real or not synthesis:
            raise ValueError("Expected Celeb-DF folders Celeb-real/, YouTube-real/ and Celeb-synthesis/")
        synthesis = sorted(rng.sample(synthesis, min(celeb_fakes, len(synthesis))))
        for label, names in ((0, celeb_real), (1, synthesis)):
            for name in names:
                stem = Path(name).stem
                identities = [part for part in stem.split("_") if part.startswith("id") and part[2:].isdigit()]
                selected.append((second, name, "Celeb-DF-v2", name, identities or [f"youtube:{stem}"],
                                 "|".join(identities), label, None))

        selected += dfdc_selection(dfdc, rng, dfdc_reals)

        # Split each dataset by connected groups, weighted by video count; the holdout method does not vote.
        splits = {}
        for dataset in sorted({row[2] for row in selected}):
            rows = [row for row in selected if row[2] == dataset and row[7] != holdout]
            roots = connected_groups({row[3]: [f"{dataset}:{value}" for value in row[4]] for row in rows})
            labels = {row[3]: row[6] for row in rows}
            sizes = {root: (0, 0) for root in roots.values()}
            for key, root in roots.items():
                real, fake = sizes[root]
                sizes[root] = (real + (labels[key] == 0), fake + (labels[key] == 1))
            group_split = weighted_group_splits(sizes, seed)
            splits.update({key: group_split[root] for key, root in roots.items()})
            for row in selected:
                if row[2] == dataset and row[7] == holdout:
                    source_splits = {splits.get(f"original/{value}") for value in row[4]}
                    splits[row[3]] = "test" if source_splits == {"test"} else None

        selected = [row for row in selected if splits[row[3]]]
        required = sum(source.size(name) for source, name, *_ in selected if isinstance(source, ZipSource))
        free_space_at = output.parent if output.parent.exists() else Path.cwd()
        if required and shutil.disk_usage(free_space_at).free < required + 1024**3:
            raise ValueError("Insufficient disk space for selected videos")
        output.mkdir(parents=True)
        manifest = []
        for index, (source, name, dataset, video_id, ids, identities, label, _) in enumerate(selected):
            path = source.video(name, output / dataset) if source is not None else Path(name)
            manifest.append(dict(path=str(path), dataset=dataset, video_id=video_id,
                                 source_ids="|".join(ids) if dataset != "Celeb-DF-v2" else Path(name).stem,
                                 identity_ids=identities, label=label, split=splits[video_id]))
            if (index + 1) % 1000 == 0:
                print(f"Indexed {index + 1}/{len(selected)}", flush=True)
        write_manifest(manifest, output / "manifest.csv")
        inventory = {"seed": seed,
                     "protocol": ("Pooled FF++ C23 + Celeb-DF v2 + DFDC; 70/15/15 split per dataset by connected source/identity "
                                  "groups, weighted by video count. Not an official benchmark split."),
                     "unseen_method": f"FF++ {holdout}: excluded from training; test-group fakes only" if holdout else None,
                     "archives": [first.describe(), second.describe(), {"name": str(dfdc), "type": "folder"}],
                     "selected_videos": len(manifest), "extracted_bytes": required,
                     "splits": {split: {"real": sum(r["split"] == split and r["label"] == 0 for r in manifest),
                                         "manipulated": sum(r["split"] == split and r["label"] == 1 for r in manifest)}
                                for split in ("train", "val", "test")},
                     "datasets": dict(Counter(r["dataset"] for r in manifest)),
                     "by_dataset": {f"{d}/{s}": {"real": sum(r["dataset"] == d and r["split"] == s and r["label"] == 0 for r in manifest),
                                                 "manipulated": sum(r["dataset"] == d and r["split"] == s and r["label"] == 1 for r in manifest)}
                                    for d in sorted({r["dataset"] for r in manifest}) for s in ("train", "val", "test")},
                     "verification": "ZIP members extracted with CRC checks; folder inputs used in place; video decoding not yet tested",
                     "identity_limit": "FF++ source-disjoint; Celeb-DF identity-disjoint; DFDC original-disjoint (actor IDs unavailable)"}
        (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(json.dumps(inventory, indent=2))
        return manifest
    finally:
        first.close()
        if second is not None:
            second.close()


def dfdc_selection(root, rng, reals=3000):
    """DFDC training parts (folders holding metadata.json): up to `reals` real videos, each with one of its fakes."""
    entries = {}
    for meta in sorted(Path(root).rglob("metadata.json")):
        for name, info in json.loads(meta.read_text(encoding="utf-8")).items():
            if (meta.parent / name).is_file():
                entries[name] = (meta.parent, info)
    if not entries:
        raise ValueError(f"No DFDC metadata.json with videos found under {root}")
    fakes = {}
    for name, (_, info) in sorted(entries.items()):
        if info.get("label") == "FAKE" and info.get("original") in entries:
            fakes.setdefault(info["original"], []).append(name)
    originals = sorted(name for name, (_, info) in entries.items() if info.get("label") == "REAL")
    originals = sorted(rng.sample(originals, min(reals, len(originals))))
    rows = []
    for name in originals:
        group = Path(name).stem
        chosen = [(name, 0)] + ([(rng.choice(fakes[name]), 1)] if name in fakes else [])
        for video, label in chosen:
            folder = entries[video][0]
            rows.append((None, str((folder / video).resolve()), "DFDC", f"{folder.name}/{video}", [group], "", label, None))
    return rows


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
