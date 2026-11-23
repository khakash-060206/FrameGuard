"""Run the full FrameGuard experiment from dataset ZIPs to evaluation reports.

Usage (from the project folder, after setup.ps1):

    .\\.venv\\Scripts\\python.exe scripts\\run_experiment.py --device auto

Stages: import ZIPs -> verify split -> prepare face crops -> train spatial CNN ->
cache features -> train LSTM -> evaluate internal test -> evaluate Celeb-DF.
Live state is written to runs/experiment-status.json. The runner stops at the first
failed stage. Every stage refuses to overwrite existing outputs, so a rerun after a
failure needs the failed stage's output removed or a fresh project folder.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

IMPORTED = Path("data/imported")
MANIFEST = IMPORTED / "manifest.csv"
REFERENCE_SPLIT = Path("splits/reference-split.csv")
DETECTOR = "models/blaze_face_short_range.tflite"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ffpp-zip", default="data/archive.zip", help="FaceForensics++ C23 ZIP (default: data/archive.zip)")
    parser.add_argument("--celeb-zip", default="data/archive (1).zip", help="Celeb-DF v2 ZIP (default: 'data/archive (1).zip')")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for each training stage")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--allow-split-mismatch", action="store_true",
                        help="Continue even if the imported split differs from splits/reference-split.csv")
    return parser.parse_args()


def verify_split(allow_mismatch):
    """Compare the freshly built split with the reference split from the original run."""
    if not REFERENCE_SPLIT.exists():
        print("No reference split found; skipping split comparison.", flush=True)
        return
    def load(path):
        with open(path, newline="", encoding="utf-8") as handle:
            return {(row["dataset"], row["video_id"]): (row["label"], row["split"]) for row in csv.DictReader(handle)}
    reference = load(REFERENCE_SPLIT)
    current = load(MANIFEST)
    missing = reference.keys() - current.keys()
    extra = current.keys() - reference.keys()
    changed = [key for key in reference.keys() & current.keys() if reference[key] != current[key]]
    if not (missing or extra or changed):
        print(f"Split matches the reference exactly ({len(reference)} videos).", flush=True)
        return
    message = (f"Split differs from {REFERENCE_SPLIT}: {len(missing)} missing, {len(extra)} extra, "
               f"{len(changed)} with a different label/split. The dataset ZIPs are probably not the same "
               f"versions as the original run, so results would not be comparable.")
    if allow_mismatch:
        print("WARNING: " + message, flush=True)
    else:
        raise ValueError(message + " Use the same ZIPs, or pass --allow-split-mismatch to continue anyway.")


def main():
    options = arguments()
    import torch
    import cv2
    from frameguard.archives import import_archives
    from frameguard.prepare import prepare
    from frameguard.train import train
    from frameguard.data import cache_features
    from frameguard.evaluate import evaluate

    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but PyTorch cannot see a CUDA GPU. Rerun setup.ps1 or use --device cpu.")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Using CPU. Training will take many hours.", flush=True)
    torch.set_num_threads(options.threads)
    cv2.setNumThreads(2)

    epochs, batch = options.epochs, options.batch_size
    stages = []
    if not MANIFEST.exists():
        for name in (options.ffpp_zip, options.celeb_zip):
            if not Path(name).is_file():
                raise SystemExit(f"Dataset ZIP not found: {name}. Pass its location with --ffpp-zip / --celeb-zip.")
        stages.append(("importing_archives", lambda: import_archives(options.ffpp_zip, options.celeb_zip, str(IMPORTED), seed=42)))
    else:
        print(f"{MANIFEST} already exists; skipping import.", flush=True)
    stages += [
        ("verifying_split", lambda: verify_split(options.allow_split_mismatch)),
        ("preparing_faces", lambda: prepare(str(MANIFEST), "data/crops", DETECTOR)),
        ("training_spatial", lambda: train("data/crops", "models/spatial.pt", "spatial",
                                           epochs=epochs, batch_size=batch, device=device)),
        ("caching_features", lambda: cache_features("data/crops", "models/spatial.pt", "data/features", device=device)),
        ("training_temporal", lambda: train("data/features", "models/temporal.pt", "temporal", "models/spatial.pt",
                                            epochs=epochs, batch_size=batch, device=device)),
        ("evaluating_internal", lambda: evaluate("data/features", "models/temporal.pt", "runs/ffpp-test.json",
                                                 device=device, batch_size=batch)),
        ("evaluating_external", lambda: evaluate("data/features", "models/temporal.pt", "runs/celeb-external.json",
                                                 split="external", device=device, batch_size=batch)),
    ]

    status_path = ROOT / "runs/experiment-status.json"
    status_path.parent.mkdir(exist_ok=True)
    status = {"pid": os.getpid(), "device": device, "epochs_per_stage": epochs, "batch_size": batch,
              "completed_stages": []}

    def record(phase, error=None):
        status.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), error=error)
        temporary = status_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)
        print(json.dumps(status), flush=True)

    try:
        for name, action in stages:
            record(name)
            action()
            status["completed_stages"].append(name)
        record("completed")
    except BaseException as error:
        failed_stage = status.get("phase")
        record("failed", f"{failed_stage}: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
