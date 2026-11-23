"""Run the full FrameGuard experiment from raw datasets to evaluation reports.

Usage (from the project folder):

    python scripts/run_experiment.py --device auto
    python scripts/run_experiment.py --ffpp /kaggle/input/ff-c23 --celeb /kaggle/input/celeb-df-v2 \
        --data-dir /tmp/frameguard --frames-per-video 32 --batch-size 32 --workers 4

Stages: import datasets -> verify split -> prepare face crops -> train spatial CNN ->
cache features -> train LSTM -> evaluate internal test -> evaluate Celeb-DF.

Mixed protocol (pass --dfdc): FF++ C23 (4 trained methods; FaceShifter held out as an unseen method)
+ Celeb-DF v2 + DFDC pooled into one 70/15/15 split by connected source/identity groups. It writes
runs/mixed-test.json with overall, per-dataset and per-method results, and has no separate external set:

    python scripts/run_experiment.py --ffpp /kaggle/input/ff-c23 --celeb /kaggle/input/celeb-df-v2         --dfdc /kaggle/input/dfdc-10 --data-dir /tmp/fg --max-samples 64 --prep-workers 4 --augment on

Datasets can be ZIP files (extracted into --data-dir) or already-extracted folders
(used in place). Live state is written to runs/experiment-status.json. The runner stops
at the first failed stage. With --resume, stages recorded as completed in that file are
skipped, so an interrupted run continues from the stage that did not finish (a partially
finished stage's output must be removed first).
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

REFERENCE_SPLIT = Path("splits/reference-split.csv")
DETECTOR = "models/blaze_face_short_range.tflite"
STATUS = ROOT / "runs/experiment-status.json"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ffpp", "--ffpp-zip", dest="ffpp", default="data/archive.zip",
                        help="FaceForensics++ C23 as a ZIP or folder (default: data/archive.zip)")
    parser.add_argument("--celeb", "--celeb-zip", dest="celeb", default="data/archive (1).zip",
                        help="Celeb-DF v2 as a ZIP or folder (default: 'data/archive (1).zip')")
    parser.add_argument("--dfdc", default=None,
                        help="DFDC training parts folder (contains metadata.json files). Enables the mixed protocol")
    parser.add_argument("--celeb-fakes", type=int, default=2500, help="Mixed protocol: Celeb-DF fakes sampled (default: 2500)")
    parser.add_argument("--dfdc-reals", type=int, default=3000,
                        help="Mixed protocol: DFDC real videos sampled, each paired with one fake (default: 3000)")
    parser.add_argument("--holdout", default="FaceShifter",
                        help="Mixed protocol: FF++ method kept out of training and tested as unseen ('' = none)")
    parser.add_argument("--data-dir", default="data",
                        help="Where imported videos, face crops and features are written (default: data)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for each training stage (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Default: 8")
    parser.add_argument("--max-samples", type=int, default=256,
                        help="Max sampled frames per video during face preparation (default: 256, ~32 s at 8 fps)")
    parser.add_argument("--frames-per-video", type=int, default=None,
                        help="Spatial training: crops drawn per training video per epoch. Default: all crops (~140/video)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--prep-workers", type=int, default=1, help="Parallel face-preparation processes (default: 1)")
    parser.add_argument("--augment", default="auto", choices=["auto", "on", "off"],
                        help="Spatial training augmentation; auto = on for the mixed protocol only (default: auto)")
    parser.add_argument("--amp", default="auto", choices=["auto", "on", "off"],
                        help="float16 mixed precision; auto = on for CUDA (default: auto)")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages already recorded as completed in runs/experiment-status.json")
    parser.add_argument("--allow-split-mismatch", action="store_true",
                        help="Continue even if the imported split differs from splits/reference-split.csv")
    return parser.parse_args()


def verify_split(manifest, allow_mismatch):
    """Compare the freshly built split with the reference split from the original run."""
    if not REFERENCE_SPLIT.exists():
        print("No reference split found; skipping split comparison.", flush=True)
        return

    def load(path):
        with open(path, newline="", encoding="utf-8") as handle:
            return {(row["dataset"], row["video_id"]): (row["label"], row["split"]) for row in csv.DictReader(handle)}

    reference, current = load(REFERENCE_SPLIT), load(manifest)
    missing = reference.keys() - current.keys()
    extra = current.keys() - reference.keys()
    changed = [key for key in reference.keys() & current.keys() if reference[key] != current[key]]
    if not (missing or extra or changed):
        print(f"Split matches the reference exactly ({len(reference)} videos).", flush=True)
        return
    message = (f"Split differs from {REFERENCE_SPLIT}: {len(missing)} missing, {len(extra)} extra, "
               f"{len(changed)} with a different label/split. The datasets are probably not the same "
               f"versions as the original run, so results would not be comparable.")
    if allow_mismatch:
        print("WARNING: " + message, flush=True)
    else:
        raise ValueError(message + " Use the same datasets, or pass --allow-split-mismatch to continue anyway.")


def main():
    options = arguments()
    import torch
    import cv2
    from frameguard.assets import ensure_assets
    from frameguard.archives import import_archives, import_mixed
    from frameguard.prepare import prepare
    from frameguard.train import train
    from frameguard.data import cache_features
    from frameguard.evaluate import evaluate

    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but PyTorch cannot see a CUDA GPU. Use --device cpu.")
    amp = {"auto": None, "on": True, "off": False}[options.amp]
    mixed = options.dfdc is not None
    augment = mixed if options.augment == "auto" else options.augment == "on"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Using CPU. Training will take many hours.", flush=True)
    torch.set_num_threads(options.threads)
    cv2.setNumThreads(2)
    ensure_assets(ROOT)

    data = Path(options.data_dir)
    imported, crops, features = data / "imported", data / "crops", data / "features"
    manifest = imported / "manifest.csv"
    epochs, batch = options.epochs, options.batch_size
    common = dict(batch_size=batch, device=device, workers=options.workers, amp=amp)

    def import_data():
        if manifest.exists():
            print(f"{manifest} already exists; reusing it.", flush=True)
            return
        for name in (options.ffpp, options.celeb) + ((options.dfdc,) if mixed else ()):
            if not Path(name).exists():
                raise ValueError(f"Dataset not found: {name}. Pass its location with --ffpp / --celeb / --dfdc.")
        if mixed:
            import_mixed(options.ffpp, options.celeb, options.dfdc, str(imported), seed=42, holdout=options.holdout or None,
                         celeb_fakes=options.celeb_fakes, dfdc_reals=options.dfdc_reals)
        else:
            import_archives(options.ffpp, options.celeb, str(imported), seed=42)

    def check_split():
        if mixed:
            print(f"Mixed protocol: new pooled split written to {manifest} (no reference split to compare).", flush=True)
        else:
            verify_split(manifest, options.allow_split_mismatch)

    stages = [
        ("importing_datasets", import_data),
        ("verifying_split", check_split),
        ("preparing_faces", lambda: prepare(str(manifest), str(crops), DETECTOR, max_samples=options.max_samples,
                                            workers=options.prep_workers)),
        ("training_spatial", lambda: train(str(crops), "models/spatial.pt", "spatial", epochs=epochs,
                                           frames_per_video=options.frames_per_video, augment=augment, **common)),
        ("caching_features", lambda: cache_features(str(crops), "models/spatial.pt", str(features), device=device)),
        ("training_temporal", lambda: train(str(features), "models/temporal.pt", "temporal", "models/spatial.pt",
                                            epochs=epochs, **common)),
        ("evaluating_internal", lambda: evaluate(str(features), "models/temporal.pt",
                                                 "runs/mixed-test.json" if mixed else "runs/ffpp-test.json",
                                                 device=device, batch_size=batch)),
    ]
    if not mixed:
        stages.append(("evaluating_external", lambda: evaluate(str(features), "models/temporal.pt", "runs/celeb-external.json",
                                                               split="external", device=device, batch_size=batch)))

    done = []
    if options.resume and STATUS.exists():
        done = json.loads(STATUS.read_text(encoding="utf-8")).get("completed_stages", [])
        # Import and split checks are cheap and validate the current inputs, so they always rerun.
        done = [name for name in done if name not in {"importing_datasets", "verifying_split"}]
        if done:
            print(f"Resuming: skipping completed stages {done}", flush=True)

    STATUS.parent.mkdir(exist_ok=True)
    settings = {"protocol": "mixed" if mixed else "ffpp-deepfakes", "augment": augment,
                "prep_workers": options.prep_workers, "device": device, "epochs_per_stage": epochs, "batch_size": batch, "max_samples": options.max_samples,
                "frames_per_video": options.frames_per_video, "workers": options.workers, "amp": options.amp,
                "data_dir": str(data)}
    status = {"pid": os.getpid(), **settings, "completed_stages": list(done)}

    def record(phase, error=None):
        status.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), error=error)
        temporary = STATUS.with_suffix(".json.partial")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(STATUS)
        print(json.dumps(status), flush=True)

    try:
        for name, action in stages:
            if name in done:
                continue
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
