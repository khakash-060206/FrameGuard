"""Run the approved experiment sequentially and persist its actual phase."""
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


def main():
    import torch
    import cv2
    from frameguard.prepare import prepare
    from frameguard.train import train
    from frameguard.data import cache_features
    from frameguard.evaluate import evaluate

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    status_path = ROOT / "runs/experiment-status.json"
    status_path.parent.mkdir(exist_ok=True)
    stages = [
        ("preparing_faces", lambda: prepare("data/imported/manifest.csv", "data/crops", "models/blaze_face_short_range.tflite")),
        ("training_spatial", lambda: train("data/crops", "models/spatial.pt", "spatial", epochs=5)),
        ("caching_features", lambda: cache_features("data/crops", "models/spatial.pt", "data/features")),
        ("training_temporal", lambda: train("data/features", "models/temporal.pt", "temporal", "models/spatial.pt", epochs=5)),
        ("evaluating_internal", lambda: evaluate("data/features", "models/temporal.pt", "runs/ffpp-test.json")),
        ("evaluating_external", lambda: evaluate("data/features", "models/temporal.pt", "runs/celeb-external.json", split="external")),
    ]
    status = {"pid": os.getpid(), "device": "cpu", "epochs_per_stage": 5, "completed_stages": []}

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
