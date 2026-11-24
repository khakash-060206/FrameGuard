"""Local, read-only experiment dashboard with an optional inference endpoint."""

import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit

from . import UNCERTAIN_MARGIN

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "dashboard"
INFERENCE_LOCK = threading.Lock()
MAX_UPLOAD = 200 * 1024 * 1024
PROTOCOL_FRAMES = 256  # frame budget for checkpoints that predate the max_samples metadata field


def snapshot(root=ROOT):
    root = Path(root)
    warnings = []

    def read(path):
        try:
            if path.stat().st_size > 30 * 1024 * 1024:
                raise ValueError("Report exceeds 30 MB")
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            warnings.append(f"{path.name}: {error}")
            return None

    histories, reports = [], []
    for folder in (root / "models", root / "runs"):
        for path in sorted(folder.glob("*.history.json")):
            data = read(path)
            if isinstance(data, list):
                histories.append({"name": path.name, "epochs": data})
    for path in sorted((root / "runs").glob("*.json")):
        if path.name.endswith(".history.json"):
            continue
        data = read(path)
        if isinstance(data, dict) and "metrics" in data and "temporal_predictions" in data:
            reports.append({"name": path.name, "report": data})
    # A run's own inventory (copied in with Kaggle results) describes the current models; a local import is the fallback.
    inventory_path = next((p for p in (root / "runs/inventory.json", root / "data/imported/inventory.json") if p.is_file()), None)
    inventory = read(inventory_path) if inventory_path else None
    if isinstance(inventory, dict) and "prepared_videos" in inventory:
        prepared_count = inventory["prepared_videos"]
    else:
        prepared_path = root / "data/crops/index.json"
        prepared = read(prepared_path) if prepared_path.is_file() else None
        prepared_count = len(prepared.get("videos", [])) if isinstance(prepared, dict) else 0
    experiment_path = root / "runs/experiment-status.json"
    experiment = read(experiment_path) if experiment_path.is_file() else None
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "numpy", "cv2", "mediapipe", "sklearn")}
    checkpoints = [p.name for p in sorted((root / "models").glob("*.pt")) if p.is_file()]
    detector = (root / "models/blaze_face_short_range.tflite").is_file()
    return {"histories": histories, "reports": reports, "inventory": inventory, "warnings": warnings,
            "experiment": experiment,
            "prepared_videos": prepared_count,
            "dependencies": dependencies, "checkpoints": checkpoints, "detector": detector,
            "can_predict": bool(checkpoints and detector and all(dependencies.values())),
            "uncertain_margin": UNCERTAIN_MARGIN}


def export_snapshot(root=ROOT):
    path = Path(root) / "dashboard/data.js"
    payload = json.dumps(snapshot(root), ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    path.write_text("window.FRAMEGUARD_DATA = " + payload + ";\n", encoding="utf-8")
    return path


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in expected:
            return False
        origin = self.headers.get("Origin")
        return not origin or origin in {f"http://{host}" for host in expected}

    def send_bytes(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status, value):
        self.send_bytes(status, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if not self.allowed():
            return self.send_json(403, {"error": "Local dashboard access only"})
        path = urlsplit(self.path).path
        if path == "/api/state":
            return self.send_json(200, snapshot())
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/style.css": ("style.css", "text/css"), "/app.js": ("app.js", "text/javascript"),
                 "/charts.js": ("charts.js", "text/javascript"), "/data.js": ("data.js", "text/javascript")}
        if path not in files:
            return self.send_json(404, {"error": "Not found"})
        filename, mime = files[path]
        self.send_bytes(200, (ASSETS / filename).read_bytes(), mime + "; charset=utf-8")

    def do_POST(self):
        if not self.allowed():
            return self.send_json(403, {"error": "Local dashboard access only"})
        url = urlsplit(self.path)
        if url.path != "/api/predict":
            return self.send_json(404, {"error": "Not found"})
        state = snapshot()
        if not state["can_predict"]:
            return self.send_json(409, {"error": "Install dependencies and provide a trained checkpoint and face detector first."})
        query = parse_qs(url.query)
        checkpoint = query.get("checkpoint", [""])[0]
        suffix = Path(query.get("name", [""])[0]).suffix.lower()
        if checkpoint not in state["checkpoints"] or suffix not in {".jpg", ".jpeg", ".png", ".mp4"}:
            return self.send_json(400, {"error": "Choose a listed checkpoint and a JPG, PNG, or MP4 file."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_json(400, {"error": "Invalid upload size"})
        if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_UPLOAD:
            return self.send_json(413, {"error": "Upload limit is 200 MB; use the CLI for larger videos."})
        if not INFERENCE_LOCK.acquire(blocking=False):
            return self.send_json(409, {"error": "An analysis is already running. Try again after it finishes."})
        try:
            from .infer import predict, verdict
            from .model import load_checkpoint
            cache = ROOT / ".cache/uploads"
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache) as directory:
                media, result = Path(directory) / ("upload" + suffix), Path(directory) / "result.jsonl"
                self.connection.settimeout(120)
                with media.open("xb") as stream:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Incomplete upload")
                        stream.write(chunk)
                        remaining -= len(chunk)
                # Same frame budget the checkpoint was trained and evaluated with, so scores are comparable.
                frames = load_checkpoint(ROOT / "models" / checkpoint)[1].get("max_samples") or PROTOCOL_FRAMES
                predict(str(media), ROOT / "models" / checkpoint, ROOT / "models/blaze_face_short_range.tflite", result,
                        max_samples=frames)
                records = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
            self.send_json(200, {"records": records, "verdict": verdict(records), "max_frames": frames,
                                 "calibrated": False})
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
            self.send_json(400, {"error": str(error)})
        finally:
            INFERENCE_LOCK.release()


def serve(port=8765, export=False):
    path = export_snapshot()
    if export:
        print(f"Dashboard snapshot: {path.parent / 'index.html'}")
        return
    # On Windows, SO_REUSEADDR lets a second server bind the same port and silently split requests with a stale one.
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"FrameGuard dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
