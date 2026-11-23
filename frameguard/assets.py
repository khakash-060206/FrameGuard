"""Download the pretrained assets FrameGuard needs and verify their SHA-256 hashes.

Cross-platform counterpart of the asset step in setup.ps1 (used on Kaggle/Linux).
"""

import hashlib
from pathlib import Path
import urllib.request

ASSETS = (
    ("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
     "models/blaze_face_short_range.tflite",
     "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"),
    ("https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
     ".cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth",
     "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934"),
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_assets(root="."):
    """Download any missing asset into `root`; raise if a hash does not match."""
    root = Path(root)
    for url, relative, expected in ASSETS:
        target = root / relative
        if target.is_file() and sha256(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, partial)
        actual = sha256(partial)
        if actual != expected:
            partial.unlink()
            raise ValueError(f"Hash mismatch for {relative}: expected {expected}, got {actual}")
        partial.replace(target)
        print(f"OK (hash verified): {relative}", flush=True)
