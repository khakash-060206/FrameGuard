"""Lazy command imports keep manifest checks available without ML packages."""

import argparse
import json
import sys
from collections import Counter


def parser():
    root = argparse.ArgumentParser(description="FrameGuard prototype: data, training, evaluation, and inference")
    commands = root.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser("dashboard", help="Serve the local results dashboard")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--export", action="store_true", help="Refresh an offline HTML snapshot without a server")
    archives = commands.add_parser("import-archives")
    archives.add_argument("ffpp")
    archives.add_argument("celeb")
    archives.add_argument("output")
    archives.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("check-manifest")
    check.add_argument("manifest")
    merge = commands.add_parser("merge-manifests")
    merge.add_argument("output")
    merge.add_argument("manifests", nargs="+")
    ffpp = commands.add_parser("index-ffpp")
    ffpp.add_argument("root")
    ffpp.add_argument("splits")
    ffpp.add_argument("output")
    ffpp.add_argument("--compression", choices=["c23", "c40", "raw"], default="c23")
    ffpp.add_argument("--methods", nargs="+", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"], default=["Deepfakes"])
    celeb = commands.add_parser("index-celeb")
    celeb.add_argument("root")
    celeb.add_argument("output")
    prep = commands.add_parser("prepare")
    prep.add_argument("manifest")
    prep.add_argument("output")
    prep.add_argument("--detector", required=True)
    prep.add_argument("--max-samples", type=int, default=256)
    cache = commands.add_parser("cache-features")
    cache.add_argument("folder")
    cache.add_argument("checkpoint")
    cache.add_argument("output")
    training = commands.add_parser("train")
    training.add_argument("stage", choices=["spatial", "temporal"])
    training.add_argument("folder")
    training.add_argument("output")
    training.add_argument("--checkpoint")
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("folder")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument("output")
    evaluation.add_argument("--split", choices=["test", "external"], default="test")
    evaluation.add_argument("--batch-size", type=int, default=8)
    inference = commands.add_parser("predict")
    inference.add_argument("source", help="Image/MP4 path, or camera index when --webcam is used")
    inference.add_argument("checkpoint")
    inference.add_argument("output", help="JSONL results; existing files are never overwritten")
    inference.add_argument("--detector", required=True)
    inference.add_argument("--heatmaps")
    inference.add_argument("--webcam", action="store_true")
    inference.add_argument("--duration", type=float, default=30)
    for command in (cache, training, evaluation, inference):
        command.add_argument("--device", default="cpu", help="cpu or cuda; CUDA requires a matching PyTorch build")
    return root


def main():
    arguments = vars(parser().parse_args())
    command = arguments.pop("command")
    try:
        if command == "dashboard":
            from .dashboard import serve
            serve(**arguments)
        elif command == "import-archives":
            from .archives import import_archives
            import_archives(**arguments)
        elif command == "check-manifest":
            from .manifest import read_manifest
            rows = read_manifest(arguments["manifest"])
            print(json.dumps({"videos": len(rows), "splits": dict(Counter(row["split"] for row in rows)),
                              "identity_metadata_missing": sum(not row["identity_ids"] for row in rows)}, indent=2))
        elif command == "merge-manifests":
            from .indexing import merge_manifests
            merge_manifests(**arguments)
        elif command == "index-ffpp":
            from .indexing import index_ffpp
            index_ffpp(**arguments)
        elif command == "index-celeb":
            from .indexing import index_celeb
            index_celeb(**arguments)
        elif command == "prepare":
            from .prepare import prepare
            arguments["detector_path"] = arguments.pop("detector")
            prepare(**arguments)
        elif command == "cache-features":
            from .data import cache_features
            cache_features(**arguments)
        elif command == "train":
            from .train import train
            train(**arguments)
        elif command == "evaluate":
            from .evaluate import evaluate
            evaluate(**arguments)
        else:
            from .infer import predict
            arguments["detector_path"] = arguments.pop("detector")
            predict(**arguments)
    except (ValueError, OSError, ModuleNotFoundError) as error:
        print(f"FrameGuard: {error}", file=sys.stderr)
        if isinstance(error, ModuleNotFoundError):
            print("Install dependencies with: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
