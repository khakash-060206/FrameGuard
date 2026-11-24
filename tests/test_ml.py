"""Requires the real ML stack; never downloads weights or substitutes fake libraries."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "cv2", "numpy", "sklearn"))


@unittest.skipUnless(AVAILABLE, "ML dependencies unavailable; install requirements.txt")
class MLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_verdict_uses_temporal_windows_like_evaluation(self):
        from frameguard.infer import verdict
        spatial = {"mode": "spatial", "score": 0.9, "threshold": 0.4}
        window = lambda score: {"mode": "temporal", "score": score, "threshold": 0.2}
        result = verdict([{"faces": [spatial]}, {"faces": [window(0.1), window(0.4)]}], margin=0)
        self.assertEqual((result["label"], result["mode"], result["units"], result["max_faces"]), ("manipulated", "temporal", 2, 2))
        self.assertAlmostEqual(result["score"], 0.25)
        self.assertEqual(verdict([{"faces": [spatial]}], margin=0)["mode"], "spatial")
        # Default band: 0.25 vs threshold 0.2 is too close to call; 0.9 and 0.01 are clear.
        self.assertEqual(verdict([{"faces": [window(0.1), window(0.4)]}])["label"], "uncertain")
        self.assertEqual(verdict([{"faces": [window(0.9)]}])["label"], "manipulated")
        self.assertEqual(verdict([{"faces": [window(0.01)]}])["label"], "real")
        low, high = verdict([{"faces": [window(0.5)]}])["uncertain_range"]
        self.assertTrue(0 < low < 0.2 < high < 1)
        self.assertEqual(verdict([{"faces": []}])["label"], "unable_to_assess")

    def test_augment_keeps_shape_and_type(self):
        import random
        import numpy as np
        from frameguard.data import augment
        crop = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        for seed in range(20):
            out = augment(crop, random.Random(seed))
            self.assertEqual((out.shape, out.dtype), ((224, 224, 3), np.uint8))
            self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertTrue(np.array_equal(augment(crop, random.Random(3)), augment(crop, random.Random(3))))

    def test_domains_split_unseen_method_from_totals(self):
        from frameguard.evaluate import assign_domains
        trained = [{"dataset": "FaceForensics++", "video_id": v} for v in ("original/000", "Deepfakes/000_001")]
        predictions = [{"dataset": "FaceForensics++", "video_id": "original/010", "label": 0},
                       {"dataset": "FaceForensics++", "video_id": "Deepfakes/010_011", "label": 1},
                       {"dataset": "FaceForensics++", "video_id": "FaceShifter/010_011", "label": 1},
                       {"dataset": "DFDC", "video_id": "p/a.mp4", "label": 0}]
        domains = [p["domains"] for p in assign_domains(predictions, trained)]
        self.assertEqual(domains[0], ["FaceForensics++", "All datasets", "FF++ Deepfakes", "FF++ FaceShifter (unseen method)"])
        self.assertEqual(domains[1], ["FaceForensics++", "All datasets", "FF++ Deepfakes"])
        self.assertEqual(domains[2], ["FF++ FaceShifter (unseen method)"])
        self.assertEqual(domains[3], ["DFDC", "All datasets"])

    def test_rgb_preprocessing(self):
        import numpy as np
        from frameguard.media import normalise
        crops = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        crops[..., 0] = 255
        tensor = normalise(crops)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertAlmostEqual(tensor[0, 0, 0, 0].item(), (1 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(tensor[0, 2, 0, 0].item(), -0.406 / 0.225, places=5)

    def test_temporal_gradient_and_update(self):
        import torch
        from frameguard.model import FrameGuard
        model = FrameGuard()
        values = torch.randn(2, 16, 1280)
        before = model.sequence_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(list(model.lstm.parameters()) + list(model.sequence_head.parameters()), lr=0.01)
        logits = model.temporal(values)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sequence_head.weight))

    def test_cnn_shape_and_checkpoint_roundtrip(self):
        import torch
        from frameguard.model import FrameGuard, save_checkpoint, load_checkpoint
        model = FrameGuard().eval()
        value = torch.zeros(1, 3, 224, 224)
        with torch.inference_mode():
            expected = model(value)
            self.assertEqual(tuple(model.encode(value).shape), (1, 1280))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint, model, {"stage": "spatial", "threshold_spatial": 0.5})
            loaded, _ = load_checkpoint(checkpoint)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(value), expected)

    def test_invalid_sequence_shape_rejected(self):
        import torch
        from frameguard.model import FrameGuard
        with self.assertRaisesRegex(ValueError, "16, 1280"):
            FrameGuard().temporal(torch.zeros(1, 15, 1280))

    def test_metrics_and_threshold(self):
        from frameguard.metrics import classification_metrics, select_threshold, diagnostic_curves
        labels, scores = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        threshold = select_threshold(labels, scores)
        result = classification_metrics(labels, scores, threshold)
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["specificity"], 1)
        self.assertEqual(diagnostic_curves(labels, scores)["average_precision"], 1)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertIsNone(classification_metrics([0], [0.2], 0.5)["roc_auc"])

    def test_untrained_checkpoint_rejected(self):
        from frameguard.model import FrameGuard, save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "untrained.pt"
            save_checkpoint(checkpoint, FrameGuard(), {})
            with self.assertRaisesRegex(ValueError, "not a trained"):
                load_checkpoint(checkpoint)

    def test_temporal_training_and_evaluation_pipeline(self):
        import json
        from dataclasses import asdict
        import numpy as np
        from frameguard.model import FrameGuard, ModelConfig, save_checkpoint
        from frameguard.manifest import file_sha256
        from frameguard.train import train
        from frameguard.evaluate import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spatial = root / "spatial.pt"
            save_checkpoint(spatial, FrameGuard(), {"stage": "spatial", "threshold_spatial": 0.5})
            rows = []
            for split in ("train", "val", "test"):
                for label in (0, 1):
                    name = f"{split}_{label}"
                    rng = np.random.default_rng(label)
                    np.savez_compressed(root / f"{name}.npz", features=rng.normal(size=(16, 1280)).astype(np.float32),
                                        spatial_logits=np.full(16, label * 2 - 1, dtype=np.float32),
                                        timestamps=np.arange(16) / 8, tracks=np.zeros(16, dtype=np.int64))
                    rows.append(dict(dataset="synthetic-test-fixture", video_id=name, source_ids=name,
                                     identity_ids="", split=split, label=label, file=f"{name}.npz"))
            index = dict(format=1, config=asdict(ModelConfig()), videos=rows, rejected=[],
                         checkpoint_sha256=file_sha256(spatial), manifest_sha256="synthetic-test-only")
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            checkpoint = root / "temporal.pt"
            train(root, checkpoint, "temporal", spatial, epochs=1, batch_size=2)
            report = root / "report.json"
            evaluate(root, checkpoint, report, batch_size=2)
            result = json.loads(report.read_text())
            self.assertEqual(len(result["temporal_predictions"]), 2)
            self.assertEqual(result["unassessed"], [])
            self.assertIn("curves", result)
            history = json.loads(checkpoint.with_suffix('.history.json').read_text())
            self.assertIn('validation_loss', history[0])
            self.assertIn('training_accuracy', history[0])
            self.assertEqual(history[0]['training_samples'], 2)


    def test_sampler_epoch_length_and_amp_guard(self):
        import json
        from dataclasses import asdict
        import numpy as np
        from frameguard.data import Samples
        from frameguard.model import ModelConfig
        from frameguard.train import use_amp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for label, frames in ((0, 10), (1, 30)):
                name = f"train_{label}"
                np.savez_compressed(root / f"{name}.npz", crops=np.zeros((frames, 224, 224, 3), dtype=np.uint8),
                                    timestamps=np.arange(frames) / 8, tracks=np.zeros(frames, dtype=np.int64))
                rows.append(dict(dataset="synthetic-test-fixture", video_id=name, source_ids=name, identity_ids="",
                                 split="train", label=label, file=f"{name}.npz"))
            index = dict(format=1, config=asdict(ModelConfig()), videos=rows, rejected=[], manifest_sha256="test")
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            samples = Samples(root, "train")
            self.assertEqual(len(samples), 40)
            self.assertEqual(len(list(samples.balanced_sampler())), 40)
            drawn = list(samples.balanced_sampler(2000))
            self.assertEqual(len(drawn), 2000)
            fake_share = sum(samples.videos[samples.samples[i][0]]["label"] for i in drawn) / len(drawn)
            self.assertAlmostEqual(fake_share, 0.5, delta=0.06)
        self.assertFalse(use_amp("cpu", None))
        self.assertTrue(use_amp("cuda", None))
        self.assertFalse(use_amp("cuda", False))
        with self.assertRaises(ValueError):
            use_amp("cpu", True)


if __name__ == "__main__":
    unittest.main()
