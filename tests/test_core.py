import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from frameguard.manifest import FIELDS, read_manifest
from frameguard.indexing import index_celeb, index_ffpp, merge_manifests
from frameguard.temporal import Track, Tracker, iou, window_starts


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, rows):
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, name="a", split="train", source="001", identity=""):
        (self.root / f"{name}.mp4").touch()
        return dict(path=f"{name}.mp4", dataset="fixture", video_id=name, source_ids=source,
                    identity_ids=identity, label="0", split=split)

    def test_resolves_relative_paths(self):
        rows = read_manifest(self.manifest([self.row()]))
        self.assertEqual(rows[0]["path"], str(self.root / "a.mp4"))
        self.assertEqual(rows[0]["label"], 0)

    def test_related_fake_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            read_manifest(self.manifest([self.row(), self.row("fake", "test", "002|001")]))

    def test_identity_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            read_manifest(self.manifest([self.row(identity="person"), self.row("b", "val", "002", "person")]))

    def test_duplicate_paths_rejected(self):
        first, second = self.row(), self.row("b")
        second["path"] = first["path"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_manifest(self.manifest([first, second]))

    def test_invalid_label_rejected(self):
        row = self.row()
        row["label"] = "2"
        with self.assertRaisesRegex(ValueError, "Invalid label"):
            read_manifest(self.manifest([row]))

    def test_merge_validates_leakage_across_manifests(self):
        first = self.manifest([self.row()])
        saved = self.root / "first.csv"
        first.rename(saved)
        second = self.manifest([self.row("b", "test", "001")])
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            merge_manifests([saved, second], self.root / "merged.csv")
        self.assertFalse((self.root / "merged.csv").exists())

    def test_missing_video_rejected(self):
        row = self.row()
        row["path"] = "absent.mp4"
        with self.assertRaisesRegex(ValueError, "Missing video"):
            read_manifest(self.manifest([row]))

    def test_empty_source_identifier_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty identifier"):
            read_manifest(self.manifest([self.row(source="001||002")]))

    def test_ffpp_preserves_official_splits(self):
        splits = self.root / "splits"
        splits.mkdir()
        for split, pair in (("train", ["001", "002"]), ("val", ["003", "004"]), ("test", ["005", "006"])):
            (splits / f"{split}.json").write_text(json.dumps([pair]))
        original = self.root / "original_sequences/youtube/c23/videos"
        fake = self.root / "manipulated_sequences/Deepfakes/c23/videos"
        original.mkdir(parents=True)
        fake.mkdir(parents=True)
        (original / "001.mp4").touch()
        (fake / "001_002.mp4").touch()
        rows = index_ffpp(self.root, splits, self.root / "out.csv")
        self.assertEqual([row["label"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["source_ids"], "001|002")
        self.assertEqual(rows[1]["split"], "train")

    def test_celeb_label_encoding_is_inverted(self):
        (self.root / "Celeb-real").mkdir()
        (self.root / "Celeb-synthesis").mkdir()
        (self.root / "Celeb-real/id0_0000.mp4").touch()
        (self.root / "Celeb-synthesis/id0_id1_0000.mp4").touch()
        (self.root / "List_of_testing_videos.txt").write_text("1 Celeb-real/id0_0000.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n")
        rows = index_celeb(self.root, self.root / "celeb.csv")
        self.assertEqual([row["label"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["identity_ids"], "id0|id1")

    def test_celeb_path_cannot_escape_root(self):
        (self.root / "List_of_testing_videos.txt").write_text("1 ../escape.mp4\n")
        with self.assertRaisesRegex(ValueError, "outside"):
            index_celeb(self.root, self.root / "celeb.csv")


class TemporalTests(unittest.TestCase):
    def test_sliding_windows_match_agreed_cadence(self):
        track = Track((0, 0, 10, 10), 0)
        windows = []
        for index in range(24):
            window = track.append(index / 8, index + 1)
            if window:
                windows.append([value for _, value in window])
        self.assertEqual(windows, [list(range(1, 17)), list(range(5, 21)), list(range(9, 25))])
        self.assertEqual(len(track.items), 16)

    def test_gap_restarts_window(self):
        track = Track((0, 0, 10, 10), 0)
        for index in range(15):
            self.assertIsNone(track.append(index / 8, index))
        self.assertIsNone(track.append(5, 16))
        self.assertEqual(len(track.items), 1)

    def test_training_windows_restart_stride_after_gap(self):
        times = [i / 8 for i in range(17)] + [10 + i / 8 for i in range(20)]
        self.assertEqual(list(window_starts(times)), [0, 17, 21])

    def test_two_faces_have_separate_tracks(self):
        tracker = Tracker()
        first = tracker.update([(0, 0, 10, 10), (30, 30, 40, 40)], 0)
        second = tracker.update([(31, 30, 41, 40), (1, 0, 11, 10)], 0.125)
        self.assertEqual(second, first[::-1])

    def test_missing_face_discards_track(self):
        tracker = Tracker()
        old = tracker.update([(0, 0, 10, 10)], 0)
        tracker.update([], 0.125)
        new = tracker.update([(0, 0, 10, 10)], 0.25)
        self.assertNotEqual(old, new)

    def test_ambiguous_matching_resets(self):
        tracker = Tracker()
        first = tracker.update([(0, 0, 10, 10)], 0)
        second = tracker.update([(0, 0, 10, 10), (1, 1, 11, 11)], 0.125)
        self.assertTrue(set(first).isdisjoint(second))

    def test_scene_change_and_long_gap_reset(self):
        tracker = Tracker()
        a = tracker.update([(0, 0, 10, 10)], 0)
        b = tracker.update([(0, 0, 10, 10)], 0.125, reset=True)
        c = tracker.update([(0, 0, 10, 10)], 10)
        self.assertEqual(len(set(a + b + c)), 3)

    def test_invalid_box_and_timestamp_rejected(self):
        for box, timestamp in [((0, 0, -1, 10), 0), ((0, 0, 10, 10), float("nan"))]:
            with self.assertRaises(ValueError):
                Tracker().update([box], timestamp)

    def test_iou(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0)


NL = bytes([10])


class ImportTests(unittest.TestCase):
    """ZIP and extracted-folder inputs must give identical video IDs, labels and splits."""

    def build(self, root):
        ffpp = root / "ffpp-folder" / "nested"
        celeb = root / "celeb-folder"
        files = {}
        for index in range(20):
            files[ffpp / f"FaceForensics++_C23/original/{index:03d}.mp4"] = b"real"
        for index in range(0, 20, 2):  # 10 independent source groups
            files[ffpp / f"FaceForensics++_C23/Deepfakes/{index:03d}_{index + 1:03d}.mp4"] = b"fake"
        files[ffpp / "FaceForensics++_C23/Face2Face/000_001.mp4"] = b"ignored"
        for name in ("Celeb-real/id0_0000.mp4", "Celeb-synthesis/id0_id1_0000.mp4", "YouTube-real/00001.mp4"):
            files[celeb / name] = b"video"
        files[celeb / "List_of_testing_videos.txt"] = (
            b"1 Celeb-real/id0_0000.mp4" + NL + b"0 Celeb-synthesis/id0_id1_0000.mp4" + NL + b"1 YouTube-real/00001.mp4" + NL)
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        zips = []
        for folder, name in ((ffpp, "ffpp.zip"), (celeb, "celeb.zip")):
            with zipfile.ZipFile(root / name, "w") as archive:
                for path in folder.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(folder).as_posix())
            zips.append(root / name)
        return (root / "ffpp-folder", celeb), zips

    def test_zip_and_folder_give_identical_split(self):
        from frameguard.archives import import_archives
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders, zips = self.build(root)
            import_archives(*zips, root / "from-zip")
            import_archives(*folders, root / "from-folder")
            def load(name):
                rows = read_manifest(root / name / "manifest.csv")
                return {(r["dataset"], r["video_id"]): (r["label"], r["split"]) for r in rows}, rows
            from_zip, _ = load("from-zip")
            from_folder, rows = load("from-folder")
            self.assertEqual(from_zip, from_folder)
            self.assertEqual(len(from_folder), 33)
            self.assertEqual(from_folder[("Celeb-DF-v2", "Celeb-synthesis/id0_id1_0000.mp4")], (1, "external"))
            # Folder inputs are referenced in place, not copied.
            self.assertTrue(all(Path(r["path"]).is_relative_to(root / "ffpp-folder") or
                                Path(r["path"]).is_relative_to(root / "celeb-folder") for r in rows))
            self.assertFalse((root / "from-folder" / "ffpp").exists())

    def test_missing_folder_marker_rejected(self):
        from frameguard.archives import import_archives
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty").mkdir()
            with self.assertRaisesRegex(ValueError, "Could not find FaceForensics"):
                import_archives(root / "empty", root / "empty", root / "out")


if __name__ == "__main__":
    unittest.main()

