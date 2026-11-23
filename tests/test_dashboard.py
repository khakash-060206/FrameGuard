import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from frameguard.archives import extract_member, grouped_splits
from frameguard.dashboard import snapshot, export_snapshot, Handler


class ArchiveTests(unittest.TestCase):
    def test_connected_sources_stay_together_and_seed_repeats(self):
        sources = [str(i) for i in range(30)]
        pairs = [(str(i), str(i + 1)) for i in range(0, 30, 2)]
        a = grouped_splits(sources, pairs)
        self.assertEqual(a, grouped_splits(sources, pairs))
        self.assertEqual(set(a.values()), {'train', 'val', 'test'})
        for first, second in pairs:
            self.assertEqual(a[first], a[second])

    def test_missing_original_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing original'):
            grouped_splits(['a'], [('a', 'missing')])

    def test_extraction_preserves_bytes_and_refuses_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('safe/video.mp4', b'test-video-bytes')
                archive.writestr('../outside.mp4', b'unsafe')
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                path = extract_member(archive, 'safe/video.mp4', root)
                self.assertEqual(path.read_bytes(), b'test-video-bytes')
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    extract_member(archive, '../outside.mp4', root)
                with self.assertRaisesRegex(ValueError, 'exists'):
                    extract_member(archive, 'safe/video.mp4', root)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('models', 'runs', 'dashboard', 'data/imported'):
            (self.root / folder).mkdir(parents=True)

    def test_empty_dashboard_does_not_invent_metrics(self):
        data = snapshot(self.root)
        self.assertEqual(data['reports'], [])
        self.assertEqual(data['histories'], [])
        self.assertFalse(data['can_predict'])

    def test_real_saved_reports_and_histories_are_loaded(self):
        report = {'metrics': {'temporal': {'fixture': {'accuracy': .75}}}, 'temporal_predictions': []}
        (self.root / 'runs/test.json').write_text(json.dumps(report))
        (self.root / 'models/model.history.json').write_text('[{"epoch": 1, "training_loss": 0.7}]')
        (self.root / 'data/imported/inventory.json').write_text('{"selected_videos": 10}')
        data = snapshot(self.root)
        self.assertEqual(data['reports'][0]['report'], report)
        self.assertEqual(data['histories'][0]['epochs'][0]['epoch'], 1)
        self.assertEqual(data['inventory']['selected_videos'], 10)

    def test_partial_json_is_reported_without_crashing(self):
        (self.root / 'runs/partial.json').write_text('{')
        data = snapshot(self.root)
        self.assertEqual(data['reports'], [])
        self.assertEqual(len(data['warnings']), 1)

    def test_snapshot_escape_and_no_unrestricted_file_endpoint(self):
        (self.root / 'data/imported/inventory.json').write_text('{"name":"</script>"}')
        text = export_snapshot(self.root).read_text()
        self.assertNotIn('</script>', text)
        self.assertIn('\\u003c/script>', text)
        handler = object.__new__(Handler)
        handler.server = type('Server', (), {'server_port': 8765})()
        handler.headers = {'Host': '127.0.0.1:8765', 'Origin': 'http://evil.example'}
        self.assertFalse(handler.allowed())
        handler.headers = {'Host': '127.0.0.1:8765', 'Origin': 'http://127.0.0.1:8765'}
        self.assertTrue(handler.allowed())
        handler.headers = {'Host': 'evil.example:8765'}
        self.assertFalse(handler.allowed())

    def test_dashboard_markup_has_all_referenced_elements(self):
        import re
        from html.parser import HTMLParser
        root = Path(__file__).resolve().parent.parent
        class IDs(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
            def handle_starttag(self, tag, attrs):
                self.ids.extend(value for key, value in attrs if key == 'id')
        parser = IDs()
        parser.feed((root / 'dashboard/index.html').read_text(encoding='utf-8'))
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        refs = set(re.findall(r"\$\('([^']+)'\)", (root / 'dashboard/app.js').read_text(encoding='utf-8')))
        self.assertFalse(refs - set(parser.ids), refs - set(parser.ids))


if __name__ == '__main__':
    unittest.main()
