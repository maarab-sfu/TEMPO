import csv
import tempfile
import unittest
from pathlib import Path
from scripts.index_next_gqa import read_manifest


class ManifestTests(unittest.TestCase):
    def write(self, path, ids):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'archive_relative_path'])
            writer.writeheader()
            for id_ in ids:
                writer.writerow({'video_id': id_, 'archive_relative_path': 'a/b'})

    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'manifest.csv'
            self.write(p, ['123', '456'])
            self.assertEqual(read_manifest(p), ['123', '456'])

    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'manifest.csv'
            self.write(p, ['123', '123'])
            with self.assertRaises(ValueError):
                read_manifest(p)

    def test_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'manifest.csv'
            self.write(p, ['bad'])
            with self.assertRaises(ValueError):
                read_manifest(p)


if __name__ == '__main__':
    unittest.main()
