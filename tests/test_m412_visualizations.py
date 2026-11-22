import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.visualize_davis_predictions import colorize, overlay, visualize_sequence


class VisualizationTests(unittest.TestCase):
    def test_colorize_preserves_background_and_distinguishes_instances(self):
        mask = np.array([[0, 1, 2]], dtype=np.uint8)
        rgb = colorize(mask)
        self.assertEqual(rgb.shape, (1, 3, 3))
        self.assertEqual(rgb[0, 0].tolist(), [0, 0, 0])
        self.assertNotEqual(rgb[0, 1].tolist(), rgb[0, 2].tolist())

    def test_overlay_preserves_background(self):
        mask = np.array([[0, 1]], dtype=np.uint8)
        frame = Image.new('RGB', (2, 1), (100, 100, 100))
        rgb = np.asarray(overlay(frame, mask))
        self.assertEqual(rgb[0, 0].tolist(), [100, 100, 100])
        self.assertNotEqual(rgb[0, 1].tolist(), [100, 100, 100])

    def test_end_to_end_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames = root / 'DAVIS/JPEGImages/480p/demo'
            masks = root / 'predictions/demo'
            frames.mkdir(parents=True)
            masks.mkdir(parents=True)
            for i in range(3):
                Image.new('RGB', (12, 8), (80, 90, 100)).save(frames / f'{i:05d}.jpg')
                mask = np.zeros((8, 12), dtype=np.uint8)
                mask[1:4, 2:6] = 1
                Image.fromarray(mask).save(masks / f'{i:05d}.png')
            original = (masks / '00000.png').read_bytes()
            result = visualize_sequence(root / 'DAVIS', root / 'predictions',
                                        root / 'viz', 'demo', stride=2)
            self.assertEqual(result['frames_rendered'], 2)
            self.assertEqual(result['object_ids_seen'], [1])
            self.assertTrue((root / 'viz/contact_sheets/demo.jpg').is_file())
            self.assertTrue((root / 'viz/overlays/demo/00002.png').is_file())
            self.assertEqual((masks / '00000.png').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
