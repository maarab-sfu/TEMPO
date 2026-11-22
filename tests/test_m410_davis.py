import unittest
import numpy as np
from scripts.benchmark_davis_sam2 import object_boxes, summarize


class TestDAVISBenchmark(unittest.TestCase):
    def test_object_boxes_excludes_void(self):
        labels = np.array([[0, 1, 1], [255, 2, 0]], dtype=np.uint8)
        self.assertEqual(object_boxes(labels), {1: [1.0, 0.0, 3.0, 1.0],
                                                 2: [1.0, 1.0, 2.0, 2.0]})

    def test_empty_metrics(self):
        self.assertIsNone(summarize([])['JF'])

    def test_macro_aggregation(self):
        rows = [
            {'sequence': 'a', 'object_id': 1, 'J': 1., 'F': 1., 'JF': 1.},
            {'sequence': 'a', 'object_id': 1, 'J': 0., 'F': 0., 'JF': 0.},
            {'sequence': 'b', 'object_id': 2, 'J': 1., 'F': 1., 'JF': 1.},
        ]
        self.assertAlmostEqual(summarize(rows)['JF'], .75)


if __name__ == '__main__':
    unittest.main()
