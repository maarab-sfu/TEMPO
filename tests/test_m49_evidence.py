import unittest
from scripts.select_tracking_evidence import center, select_indices, trajectory_motion


class TestTrackingEvidence(unittest.TestCase):
    def test_center(self):
        self.assertEqual(center([0, 2, 10, 6]), (5, 4))

    def test_motion(self):
        rows = [
            {"timestamp": 0, "sam2_box": [0, 0, 2, 2]},
            {"timestamp": 1, "sam2_box": [3, 4, 5, 6]},
            {"timestamp": 2, "sam2_box": None},
        ]
        self.assertEqual(trajectory_motion(rows), [0, 5, 0])

    def test_endpoints_and_spacing(self):
        rows = [
            {"timestamp": i, "sam2_box": [i, 0, i + 2, 2]}
            for i in range(10)
        ]
        selected = select_indices(rows, 4, min_spacing=2)
        self.assertEqual(len(selected), 4)
        self.assertIn(0, selected)
        self.assertIn(9, selected)
        self.assertTrue(all(b - a >= 2 for a, b in zip(selected, selected[1:])))

    def test_empty(self):
        self.assertEqual(select_indices([{"timestamp": 0, "sam2_box": None}], 8), [])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            select_indices([], 0)


if __name__ == "__main__":
    unittest.main()
