import unittest
from scripts.benchmark_davis_detector_sam2 import assign_detections, box_iou

class TestM411(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)

    def test_disjoint(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_one_to_one(self):
        boxes = {1: [0, 0, 10, 10], 2: [20, 20, 30, 30]}
        detections = [{'box': [20, 20, 30, 30]}, {'box': [0, 0, 10, 10]}]
        matched, ious = assign_detections(boxes, detections, 0.3)
        self.assertEqual(set(matched), {1, 2})
        self.assertEqual(ious[1], 1)

    def test_missed_object(self):
        matched, ious = assign_detections({1: [0, 0, 10, 10]}, [], 0.3)
        self.assertEqual(matched, {})
        self.assertEqual(ious[1], 0)

if __name__ == '__main__':
    unittest.main()
