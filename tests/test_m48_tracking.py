import unittest
from scripts.experiment_sam2_tracking import choose_seed, scale_box, box_iou


class TrackingHelpersTests(unittest.TestCase):
    def test_choose_seed(self):
        tracks = {'tracks': [{'track_id': 'car', 'observations': [
            {'timestamp': 2, 'score': 0.3}, {'timestamp': 5, 'score': 0.9}]}]}
        self.assertEqual(choose_seed(tracks, 'car', None)['timestamp'], 5)
        self.assertEqual(choose_seed(tracks, 'car', 2)['timestamp'], 2)

    def test_rescale_from_detector_space(self):
        self.assertEqual(scale_box([100, 50, 300, 150], 1600, 800, 800, 400),
                         [100, 50, 300, 150])
        self.assertEqual(scale_box([100, 50, 300, 150], 1600, 800, 1600, 800),
                         [200, 100, 600, 300])

    def test_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)


if __name__ == '__main__':
    unittest.main()
