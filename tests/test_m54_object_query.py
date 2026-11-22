import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from src.api.object_query import ObjectQueryRequest, merge_frames, video_source

class M54Tests(unittest.TestCase):
    def test_request_bounds(self):
        with self.assertRaises(Exception):ObjectQueryRequest(question='x',object_phrase='car',evidence_budget=9)
    def test_invalid_video_id(self):
        with self.assertRaises(ValueError):video_source('../foo')
    def test_semantic_preserved(self):
        with TemporaryDirectory() as d:
            sem=[{'timestamp':1,'image_path':'anchor.jpg'}]
            frames,added=merge_frames(sem,[],Path(d)/'missing.mp4',Path(d))
            self.assertEqual(frames,sem);self.assertEqual(added,[])
    def test_no_duplicate_timestamp(self):
        with TemporaryDirectory() as d:
            sem=[{'timestamp':1,'image_path':'anchor.jpg'}]
            with patch('cv2.VideoCapture') as cap:
                cap.return_value.isOpened.return_value=True
                frames,added=merge_frames(sem,[{'timestamp':1.1,'frame_index':30}],Path(d)/'fake.mp4',Path(d))
                self.assertEqual(len(frames),1)
if __name__=='__main__':unittest.main()
