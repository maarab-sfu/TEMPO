import unittest
import numpy as np
from scripts.track_region_memory import iou,nms,associate_frames


def d(t,idx,label='person',box=(0,0,20,20),score=.9):
    return {'timestamp':t,'frame_index':idx,'label':label,'box':list(box),'score':score,'_feature':np.ones(384,dtype=np.float32)/20}

class TrackingTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0,0,10,10],[20,20,30,30]),0)
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
    def test_nms(self):
        ds=[d(0,0),d(0,0,score=.4),d(0,0,label='chair')]
        self.assertEqual(len(nms(ds)),2)
    def test_identity_stable(self):
        tracks=associate_frames({0:[d(0,0)],2:[d(2,60,box=(2,0,22,20))]},100)
        self.assertEqual(len(tracks),1)
        self.assertEqual(tracks[0]['observation_count'],2)
    def test_gap(self):
        tracks=associate_frames({0:[d(0,0)],8:[d(8,240)]},100)
        self.assertEqual(len(tracks),2)
    def test_one_to_one(self):
        tracks=associate_frames({0:[d(0,0)],2:[d(2,60),d(2,60,box=(40,0,60,20))]},100)
        self.assertEqual(sorted(t['observation_count'] for t in tracks),[1,2])
    def test_label_gate(self):
        tracks=associate_frames({0:[d(0,0)],2:[d(2,60,label='chair')]},100)
        self.assertEqual(len(tracks),2)

if __name__=='__main__':unittest.main()
