import unittest
from scripts.build_region_memory import iou, sample_positions, associate
class Tests(unittest.TestCase):
    def test_sampling(self): self.assertEqual(sample_positions(100,10,2),[0,20,40,60,80])
    def test_iou(self):
        self.assertEqual(iou([0,0,2,2],[0,0,2,2]),1)
        self.assertEqual(iou([0,0,1,1],[2,2,3,3]),0)
    def test_association(self):
        d=[{'frame_index':0,'timestamp':0.,'label':'knife','box':[0,0,10,10],'score':.8},
           {'frame_index':60,'timestamp':2.,'label':'knife','box':[1,0,11,10],'score':.7},
           {'frame_index':120,'timestamp':4.,'label':'cup','box':[1,0,11,10],'score':.7}]
        tracks=associate(d)
        self.assertEqual(len(tracks),2)
        self.assertEqual(tracks[0]['observation_count'],2)
    def test_gap(self):
        d=[{'frame_index':0,'timestamp':0.,'label':'knife','box':[0,0,10,10],'score':.8},
           {'frame_index':300,'timestamp':10.,'label':'knife','box':[0,0,10,10],'score':.8}]
        self.assertEqual(len(associate(d)),2)
if __name__=='__main__': unittest.main()
