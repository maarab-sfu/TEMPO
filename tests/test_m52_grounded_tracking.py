import unittest
from scripts.track_grounded_object import eligible, scale_to_frame

class TestM52(unittest.TestCase):
    def test_reject_giant_boxes(self):
        c=[dict(score=.9,area_fraction=.95,box=[0,0,640,360],timestamp=1),
           dict(score=.8,area_fraction=.08,box=[10,10,50,50],timestamp=2)]
        self.assertEqual(len(eligible(c)),1)
        self.assertEqual(eligible(c)[0]['timestamp'],2)
    def test_sort(self):
        c=[dict(score=s,area_fraction=.05,box=[1,1,3,3],timestamp=i) for i,s in enumerate([.5,.8])]
        self.assertEqual(eligible(c)[0]['score'],.8)
    def test_scale(self):
        self.assertEqual(scale_to_frame([100,50,200,100],640,360,320,180),[50,25,100,50])
if __name__=='__main__':unittest.main()
