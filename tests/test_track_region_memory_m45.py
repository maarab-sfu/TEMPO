import unittest
import numpy as np
from scripts.track_region_memory_m45 import containment,nms,associate_frames,camera_shift

class TestM45(unittest.TestCase):
    def test_containment_suppresses_nested_person(self):
        a={'label':'person','box':[0,0,100,100],'score':.9}
        b={'label':'person','box':[10,10,80,80],'score':.8}
        self.assertEqual(len(nms([a,b])),1)
    def test_distinct_chairs_remain(self):
        a={'label':'chair','box':[0,0,40,40],'score':.9}
        b={'label':'chair','box':[45,0,85,40],'score':.8}
        self.assertEqual(len(nms([a,b])),2)
    def test_camera_shift_changes_association(self):
        f=np.ones(10,dtype=np.float32)
        d=lambda frame,box:{'frame_index':frame,'timestamp':frame*.5,'box':box,'score':.9,'label':'chair','_feature':f}
        ds={0:[d(0,[0,0,20,20])],.5:[d(1,[150,0,170,20])]}
        without=associate_frames(ds,150,max_gap=2,max_cost=.62,min_appearance=0,shifts={})
        with_shift=associate_frames(ds,150,max_gap=2,max_cost=.62,min_appearance=0,shifts={1:(150,0)})
        self.assertEqual(len(without),2)
        self.assertEqual(len(with_shift),1)
    def test_blank_motion_fallback(self):
        blank=np.zeros((80,80,3),dtype=np.uint8)
        self.assertEqual(camera_shift(blank,blank),(0.,0.))
if __name__=='__main__':unittest.main()
