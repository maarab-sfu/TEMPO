import unittest,tempfile
from pathlib import Path
import numpy as np
from src.retrieval.object_hybrid import build_candidates,save_index,search_index,hybrid_select
class TestObjectHybrid(unittest.TestCase):
 def test_representatives(self):
  t=[{'track_id':'person_1','label':'person','observations':[{'timestamp':i,'frame_index':i,'box':[0,0,20,20],'score':.9} for i in range(10)]}]
  self.assertEqual(len(build_candidates(t)),3)
 def test_low_quality_track(self):
  t=[{'track_id':'x','label':'toy car','observations':[{'timestamp':0,'frame_index':0,'box':[0,0,20,20],'score':.1}]}]
  self.assertEqual(build_candidates(t),[])
 def test_index_search(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.npz';save_index([{'track_id':'x','label':'person'}],[[1,0]],p,'a.mp4')
   self.assertEqual(search_index(p,[1,0])[0]['track_id'],'x')
 def test_preserves_baseline(self):
  base=[{'timestamp':0.,'image_path':'a','evidence_role':'anchor'}]
  cat=base+[{'timestamp':2.,'image_path':'b'}]
  out=hybrid_select(base,[{'timestamp':2.,'track_id':'t','label':'person','object_similarity':.8}],cat,2,1)
  self.assertEqual([x['image_path'] for x in out],['a','b'])
  self.assertEqual(out[0]['evidence_role'],'anchor')
 def test_no_overflow(self):
  base=[{'timestamp':0.,'image_path':'a'},{'timestamp':1.,'image_path':'b'}]
  self.assertEqual(len(hybrid_select(base,[{'timestamp':2.,'track_id':'t','label':'person','object_similarity':.8}],base,2,1)),2)
if __name__=='__main__':unittest.main()
