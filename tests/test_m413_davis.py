import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.analyze_davis_failures import analyze

class AnalysisTest(unittest.TestCase):
    def test_matched_and_missed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            gt=pd.DataFrame([dict(sequence='s',frame=i,object_id=o,J=.8,F=.8,JF=.8) for o in [1,2] for i in [1,2,3]])
            det=gt.copy();det.loc[det.object_id==1,['J','F','JF']]=.6;det.loc[det.object_id==2,['J','F','JF']]=0
            gt.to_csv(root/'gt.csv',index=False);det.to_csv(root/'det.csv',index=False)
            (root/'detections.json').write_text(json.dumps([dict(sequence='s',gt_objects=2,matched_boxes={'1':[0,0,2,2]},missed_object_ids=[2],first_frame_match_iou={'1':.9,'2':0})]))
            s=analyze(root/'gt.csv',root/'det.csv',root/'detections.json',root/'out')
            self.assertEqual(s['matched_objects'],1)
            self.assertAlmostEqual(s['all_objects']['delta_JF'],-.5)
            self.assertAlmostEqual(s['macro_delta_contribution_missed'],-.4)
    def test_mismatched_frames_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for fn,frame in [('gt.csv',1),('det.csv',2)]:
                pd.DataFrame([dict(sequence='s',frame=frame,object_id=1,J=.8,F=.8,JF=.8)]).to_csv(root/fn,index=False)
            (root/'detections.json').write_text(json.dumps([dict(sequence='s',gt_objects=1,matched_boxes={'1':[0,0,1,1]},missed_object_ids=[],first_frame_match_iou={'1':.8})]))
            with self.assertRaises(ValueError):analyze(root/'gt.csv',root/'det.csv',root/'detections.json',root/'out')
if __name__=='__main__':unittest.main()
