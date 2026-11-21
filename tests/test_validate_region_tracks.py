import unittest
from scripts.validate_region_tracks import overlap,analyze

class TestValidation(unittest.TestCase):
    def test_overlap(self):
        i,c=overlap([0,0,10,10],[0,0,10,10]);self.assertEqual((i,c),(1,1))
        i,c=overlap([0,0,10,10],[20,20,30,30]);self.assertEqual((i,c),(0,0))
        i,c=overlap([0,0,10,10],[2,2,4,4]);self.assertAlmostEqual(c,1)
    def test_duplicate_flags(self):
        def t(id,b):return {'track_id':id,'label':'person','observations':[{'frame_index':0,'timestamp':0.,'box':b,'score':.8}]}
        doc={'tracks':[t('p1',[0,0,10,10]),t('p2',[1,1,9,9])]}
        summary,flags,_=analyze(doc,'2s')
        self.assertEqual(summary['overlap_conflict_events'],1)
        self.assertEqual(summary['singleton_tracks'],2)
    def test_distinct_objects_not_conflicts(self):
        def t(id,b):return {'track_id':id,'label':'chair','observations':[{'frame_index':0,'timestamp':0.,'box':b,'score':.8}]}
        doc={'tracks':[t('c1',[0,0,10,10]),t('c2',[20,20,30,30])]}
        self.assertEqual(analyze(doc,'2s')[0]['overlap_conflict_events'],0)
if __name__=='__main__':unittest.main()
