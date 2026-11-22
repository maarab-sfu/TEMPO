import unittest
from scripts.select_object_evidence import select_evidence

class TestEvidence(unittest.TestCase):
    def frame(self,t,visible=True):
        return {'timestamp':float(t),'frame_index':t,'box':[t,0,t+10,10] if visible else None,'mask_pixels':100 if visible else 0}
    def test_chronological_and_budget(self):
        r=select_evidence({'frames':[self.frame(i) for i in range(12)]},budget=5)
        self.assertEqual(len(r['evidence']),5)
        self.assertEqual([f['timestamp'] for f in r['evidence']],sorted(f['timestamp'] for f in r['evidence']))
        self.assertEqual(r['evidence'][0]['timestamp'],0)
        self.assertEqual(r['evidence'][-1]['timestamp'],11)
    def test_no_visible(self):
        r=select_evidence({'frames':[self.frame(1,False)]})
        self.assertEqual(r['status'],'no_visible_object')
    def test_longest_run(self):
        r=select_evidence({'frames':[self.frame(0),self.frame(1,False),self.frame(2),self.frame(3),self.frame(4)]},budget=3)
        self.assertEqual(r['primary_run_interval'],[2.,4.])
        self.assertEqual(len(r['missing_intervals']),1)
    def test_bad_budget(self):
        with self.assertRaises(ValueError):select_evidence({'frames':[]},budget=0)

if __name__=='__main__':unittest.main()
