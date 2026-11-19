import unittest
from scripts.ablate_next_gqa import evidence_hit, select_coverage, retrieval_query, metrics

class TestAblations(unittest.TestCase):
    def setUp(self):
        self.frames = [dict(frame_id=i, image_path=f'{i}.jpg', timestamp=float(i*2)) for i in range(12)]
    def test_coverage_preserves_anchors_and_budget(self):
        anchors = [dict(self.frames[2], similarity=.8), dict(self.frames[3], similarity=.7)]
        result = select_coverage(self.frames, anchors, 8)
        self.assertEqual(len(result), 8)
        self.assertEqual(len(set(f['frame_id'] for f in result)), 8)
        self.assertTrue({2,3}.issubset({f['frame_id'] for f in result}))
        self.assertEqual(sum(f['evidence_role']=='semantic_anchor' for f in result), 2)
        self.assertEqual([f['timestamp'] for f in result], sorted(f['timestamp'] for f in result))
    def test_short_video(self):
        self.assertEqual(len(select_coverage(self.frames[:3], [self.frames[0]], 8)), 3)
    def test_no_gold_leak(self):
        row = dict(question='What happens?', answer='secret', a0='walk',a1='run',a2='jump',a3='sit',a4='sleep')
        self.assertNotIn('secret', retrieval_query(row, 'option_query'))
        self.assertIn('sleep', retrieval_query(row, 'option_query'))
        self.assertEqual(retrieval_query(row, 'coverage'), 'What happens?')
    def test_tolerance(self):
        self.assertTrue(evidence_hit([10], [[11,12]], 1))
        self.assertFalse(evidence_hit([10], [[11.1,12]], 1))
    def test_metrics(self):
        r = dict(status='ok',correct=True,type='CW',citation_count=1,citation_hits=1,
                 citation_valid=True,any_citation_in_gt=True,latency_seconds=2,
                 retrieved_timestamps=[3], evidence_intervals=[[2,4]])
        self.assertEqual(metrics([r],1)['retrieval_hit_rate'],1)
        self.assertEqual(metrics([r],1)['citation_hit_given_retrieval_hit'],1)

if __name__ == '__main__': unittest.main()
