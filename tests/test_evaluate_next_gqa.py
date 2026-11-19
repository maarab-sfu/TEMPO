import unittest
from scripts.evaluate_next_gqa import correct_letter, parse_choice, citation_metrics, summarize, mc_prompt

class TestNextGQA(unittest.TestCase):
    def setUp(self):
        self.row = dict(video_id='1',qid='2',type='CW',question='What?',answer='third',
                        a0='first',a1='second',a2='third',a3='fourth',a4='fifth')
    def test_gold(self):
        self.assertEqual(correct_letter(self.row), 'C')
    def test_prompt(self):
        p = mc_prompt(self.row)
        self.assertIn('C. third',p)
        self.assertIn('ONLY one letter',p)
    def test_parse(self):
        for s in ['C','C.','Option C','Answer: C','The answer is C']:
            self.assertEqual(parse_choice(s),'C')
        for s in ['A or B','I think C','C. third','', 'F']:
            self.assertIsNone(parse_choice(s))
    def test_citations(self):
        frames = [{'frame_id':1,'timestamp':5.0},{'frame_id':2,'timestamp':15.0}]
        m = citation_metrics(frames,[1,2],[[4.0,6.0]],0)
        self.assertTrue(m['citation_valid'])
        self.assertEqual(m['citation_hits'],1)
        self.assertEqual(m['citation_precision_temporal'],.5)
        self.assertFalse(citation_metrics(frames,[999],[[4,6]])['citation_valid'])
        self.assertEqual(citation_metrics(frames,[],[[4,6]])['citation_count'],0)
    def test_summary(self):
        record = dict(status='ok',correct=True,type='CW',citation_count=1,citation_hits=1,
                      citation_valid=True,any_citation_in_gt=True,latency_seconds=2)
        s=summarize([record])
        self.assertEqual(s['accuracy'],1)
        self.assertEqual(s['temporal_citation_precision'],1)

if __name__ == '__main__': unittest.main()
