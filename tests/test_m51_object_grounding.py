import unittest
from scripts.query_object_grounding import Proposal, box_area_fraction, normalize_object_phrase, parse_timestamps, select_proposals

class TestM51(unittest.TestCase):
    def test_phrase(self):
        self.assertEqual(normalize_object_phrase(' Toy   Car. '), 'toy car')
        with self.assertRaises(ValueError): normalize_object_phrase(' ')
    def test_area(self):
        self.assertAlmostEqual(box_area_fraction([-10,0,50,100],100,100), .5)
        self.assertEqual(box_area_fraction([20,20,10,10],100,100), 0)
    def test_timestamps(self):
        self.assertEqual(parse_timestamps('1, 2.5'), [1,2.5])
        with self.assertRaises(ValueError): parse_timestamps('nan')
    def test_diverse_selection(self):
        def p(t,s): return Proposal(t,int(t*10),[0,0,10,10],'car',s,.1,s)
        result=select_proposals([p(1,.9),p(1.1,.8),p(4,.7)],2,.5)
        self.assertEqual([x.timestamp for x in result],[1,4])
    def test_no_candidates(self):
        self.assertEqual(select_proposals([]), [])

if __name__ == '__main__': unittest.main()
