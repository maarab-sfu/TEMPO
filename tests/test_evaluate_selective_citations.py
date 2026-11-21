import unittest
from scripts.evaluate_selective_citations import choose, evaluate, is_hit


class SelectiveCitationTests(unittest.TestCase):
    def setUp(self):
        self.row = {'correct': True, 'ids': [1, 2, 3], 'timestamps': [2.0, 6.0, 10.0]}
        self.gt = {('v', 'q'): {'correct': True, 'intervals': [[5.0, 7.0]]}}
        self.rows = {('v', 'q'): self.row}

    def test_selection_order(self):
        self.assertEqual(choose(self.row, 'first1', [[5, 7]], 1), [(1, 2.0)])
        self.assertEqual(choose(self.row, 'last2', [[5, 7]], 1), [(2, 6.0), (3, 10.0)])

    def test_oracle_diagnostic(self):
        self.assertEqual(choose(self.row, 'oracle1', [[5, 7]], 1), [(2, 6.0)])

    def test_temporal_tolerance(self):
        self.assertTrue(is_hit(4.0, [[5, 7]], 1))
        self.assertFalse(is_hit(3.99, [[5, 7]], 1))

    def test_metrics(self):
        report, _ = evaluate(self.rows, self.gt, 'all3', 1)
        self.assertAlmostEqual(report['temporal_citation_precision'], 1/3)
        self.assertEqual(report['joint_correct_and_temporally_grounded_rate'], 1)
        report, _ = evaluate(self.rows, self.gt, 'first1', 1)
        self.assertEqual(report['any_temporal_hit_rate'], 0)

    def test_reject_mismatched_questions(self):
        with self.assertRaises(ValueError):
            evaluate({}, self.gt, 'first1', 1)

    def test_reject_correctness_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate({('v','q'): {**self.row, 'correct': False}}, self.gt, 'first1', 1)


if __name__ == '__main__': unittest.main()
