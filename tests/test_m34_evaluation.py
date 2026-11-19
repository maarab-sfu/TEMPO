import unittest
from scripts.evaluate_grounding import citation_validity, validate_questions

class EvaluationTests(unittest.TestCase):
    def test_valid_citations(self):
        self.assertTrue(citation_validity([3, 5], [{'frame_id': 3}, {'frame_id': 5}]))
    def test_invalid_citations(self):
        self.assertFalse(citation_validity([3, 99], [{'frame_id': 3}]))
        self.assertFalse(citation_validity([True], [{'frame_id': 1}]))
    def test_no_citations(self):
        self.assertIsNone(citation_validity([], [{'frame_id': 1}]))
    def test_duplicate_questions(self):
        with self.assertRaises(ValueError):
            validate_questions([{'question_id':'a','video_id':'v','question':'q'},
                                {'question_id':'a','video_id':'v','question':'q2'}])
if __name__ == '__main__':
    unittest.main()
