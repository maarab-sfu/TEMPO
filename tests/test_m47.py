import unittest
from types import SimpleNamespace
from scripts.evaluate_object_next_gqa import retrieval_hit, summarize, report

class TestM47(unittest.TestCase):
    def test_retrieval_hit(self):
        self.assertTrue(retrieval_hit([{"timestamp": 10}], [[11, 12]], 1))
        self.assertFalse(retrieval_hit([{"timestamp": 9}], [[11, 12]], 1))

    def test_summary(self):
        score = {
            "correct": True, "retrieval_hit": True, "citation_count": 2,
            "citation_hits": 1, "any_citation_in_gt": True
        }
        row = {
            "status": "ok", "baseline": score, "hybrid": score,
            "baseline_answer_seconds": 2, "hybrid_answer_seconds": 3,
            "question_group": "object_routed", "added_frames": 1,
            "sidecar_present": True
        }
        summary = summarize([row], "baseline")
        self.assertEqual(summary["accuracy"], 1)
        self.assertEqual(summary["citation_temporal_precision"], 0.5)
        overall = report([row], {"seed": 42}, SimpleNamespace(
            tolerance=1, object_slots=2, max_frames=8
        ))
        self.assertEqual(overall["changed_retrieval_pairs"], 1)
        self.assertEqual(overall["sidecars_present"], 1)

if __name__ == "__main__":
    unittest.main()
