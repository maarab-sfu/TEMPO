import unittest
from src.retrieval.temporal_retriever import TemporalRetriever

class FakeRetriever:
    def __init__(self):
        self.metadata = [{"frame_id": i, "timestamp": float(i * 2), "image_path": f"f{i}.jpg"} for i in range(6)]
    def search(self, embedding, top_k=2):
        return [dict(self.metadata[i], similarity=score) for i, score in [(2, 0.72), (3, 0.63)]][:top_k]

class EvidenceTests(unittest.TestCase):
    def test_roles_scores_and_anchor_preservation(self):
        r = FakeRetriever()
        frames = TemporalRetriever(r).search_with_context(None, top_k=2, context_before=2, context_after=2, max_frames=5)
        self.assertEqual(len(frames), 5)
        by_id = {f["frame_id"]: f for f in frames}
        self.assertEqual(by_id[2]["anchor_rank"], 1)
        self.assertEqual(by_id[3]["anchor_rank"], 2)
        self.assertAlmostEqual(by_id[2]["similarity"], 0.72)
        self.assertEqual(by_id[2]["evidence_role"], "semantic_anchor")
        for i in set(by_id) - {2, 3}:
            self.assertEqual(by_id[i]["evidence_role"], "temporal_context")
            self.assertIsNone(by_id[i]["similarity"])
            self.assertIsNone(by_id[i]["anchor_rank"])
        self.assertNotIn("evidence_role", r.metadata[2])
    def test_no_budget_and_zero_context(self):
        frames = TemporalRetriever(FakeRetriever()).search_with_context(None, top_k=2, context_before=0, context_after=0)
        self.assertEqual([f["frame_id"] for f in frames], [2, 3])

if __name__ == "__main__":
    unittest.main()
