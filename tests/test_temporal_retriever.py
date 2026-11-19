"""Run: python -m unittest discover -s tests -p test_temporal_retriever.py
Place this file in tests/test_temporal_retriever.py.
"""
import unittest
from src.retrieval.temporal_retriever import TemporalRetriever


class FakeRetriever:
    def __init__(self, n, anchors):
        self.metadata = [dict(frame_id=i, timestamp=float(i * 2), image_path=f"f{i}") for i in range(n)]
        self.anchors = anchors

    def search(self, query_embedding, top_k):
        return [self.metadata[i].copy() for i in self.anchors[:top_k]]


class TemporalRetrieverTests(unittest.TestCase):
    def test_preserves_both_anchors_under_budget(self):
        frames = TemporalRetriever(FakeRetriever(30, [5, 25])).search_with_context(
            None, top_k=2, context_before=2, context_after=2, max_frames=8
        )
        ids = [f['frame_id'] for f in frames]
        self.assertEqual(len(ids), 8)
        self.assertTrue({5, 25}.issubset(ids))
        self.assertEqual(ids, sorted(ids))

    def test_overlapping_neighborhoods_are_unique(self):
        frames = TemporalRetriever(FakeRetriever(12, [5, 6])).search_with_context(
            None, top_k=2, context_before=2, context_after=2, max_frames=8
        )
        ids = [f['frame_id'] for f in frames]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({5, 6}.issubset(ids))
        self.assertLessEqual(len(ids), 8)

    def test_short_video(self):
        frames = TemporalRetriever(FakeRetriever(3, [0, 2])).search_with_context(
            None, top_k=2, max_frames=8
        )
        self.assertEqual(len(frames), 3)

    def test_budget_smaller_than_anchors_is_rejected(self):
        with self.assertRaises(ValueError):
            TemporalRetriever(FakeRetriever(10, [2, 8])).search_with_context(
                None, top_k=2, max_frames=1
            )

    def test_no_budget_preserves_previous_behavior(self):
        frames = TemporalRetriever(FakeRetriever(20, [5, 15])).search_with_context(
            None, top_k=2, context_before=2, context_after=2
        )
        self.assertEqual(len(frames), 10)


if __name__ == '__main__':
    unittest.main()
