import unittest
import json
import sys
import types
from unittest.mock import patch
from scripts.ablate_attribution_next_gqa import interval_recall, summary, select_citations

class FakeVLM:
    def __init__(self, outputs):self.outputs=iter(outputs)
    def answer_video_frames(self, frames, question, max_new_tokens, attribution_answer):
        return next(self.outputs)

def fake_parse(raw, frames, answer):
    payload=json.loads(raw)
    return {"citation_parse_ok": True, "cited_frame_ids": [frames[i-1]["frame_id"] for i in payload["cited_frames"]]}

class AttributionTests(unittest.TestCase):
    def setUp(self):
        module=types.ModuleType("src.reasoning.vlm")
        module.parse_attribution_response=fake_parse
        self.patch=patch.dict(sys.modules, {"src":types.ModuleType("src"), "src.reasoning":types.ModuleType("src.reasoning"), "src.reasoning.vlm":module})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.frames=[{'frame_id':i,'timestamp':float(i*2)} for i in range(1,5)]
    def test_candidate_mapping(self):
        vlm=FakeVLM(['{"cited_frames":[1]}','{"cited_frames":[]}',
                     '{"cited_frames":[1]}','{"cited_frames":[]}'])
        ids,valid,calls=select_citations(vlm,self.frames,'q','A','candidate')
        self.assertEqual(ids,[1,3]);self.assertEqual((valid,calls),(4,4))
    def test_window_mapping(self):
        vlm=FakeVLM(['{"cited_frames":[2]}','{"cited_frames":[1]}'])
        ids,valid,calls=select_citations(vlm,self.frames,'q','A','window')
        self.assertEqual(ids,[2,3]);self.assertEqual((valid,calls),(2,2))
    def test_cap(self):
        vlm=FakeVLM(['{"cited_frames":[1]}']*4)
        ids,_,_=select_citations(vlm,self.frames,'q','A','candidate',max_citations=2)
        self.assertEqual(ids,[1,2])
    def test_interval_recall(self):
        self.assertEqual(interval_recall([3,10],[[2,4],[7,8],[9,11]],0),(2,3))
    def test_empty_summary(self):
        self.assertEqual(summary([])['completed'],0)

if __name__=='__main__':unittest.main()
