import unittest
from src.retrieval.object_hybrid import rank_object_hits,hybrid_select

class TestM461(unittest.TestCase):
    def test_toy_priority_and_diversity(self):
        hits=[{'track_id':'chair1','label':'chair','object_similarity':.9},
              {'track_id':'toy1','label':'toy car','object_similarity':.7},
              {'track_id':'toy1','label':'toy car','object_similarity':.6},
              {'track_id':'toy2','label':'toy car','object_similarity':.5}]
        out=rank_object_hits(hits,'What toy is the child holding?')
        self.assertEqual([x['track_id'] for x in out],['toy1','toy2','toy1'])
    def test_no_relevant_class_fallback(self):
        self.assertEqual(rank_object_hits([{'label':'chair','track_id':'a','object_similarity':.9}],
                                         'What toy is the child holding?'),[])
    def test_actual_new_frame_and_anchor_preservation(self):
        base=[{'timestamp':0,'image_path':'a','evidence_role':'semantic_anchor'},
              {'timestamp':2,'image_path':'b','evidence_role':'temporal_context'}]
        cat=base+[{'timestamp':4,'image_path':'c'}]
        hit=[{'timestamp':4,'track_id':'toy1','label':'toy car','object_similarity':.6}]
        out=hybrid_select(base,hit,cat,max_frames=2,object_slots=1,question='Which toy?')
        self.assertEqual({f['image_path'] for f in out},{'a','c'})
        self.assertEqual(next(f for f in out if f['image_path']=='c')['object_observation_timestamp'],4)
    def test_duplicate_keeps_baseline(self):
        base=[{'timestamp':0,'image_path':'a','evidence_role':'semantic_anchor'},
              {'timestamp':2,'image_path':'b','evidence_role':'temporal_context'}]
        hit=[{'timestamp':2,'track_id':'toy1','label':'toy car','object_similarity':.8}]
        self.assertEqual(hybrid_select(base,hit,base,2,1,question='Which toy?'),base)
    def test_no_mutation(self):
        base=[{'timestamp':0,'image_path':'a','evidence_role':'semantic_anchor'}]
        hybrid_select(base,[],base)
        self.assertEqual(len(base),1)
if __name__=='__main__': unittest.main()
