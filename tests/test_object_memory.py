import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.memory.object_memory import build_memory, save_memory

class ObjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{'frame_id': f'f{i}', 'timestamp': t} for i,t in enumerate([0,2,4,12])]
        self.images = np.array([[1,0],[1,0],[0,1],[1,0]], dtype=np.float32)
        self.texts = np.eye(2, dtype=np.float32)

    def test_episode_gap(self):
        result = build_memory(self.frames,self.images,['knife','bowl'],self.texts,
                              top_labels=1,min_similarity=.8,max_gap_seconds=4.1)
        knife = [e for e in result['episodes'] if e['label']=='knife']
        self.assertEqual(len(knife),2)
        self.assertEqual(knife[0]['frame_ids'],['f0','f1'])
        self.assertEqual(knife[1]['frame_ids'],['f3'])

    def test_label_matches(self):
        result = build_memory(self.frames,self.images,['knife','bowl'],self.texts,
                              top_labels=1,min_similarity=.8)
        self.assertEqual(len(result['observations']),4)
        self.assertEqual(result['observations'][2]['label'],'bowl')

    def test_bad_dimensions(self):
        with self.assertRaises(ValueError):
            build_memory(self.frames,self.images,['knife'],self.texts)

    def test_empty_labels(self):
        with self.assertRaises(ValueError):
            build_memory(self.frames,self.images,[],self.texts)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'object_memory_v1.json'
            save_memory(path,{'schema_version':1})
            self.assertEqual(json.loads(path.read_text())['schema_version'],1)

if __name__ == '__main__':
    unittest.main()
