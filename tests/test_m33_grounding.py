import ast
import unittest
from pathlib import Path

# Extract the pure parser without importing torch/transformers in unit tests.
source = Path(__file__).resolve().parents[1] / 'src/reasoning/vlm.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_grounded_response')
module = ast.Module(body=[function], type_ignores=[])
import json, re
namespace = {'json': json, 're': re}
exec(compile(module, str(source), 'exec'), namespace)
parse = namespace['parse_grounded_response']

class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{'frame_id': 42}, {'frame_id': 99}, {'frame_id': 100}]

    def test_valid_citations_map_to_actual_frame_ids(self):
        result = parse('{"answer":"Garlic is chopped.","cited_frames":[2,1,2]}', self.frames)
        self.assertEqual(result['cited_frame_ids'], [99, 42])
        self.assertEqual(result['grounding_status'], 'cited')

    def test_invalid_citations_are_rejected(self):
        result = parse('{"answer":"Unknown.","cited_frames":[0,4,-1,true,"2"]}', self.frames)
        self.assertEqual(result['cited_frame_ids'], [])
        self.assertEqual(result['grounding_status'], 'unverified')

    def test_unstructured_response_falls_back(self):
        result = parse('I cannot tell from these frames.', self.frames)
        self.assertEqual(result['answer'], 'I cannot tell from these frames.')
        self.assertFalse(result['citation_parse_ok'])

    def test_fenced_json(self):
        result = parse('```json\n{"answer":"Knife.","cited_frames":[3]}\n```', self.frames)
        self.assertEqual(result['cited_frame_ids'], [100])

if __name__ == '__main__':
    unittest.main()
