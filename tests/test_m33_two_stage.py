import ast
import json
import re
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "src/reasoning/vlm.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_attribution_response")
ns = {"json": json, "re": re}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
parse = ns["parse_attribution_response"]
frames = [{"frame_id": 4}, {"frame_id": 5}, {"frame_id": 6}]

class TwoStageTests(unittest.TestCase):
    def test_valid_citations_and_answer_unchanged(self):
        r = parse('{"cited_frames":[2,1,2]}', frames, "chef's knife")
        self.assertEqual(r["answer"], "chef's knife")
        self.assertEqual(r["cited_frame_ids"], [5, 4])
        self.assertEqual(r["grounding_status"], "cited")

    def test_hallucinated_numbers_rejected(self):
        r = parse('{"cited_frames":[179,179]}', frames, "chef's knife")
        self.assertEqual(r["cited_frame_ids"], [])
        self.assertTrue(r["citation_parse_ok"])
        self.assertEqual(r["grounding_status"], "unverified")

    def test_malformed_does_not_overwrite_answer(self):
        r = parse("chopstick", frames, "chef's knife")
        self.assertEqual(r["answer"], "chef's knife")
        self.assertFalse(r["citation_parse_ok"])

    def test_reject_boolean_and_string(self):
        r = parse('{"cited_frames":[true,"2",3,0]}', frames, "answer")
        self.assertEqual(r["cited_frame_ids"], [6])

if __name__ == "__main__":
    unittest.main()
