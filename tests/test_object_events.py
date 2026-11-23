import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.api.object_query import ObjectQueryRequest, execute_event_query
from src.reasoning.object_events import (
    MAX_FOLLOWUPS, MAX_INITIAL_WINDOWS, Window, decode_window_frames, investigate,
    parse_review, propose_windows,
)


def frame(timestamp, frame_id):
    return {"timestamp": timestamp, "frame_id": frame_id, "evidence_role": "event_context"}


class ObjectEventTests(unittest.TestCase):
    def test_proposals_cover_clip_boundaries_and_include_track_change(self):
        rows = [SimpleNamespace(timestamp=5., box=(0, 0, 10, 10)),
                SimpleNamespace(timestamp=5.2, box=(20, 0, 30, 10))]
        windows = propose_windows([SimpleNamespace(observations=rows)], [0], [], 20.)
        self.assertEqual(windows[0].start, 0.)
        self.assertEqual(windows[-1].end, 20.)
        self.assertTrue(any(window.cue == "box_change" for window in windows))
        self.assertLessEqual(len(windows), MAX_INITIAL_WINDOWS)

    def test_parser_requires_visible_references_for_actions(self):
        frames = [frame(4.3, "E129"), frame(4.8, "E144")]
        raw = json.dumps({"events": [
            {"type": "picked_up", "description": "A hand lifts the toy.", "frames": [2, 1]},
            {"type": "set_down", "description": "Unsupported", "frames": [1, 3]},
            {"type": "carried", "description": "Only one frame", "frames": [1, 1]},
        ], "inspect": {"frame": 2, "direction": "after"}})
        events, request, parsed = parse_review(raw, frames)
        self.assertTrue(parsed)
        self.assertEqual(request, (4.8, "after"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["frame_ids"], ["E129", "E144"])
        self.assertEqual((events[0]["start"], events[0]["end"]), (4.3, 4.8))
        self.assertEqual(parse_review("A guess.", frames), ([], None, False))

    def test_format_failure_retries_and_keeps_raw_outputs(self):
        req = ObjectQueryRequest(question="What happens?", object_phrase="mug", review_mode="events")
        invalid = '{"events":[{"type":"state","description":"A mug.","frames":[99]}]}'
        valid = 'Review: ```json\n{"events":[{"type":"state","description":"A mug is visible.","frames":[1]}]}\n```'
        outputs = iter([invalid, valid, "yes"])
        vlm = SimpleNamespace(answer_video_frames=lambda *a, **kw: next(outputs))
        events, reviews, _ = investigate(
            vlm, req, None, None, [], [], [], 1.,
            decode=lambda *args: [frame(.5, "E15")],
        )
        self.assertEqual(events[0]["frame_ids"], ["E15"])
        self.assertEqual(reviews[0]["raw_responses"], [invalid, valid])
        self.assertTrue(reviews[0]["parsed"])

    def test_uncited_proposal_requires_a_separate_citation_check(self):
        req = ObjectQueryRequest(question="What happens?", object_phrase="toy car",
                                 review_mode="events")
        raw = '[{"type":"moved_on_surface","description":"The toy car rolls."}]'
        frames = [frame(0., "E0"), frame(1., "E30")]
        self.assertEqual(parse_review(raw, frames), ([], None, False))

        for response, expected, parsed in (("[1,2]", ["E0", "E30"], True),
                                            ("[]", [], True), ("[1,99]", [], False),
                                            ("[1]", [], False)):
            with self.subTest(response=response):
                outputs = iter([raw, response, "yes"])

                def review(frames, prompt, **kwargs):
                    output = next(outputs)
                    if output == response:
                        self.assertIn("Target object: toy car", prompt)
                        self.assertIn("The toy car rolls", prompt)
                    return output

                events, reviews, _ = investigate(
                    SimpleNamespace(answer_video_frames=review), req, None, None,
                    [], [], [], 1., decode=lambda *args: frames,
                )
                self.assertEqual([fid for event in events for fid in event["frame_ids"]], expected)
                self.assertEqual(reviews[0]["citation_response"], response)
                self.assertEqual(reviews[0]["parsed"], parsed)

    def test_pickup_is_rechecked_even_with_valid_initial_citations(self):
        req = ObjectQueryRequest(question="What happens?", object_phrase="yellow car")
        raw = '{"events":[{"type":"picked_up","description":"A hand lifts the car.","frames":[1,2]}]}'
        frames = [frame(34., "E1020"), frame(34.4, "E1032")]
        for response, types in (('["touch",1,2]', ["touched"]), ('["lift",1,2]', ["picked_up"]),
                                ('["lift",2,1]', []), ('["lift",1,9]', []), ('[]', [])):
            with self.subTest(response=response):
                outputs = iter([raw, response, "yes"])
                vlm = SimpleNamespace(answer_video_frames=lambda *a, **kw: next(outputs))
                events, reviews, _ = investigate(
                    vlm, req, None, None, [], [], [], 1., decode=lambda *args: frames,
                )
                self.assertEqual([event["type"] for event in events], types)
                self.assertEqual(reviews[0]["pickup_response"], response)

    def test_blue_car_review_rejects_yellow_car_and_checks_cited_frames(self):
        req = ObjectQueryRequest(question="What happened to the blue car?", object_phrase="blue toy car")
        frames = [frame(33.83, "E1014"), frame(34., "E1019"), frame(34.17, "E1024")]
        frames[1]["evidence_role"] = "object_track"
        for description, verdict, accepted, reason in (
            ("A hand touches the yellow toy car.", None, False, "conflicting_object_color"),
            ("A hand touches the toy car.", "no", False, "visual_check"),
            ("A hand touches the blue toy car.", "Yes.", True, "visual_check"),
            ("A hand touches the blue toy car.", "Unclear", False, "invalid_response"),
        ):
            with self.subTest(description=description, verdict=verdict):
                raw = json.dumps({"events": [
                    {"type": "touched", "description": description, "frames": [1, 3]},
                ]})
                calls = []

                def review(images, prompt, **kwargs):
                    calls.append(images)
                    if len(calls) == 1:
                        return raw
                    self.assertIn("Requested object: blue toy car", prompt)
                    return verdict

                events, reviews, _ = investigate(
                    SimpleNamespace(answer_video_frames=review), req, None, None,
                    [], [], [], 1., decode=lambda *args: frames,
                )
                self.assertEqual(bool(events), accepted)
                self.assertEqual(reviews[0]["event_checks"][0]["reason"], reason)
                self.assertEqual(reviews[0]["parsed"], reason != "invalid_response")
                self.assertEqual(calls, [frames] if verdict is None else [frames, [frames[0], frames[2]]])

    def test_dense_decode_uses_source_time_and_exact_mask_matches(self):
        import numpy as np

        class Capture:
            def isOpened(self):
                return True

            def get(self, prop):
                return {1: 10., 2: 100}[prop]

            def set(self, prop, index):
                self.index = index

            def read(self):
                return True, np.zeros((16, 16, 3), dtype=np.uint8)

            def release(self):
                pass

        def write(path, image):
            Path(path).touch()
            return True

        cv2 = SimpleNamespace(VideoCapture=lambda _: Capture(), CAP_PROP_FPS=1,
                              CAP_PROP_FRAME_COUNT=2, CAP_PROP_POS_FRAMES=3, imwrite=write)
        with TemporaryDirectory() as temp, patch.dict(sys.modules, {"cv2": cv2}):
            with patch("src.reasoning.object_events.render_evidence",
                       side_effect=lambda obj, view, path: path.touch()):
                observed = {12: (object(), 4)}
                frames = decode_window_frames("sample.mp4", Window(1., 3.4, "coverage"),
                                              observed, temp, "mask")
            self.assertEqual((frames[0]["timestamp"], frames[-1]["timestamp"]), (1., 3.4))
            masks = [row for row in frames if row["evidence_role"] == "object_track"]
            self.assertEqual([(row["timestamp"], row["tracklet"]) for row in masks], [(1.2, 4)])
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in frames))

    def test_review_fetches_new_frames_for_followups_and_bounds_requests(self):
        windows = []
        req = ObjectQueryRequest(question="What happens to the mug?", object_phrase="mug",
                                 review_mode="events")

        def decode(video, window, observations, work, evidence_view):
            windows.append(window)
            return [frame(window.start, f"E{window.start}"), frame(window.end, f"E{window.end}")]

        def review(frames, question, **kwargs):
            self.assertIn("Target: mug", question)
            return json.dumps({"events": [], "inspect": {"frame": 2, "direction": "after"}})

        events, reviews, _ = investigate(SimpleNamespace(answer_video_frames=review), req,
                                         None, None, [], [], [], 20., decode=decode)
        self.assertEqual(events, [])
        self.assertEqual(sum(row["cue"] == "model_followup" for row in reviews), MAX_FOLLOWUPS)
        self.assertLessEqual(len(windows), MAX_INITIAL_WINDOWS + MAX_FOLLOWUPS)
        self.assertTrue(all(0 <= row.start <= row.end <= 20. for row in windows))

    def test_event_response_and_trace_keep_only_provided_citations(self):
        req = ObjectQueryRequest(question="What happened to the cup?", object_phrase="cup",
                                 review_mode="events")
        frames = [frame(2., "E60"), frame(2.5, "E75"), frame(3., "E90")]
        events = [{"type": "picked_up", "description": "The person lifts the cup.",
                   "start": 2., "end": 2.5, "frame_ids": ["E60", "E75"]}]
        reviews = [{"parsed": True}]
        trace = {"accepted_tracklets": [], "video_interval": [0, 7.]}
        with TemporaryDirectory() as temp:
            work = Path(temp)
            with patch("src.api.object_query.ground_and_track", return_value=([], trace, False)), \
                    patch("src.reasoning.object_events.investigate",
                          return_value=(events, reviews, frames)):
                pipeline = SimpleNamespace(encoder=None, vlm=object())
                result = execute_event_query("v", work / "test.mp4", req, pipeline, [], work, 7.)
            self.assertEqual(result["cited_frame_ids"], ["E60", "E75"])
            self.assertEqual(json.loads((work / "evidence.json").read_text())["answer"]["events"],
                             events)
