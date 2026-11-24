import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.api.object_query import ObjectQueryRequest, execute_event_query
from src.reasoning.frame_labels import citation_token_prefix, parse_frame_labels
from src.reasoning.object_events import (
    COVERAGE_SECONDS, MAX_FOCUS_WINDOWS, Window, decode_window_frames, investigate,
    propose_windows, verified_sightings,
)


def frame(timestamp, index, *, tracked=False):
    return {"timestamp": timestamp, "frame_id": f"E{index}",
            "evidence_role": "object_track" if tracked else "event_context"}


class ObjectEventTests(unittest.TestCase):
    def test_covers_clip_and_adds_dense_windows_near_motion(self):
        rows = [SimpleNamespace(timestamp=21.25, box=(0, 0, 10, 10)),
                SimpleNamespace(timestamp=21.45, box=(20, 0, 30, 10))]
        windows = propose_windows([SimpleNamespace(observations=rows)], [0], [], 56.02269)
        coverage = [window for window in windows if window.cue == "coverage"]
        focused = [window for window in windows if window.cue != "coverage"]
        self.assertEqual(coverage[0].start, 0)
        self.assertEqual(coverage[-1].end, 56.02269)
        self.assertTrue(all(left.end == right.start for left, right in zip(coverage, coverage[1:])))
        self.assertTrue(all(window.end - window.start <= COVERAGE_SECONDS for window in coverage))
        self.assertLessEqual(len(focused), MAX_FOCUS_WINDOWS)
        self.assertTrue(any(window.cue == "box_change" and window.start <= 21.35 <= window.end
                            for window in focused))

    def test_literal_labels_do_not_turn_timestamps_into_citations(self):
        self.assertEqual(parse_frame_labels(" C, A, A. ", 3), [0, 2])
        self.assertEqual(parse_frame_labels('["A", "C"]', 3), [0, 2])
        self.assertEqual(parse_frame_labels("None.", 2), [])
        self.assertIsNone(parse_frame_labels("[19.05, 20.12]", 8))
        self.assertIsNone(parse_frame_labels("A D", 3))

    def test_constrained_decoder_allows_only_in_order_letters_or_none(self):
        import numpy as np

        def tokenize(options, *, add_special_tokens):
            self.assertFalse(add_special_tokens)
            return {"input_ids": [[ord(word[0]) for word in option.split()] for option in options]}

        next_tokens = citation_token_prefix(tokenize, "ABC", [0, 1], prompt_length=2)
        self.assertEqual(next_tokens(0, np.array([999, 999])), list(map(ord, "ABCN")))
        self.assertEqual(next_tokens(0, np.array([999, 999, ord("A")])), [0, 1, ord("B"), ord("C")])
        self.assertEqual(next_tokens(0, np.array([999, 999, ord("A"), ord("C")])), [0, 1])
        self.assertEqual(next_tokens(0, np.array([999, 999, ord("N")])), [0, 1])

    def test_wrong_car_is_rejected_and_state_cites_actual_track_frame(self):
        req = ObjectQueryRequest(question="What happens to the blue car?", object_phrase="blue toy car")
        frames = [frame(33.83, 1014), frame(34., 1019), frame(34.17, 1024, tracked=True)]
        responses = iter([
            "touched | A hand touches the yellow toy car.\n"
            "state | The blue toy car is partly visible at the bottom edge. |",
            "C", "A",
        ])
        calls = []
        def answer(images, prompt, **kwargs):
            calls.append((images, prompt, kwargs))
            return next(responses)

        events, trace, _ = investigate(
            SimpleNamespace(answer_video_frames=answer), req, None, None,
            [], [], [], 2., decode=lambda *args: frames,
        )
        self.assertEqual([event["frame_ids"] for event in events], [["E1024"]])
        self.assertEqual([check["status"] for check in trace[0]["claim_checks"]],
                         ["rejected_target", "supported"])
        self.assertIn("Requested object: blue toy car", calls[1][1])
        self.assertEqual(calls[1][2]["frame_labels"], "ABC")
        self.assertEqual(calls[2][0], [frames[2]])
        self.assertEqual(events[0]["description"], "The blue toy car is partly visible at the bottom edge.")
        self.assertEqual((events[0]["start"], events[0]["end"]), (34.17, 34.17))

    def test_rejects_reported_irrelevant_and_clip_wide_claims_before_citation(self):
        req = ObjectQueryRequest(question="What happened?", object_phrase="the blue toy car")
        frames = [frame(56.02, 1679)]
        for raw, statuses in (
            ("The boy holds his hands together near his mouth.", []),
            ("state | The boy holds his hands together near his mouth.", ["rejected_target"]),
            ("state | The blue toy car remains stationary throughout the video.", ["unsupported_scope"]),
            ("state | The blue toy car remains stationary on the ground near the child's feet.",
             ["persistence_labeled_as_state"]),
            ("state | The blue toy car is placed on the carpet. |", ["transition_labeled_as_state"]),
            ("touched | A hand touches the yellow toy car near the blue toy car.", ["rejected_target"]),
        ):
            with self.subTest(raw=raw):
                answer = iter([raw])
                events, reviews, _ = investigate(
                    SimpleNamespace(answer_video_frames=lambda *args, **kw: next(answer)),
                    req, None, None, [], [], [], 1., decode=lambda *args: frames,
                )
                self.assertEqual(events, [])
                self.assertEqual([check["status"] for check in reviews[0]["claim_checks"]], statuses)

    def test_action_requires_track_anchor_and_two_target_frames(self):
        req = ObjectQueryRequest(question="What happened?", object_phrase="blue toy car")
        frames = [frame(0., 0), frame(1., 30, tracked=True), frame(2., 60)]
        for presence, citation, expected in (
            ("B C", "A B", "supported"),
            ("B", None, "insufficient_target_frames"),
            ("NONE", None, "insufficient_target_frames"),
            ("A C", None, "missing_track_anchor"),
            ("A B C", "A C", "missing_track_anchor"),
            ("A B C", "A", "insufficient_support"),
        ):
            with self.subTest(presence=presence, citation=citation):
                outputs = ["set_down | The blue toy car is placed on the carpet.", presence]
                if citation is not None:
                    outputs.append(citation)
                responses = iter(outputs)
                events, reviews, _ = investigate(
                    SimpleNamespace(answer_video_frames=lambda *args, **kw: next(responses)),
                    req, None, None, [], [], [], 2., decode=lambda *args: frames,
                )
                self.assertEqual(reviews[0]["claim_checks"][0]["status"], expected)
                self.assertEqual([event["frame_ids"] for event in events],
                                 [["E30", "E60"]] if expected == "supported" else [])

    def test_local_state_is_rejected_when_target_is_absent(self):
        req = ObjectQueryRequest(question="Where is the car?", object_phrase="blue toy car")
        responses = iter(["state | The blue toy car rests by the child's feet.", "NONE"])
        events, reviews, _ = investigate(
            SimpleNamespace(answer_video_frames=lambda *args, **kw: next(responses)),
            req, None, None, [], [], [], 1., decode=lambda *args: [frame(56.02, 1679)],
        )
        self.assertEqual(events, [])
        self.assertEqual(reviews[0]["target_visible_frame_ids"], [])

    def test_untracked_final_sighting_cannot_support_state_even_if_model_selects_it(self):
        req = ObjectQueryRequest(question="Where is the car?", object_phrase="blue toy car")
        responses = iter(["state | The blue toy car rests by the child's feet.", "A"])
        events, reviews, _ = investigate(
            SimpleNamespace(answer_video_frames=lambda *args, **kw: next(responses)),
            req, None, None, [], [], [], 1., decode=lambda *args: [frame(56.02, 1679)],
        )
        self.assertEqual(events, [])
        self.assertEqual(reviews[0]["claim_checks"][0]["status"], "missing_track_anchor")

    def test_trajectory_motion_is_reviewed_even_when_narration_offers_no_event(self):
        req = ObjectQueryRequest(question="What happened?", object_phrase="the blue toy car")
        first = SimpleNamespace(frame_index=6, timestamp=.2, box=(10, 10, 30, 30))
        second = SimpleNamespace(frame_index=36, timestamp=1.2, box=(41, 10, 61, 30))
        tracks = [SimpleNamespace(observations=[first, second])]
        for verdict in ("A B", "NONE"):
            with self.subTest(verdict=verdict), TemporaryDirectory() as tmp, patch(
                "src.reasoning.object_events.render_evidence"
            ):
                responses = iter(["NONE", "A B", verdict])

                def answer(frames, prompt, **kwargs):
                    if "Proposed changed_position" in prompt:
                        self.assertIn("same fixed scene reference", prompt)
                        self.assertIn("camera movement", prompt)
                        self.assertEqual([f["tracklet"] for f in frames], [0, 0])
                    return next(responses)

                events, reviews, used = investigate(
                    SimpleNamespace(answer_video_frames=answer), req, None, Path(tmp), tracks,
                    [0], [], 2., decode=lambda *args: [frame(0, 0)],
                )
                self.assertEqual(reviews[-1]["cue"], "track_motion")
                self.assertEqual([f["frame_id"] for f in used], ["E0", "M6", "M36"])
                self.assertEqual([event["frame_ids"] for event in events],
                                 [["M6", "M36"]] if verdict == "A B" else [])

    def test_verified_sightings_use_only_the_checked_sample_frames(self):
        rows = [SimpleNamespace(frame_index=i, timestamp=i / 10) for i in range(5)]
        tracklets = [SimpleNamespace(observations=rows), SimpleNamespace(observations=rows)]
        checks = [{"tracklet": 0, "accepted": True, "frame_indices": [1, 3]},
                  {"tracklet": 1, "accepted": False, "frame_indices": [0, 2, 4]}]
        with TemporaryDirectory() as d, patch("src.reasoning.object_events.render_evidence") as render:
            frames = verified_sightings(tracklets, [0, 1], checks, Path(d), "mask")
            self.assertEqual([row["frame_id"] for row in frames], ["S1", "S3"])
            self.assertEqual([call.args[2].name for call in render.call_args_list],
                             ["sighting_000001.jpg", "sighting_000003.jpg"])

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
                observed = {12: (SimpleNamespace(frame_index=12, timestamp=1.2), 4)}
                frames = decode_window_frames("sample.mp4", Window(1., 3.4, "coverage"),
                                              observed, temp, "mask")
            self.assertEqual((frames[0]["timestamp"], frames[-1]["timestamp"]), (1., 3.4))
            masks = [row for row in frames if row["evidence_role"] == "object_track"]
            self.assertEqual([(row["timestamp"], row["tracklet"]) for row in masks], [(1.2, 4)])
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in frames))

    def test_api_returns_sightings_when_action_citation_fails(self):
        req = ObjectQueryRequest(question="What happened to the blue toy car?", object_phrase="blue toy car")
        seen = frame(20., 600, tracked=True)
        trace = {"accepted_tracklets": [], "video_interval": [0, 56.02]}
        review = {"cue": "coverage", "claim_checks": [{"status": "invalid_citation"}]}
        sighting = {**seen, "frame_id": "S600"}
        with TemporaryDirectory() as temp:
            work = Path(temp)
            with patch("src.api.object_query.ground_and_track", return_value=([], trace, True)), \
                    patch("src.reasoning.object_events.investigate", return_value=([], [review], [seen])), \
                    patch("src.reasoning.object_events.verified_sightings", return_value=[sighting]):
                pipeline = SimpleNamespace(encoder=None, vlm=object())
                result = execute_event_query("v", work / "test.mp4", req, pipeline, [], work, 56.02)
            self.assertEqual(result["grounding_status"], "sightings_only")
            self.assertEqual(result["cited_frame_ids"], [])
            self.assertEqual(result["sighting_frame_ids"], ["S600"])
            self.assertEqual(result["object_evidence_count"], 1)
            self.assertEqual(result["review_summary"]["citation_errors"], 1)
            self.assertIn("did not establish what happened", result["answer"])
            saved = json.loads((work / "evidence.json").read_text())
            self.assertEqual(saved["answer"]["sighting_frame_ids"], ["S600"])

    def test_scene_review_proceeds_when_tracking_fails(self):
        req = ObjectQueryRequest(question="What happened to the cup?", object_phrase="cup")
        frame_row = frame(0., 0)
        event = {"type": "state", "description": "The cup is on the counter.",
                 "start": 0., "end": 0., "frame_ids": ["E0"]}
        with TemporaryDirectory() as temp:
            work = Path(temp)
            with patch("src.api.object_query.ground_and_track", side_effect=RuntimeError("SAM unavailable")), \
                    patch("src.reasoning.object_events.investigate", return_value=([event], [], [frame_row])) as review:
                pipeline = SimpleNamespace(encoder=None, vlm=object())
                result = execute_event_query("v", work / "test.mp4", req, pipeline, [], work, 1.)
            self.assertEqual(review.call_args.args[4:6], ([], []))
            self.assertEqual(result["object_status"], "object_pipeline_error")
            self.assertEqual(result["grounding_status"], "cited")
            self.assertEqual(result["cited_frame_ids"], ["E0"])
