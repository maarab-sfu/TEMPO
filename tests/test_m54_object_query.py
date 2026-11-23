import json
import sys
from threading import Lock
from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.object_query import (
    ObjectQueryRequest, answer_by_interval, boundary_context, build_evidence, evidence_question,
    ground_and_track, router, video_source,
)
from src.memory.segmented_timeline import (
    Observation, Tracklet, associate, choose_seeds, color_distribution, grounding_times,
    masked_crop, render_evidence, visible_runs,
)


def proposal(time, frame, box, score):
    return {"timestamp": time, "frame_index": frame, "box": box, "score": score,
            "area_fraction": .05}


def observation(time, frame, color, box=(4, 4, 12, 12)):
    image = Image.new("RGB", (32, 32), (20, 120, 20))
    image.paste(color, box)
    mask = np.zeros((32, 32), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return Observation(frame, time, image, mask, box)


def track(seed, rows, rank):
    return Tracklet(seed, rows, np.array([1., 0.]), color_distribution(rows[0]), rank)


class ObjectQueryTests(unittest.TestCase):
    def test_whole_video_keeps_context_outside_accepted_tracks(self):
        catalog = [{"timestamp": t, "frame_id": i} for i, t in enumerate(range(0, 57, 2))]
        question = "What happens to the car from the beginning to the end?"
        frames = boundary_context(catalog, question, 18.85, 42.64)
        self.assertEqual([frame["timestamp"] for frame in frames], [0, 4, 56])
        self.assertTrue(all(frame["evidence_role"] == "boundary_context" for frame in frames))
        self.assertEqual(boundary_context(catalog, "Which way did it move?", 18.85, 42.64), [])

    def test_interval_replay_keeps_late_evidence_and_answer(self):
        catalog = [{"timestamp": t, "frame_id": i} for i, t in enumerate(range(0, 57, 2))]
        tracked = [{"timestamp": t, "frame_id": f"O{i}", "evidence_role": "object_track"}
                   for i, t in enumerate([18.85, 23.06, 23.26, 33.60, 38.04, 42.64])]
        req = ObjectQueryRequest(question="Follow the yellow car from beginning to end.",
                                 object_phrase="yellow car")
        seen = []

        def review(frames, question):
            self.assertLessEqual(len(frames), 8)
            seen.append([frame["timestamp"] for frame in frames])
            return {"answer": f"Part {len(seen)} observation",
                    "cited_frame_ids": [frames[-1]["frame_id"], "outside"],
                    "citation_parse_ok": True}

        answer, frames = answer_by_interval(
            SimpleNamespace(answer_video_frames_grounded=review), req, tracked, catalog,
            {"video_interval": [0, 56.02]},
        )
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0][0], 0)
        self.assertEqual(seen[-1][-1], 56)
        self.assertTrue(any(48 <= time < 56 for time in seen[-1]))
        self.assertIn("Part 3 observation", answer["answer"])
        self.assertNotIn("outside", answer["cited_frame_ids"])
        self.assertEqual(frames[-1]["timestamp"], 56)

    def test_request_bounds(self):
        with self.assertRaises(ValidationError):
            ObjectQueryRequest(question="x", object_phrase="car", evidence_budget=9)

    def test_invalid_video_id(self):
        with self.assertRaises(ValueError):
            video_source("../foo")

    def test_later_yellow_car_proposal_survives_seed_selection(self):
        early = proposal(21.98866, 659, [151, 148, 299, 231], .925)
        late = proposal(34.00067, 1019, [198, 234, 335, 328], .831)
        competing = proposal(34.00067, 1019, [5, 10, 30, 40], .4)
        seeds = choose_seeds([early, late, competing], 57)
        self.assertEqual(seeds, [early, late, competing])
        times = grounding_times([{"timestamp": 21.989}], 57)
        self.assertEqual((times[0], times[-1]), (0, 57))
        self.assertIn(21.989, times)

    def test_blue_shoe_cannot_anchor_blue_car_evidence(self):
        time = 34.00067
        yellow_car = proposal(time, 1019, [198, 234, 335, 328], .572)
        blue_shoe = proposal(time, 1019, [419, 195, 458, 241], .401)
        blue_car = proposal(time, 1019, [399, 340, 444, 359], .361)
        seeds = [yellow_car, blue_shoe, blue_car]
        tracks = [
            track(seed, [observation(time, 1019, color)], rank)
            for seed, color, rank in zip(seeds, [(255, 255, 0), (0, 0, 255), (0, 0, 255)],
                                         [.419, .887, .6])
        ]
        replies = iter(["No.", "no", "Yes"])

        def detect(*args):
            Path(args[-1]).write_text(json.dumps({"candidates": seeds}))

        def track_seeds(video, candidates, *args):
            self.assertEqual(candidates, seeds)
            return tracks, []

        def verify(path, prompt, **kwargs):
            self.assertTrue(path.is_file())
            self.assertIn("blue toy car", prompt)
            return next(replies)

        req = ObjectQueryRequest(question="What happened to it?", object_phrase="blue toy car")
        with TemporaryDirectory() as d, patch("src.api.object_query.run_step", side_effect=detect), \
                patch("src.memory.segmented_timeline.track_windows", side_effect=track_seeds):
            _, trace, _ = ground_and_track(
                Path("car.mp4"), req, None, [], Path(d), 56.02,
                load_vlm=lambda: SimpleNamespace(answer_image=verify),
            )
        self.assertEqual(trace["accepted_tracklets"], [2])
        self.assertEqual([check["accepted"] for check in trace["tracklet_checks"]],
                         [False, False, True])

    def test_missed_mask_splits_runs(self):
        first = observation(0., 0, (255, 255, 0))
        frame = np.asarray(first.image)[:, :, ::-1]
        runs = visible_runs([frame] * 5, [0, 1, 2, 3, 4],
                            {0: first.mask, 1: first.mask, 3: first.mask, 4: first.mask},
                            5, 0)
        self.assertEqual([[o.frame_index for o in rows] for _, rows in runs], [[0, 1], [3, 4]])
        self.assertEqual(runs[0][0].box, first.box)

    def test_similar_simultaneous_objects_are_ambiguous(self):
        yellow = (255, 255, 0)
        root = track({}, [observation(20., 600, yellow)], 1.)
        late = track({}, [observation(34., 1020, yellow)], .8)
        competitor = track({}, [observation(34., 1020, yellow, (18, 18, 26, 26))], .7)
        selected, decisions = associate([root, late, competitor])
        self.assertEqual(selected, [0])
        self.assertEqual([d["decision"] for d in decisions], ["ambiguous", "ambiguous"])

    def test_mask_crop_excludes_background_and_highlight_preserves_scene(self):
        sample = observation(0., 0, (255, 255, 0))
        sample.mask[4, 4] = False
        self.assertEqual(masked_crop(sample).getpixel((0, 0)), (127, 127, 127))
        self.assertAlmostEqual(color_distribution(sample)[2], 1.)
        with TemporaryDirectory() as d:
            path = Path(d) / "evidence.jpg"
            render_evidence(sample, "mask", path)
            with Image.open(path) as rendered:
                self.assertGreater(rendered.width, sample.image.width)
                self.assertEqual(rendered.height, sample.image.height)

    def test_evidence_budget_and_prompt_across_windows(self):
        yellow = (255, 255, 0)
        seed = proposal(21.98866, 659, [4, 4, 12, 12], .925)
        later = proposal(34.00067, 1019, [4, 4, 12, 12], .831)
        tracklets = [
            track(seed, [observation(18.78545, 563, yellow), observation(19.38605, 581, yellow),
                         observation(21.38805, 641, yellow), observation(23.18986, 695, yellow)], 1.),
            track(later, [observation(34.00067, 1019, yellow), observation(35., 1049, yellow)], .8),
            track(later, [observation(34.00067, 1019, (0, 0, 255), (18, 18, 26, 26))], .7),
        ]
        timestamps = [19.987, 21.989, 23.991, 31.999, 34.001, 36.003]
        semantic = [{"frame_id": i, "timestamp": t, "anchor_rank": None,
                     "evidence_role": "temporal_context"} for i, t in enumerate(timestamps)]
        req = ObjectQueryRequest(question="What happened to the yellow car throughout?",
                                 object_phrase="yellow toy car")

        def stub_detector(*args):
            Path(args[-1]).write_text(json.dumps({"candidates": [seed, later]}))

        with TemporaryDirectory() as d, patch("src.api.object_query.source_end", return_value=57), \
                patch("src.api.object_query.run_step", side_effect=stub_detector), \
                patch("src.memory.segmented_timeline.track_windows", return_value=(tracklets, [])):
            work = Path(d)
            frames, added, trace, status = build_evidence(work / "car.mp4", req, None, semantic, work)
            self.assertEqual(status, "object_evidence_found")
            self.assertEqual(len(frames), 8)
            self.assertEqual(len(added), 6)
            self.assertIn(23.18986, [f["timestamp"] for f in added])
            self.assertIn(35., [f["timestamp"] for f in added])
            self.assertEqual(trace["accepted_tracklets"], [0, 1])
            self.assertIn("tracklet 1: 34.00–35.00 s", evidence_question(req, frames, trace))
            self.assertTrue(all(Path(f["image_path"]).is_file() for f in added))
            json.dumps(trace)

    def test_route_passes_coverage_to_vlm_and_records_citations(self):
        app = FastAPI()
        app.include_router(router)
        frame = {"frame_id": "O12", "timestamp": 1., "tracklet": 0,
                 "evidence_role": "object_track"}
        trace = {"accepted_tracklets": [0], "video_interval": [0, 10],
                 "tracklets": [{"observations": [{"timestamp": 1.}]}]}

        def generate_answer(frames, question):
            self.assertEqual(frames, [frame])
            self.assertIn("The video runs from 0 to 10.00 s", question)
            self.assertIn("tracklet 0: 1.00–1.00 s", question)
            return {"answer": "The later location is unclear.", "cited_frame_ids": ["O12"],
                    "grounding_status": "cited", "citation_parse_ok": True}

        pipeline = SimpleNamespace(
            encoder=SimpleNamespace(encode_text=lambda _: []),
            vlm=SimpleNamespace(answer_video_frames_grounded=generate_answer),
        )
        modules = {
            "src.api.main": SimpleNamespace(_query_lock=Lock(), get_pipeline=lambda: pipeline),
            "src.storage.index_store": SimpleNamespace(IndexStore=lambda _: SimpleNamespace(
                load=lambda: (SimpleNamespace(metadata=[]), {}))),
            "src.retrieval.temporal_retriever": SimpleNamespace(TemporalRetriever=lambda _:
                SimpleNamespace(search_with_context=lambda *args, **kwargs: [frame])),
        }
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "clip").mkdir()
            (root / "clip" / "manifest.json").touch()
            with patch.dict(sys.modules, modules), patch("src.api.object_query.INDEX", root), \
                    patch("src.api.object_query.OUT", root / "outputs"), \
                    patch("src.api.object_query.video_source", return_value=root / "clip.mp4"), \
                    patch("src.api.object_query.build_evidence",
                          return_value=([frame], [frame], trace, "object_evidence_found")):
                with TestClient(app) as client:
                    response = client.post("/api/videos/clip/query-object", json={
                        "question": "Where did it go?", "object_phrase": "yellow car",
                    })
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["cited_frame_ids"], ["O12"])
            saved = root / "outputs" / "clip" / data["run_id"] / "evidence.json"
            self.assertEqual(json.loads(saved.read_text())["answer"]["cited_frame_ids"], ["O12"])


if __name__ == "__main__":
    unittest.main()
