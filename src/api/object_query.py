"""Answer object-focused queries using a segmented timeline."""
from __future__ import annotations
import json
import math
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'data' / 'indexes'
UPLOAD = ROOT / 'data' / 'uploads'
OUT = ROOT / 'outputs' / 'segmented_object_queries'

class ObjectQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    object_phrase: str = Field(min_length=1, max_length=120)
    top_k: int = Field(default=2, ge=1, le=5)
    evidence_budget: int = Field(default=6, ge=1, le=6)
    evidence_view: Literal["full", "box", "mask"] = "mask"
    review_mode: Literal["summary", "events"] = "summary"
    before: float = Field(default=4, ge=0, le=12)
    after: float = Field(default=4, ge=0, le=12)


def video_source(video_id: str) -> Path:
    if not video_id or any(not(c.isalnum() or c in '-_') for c in video_id):
        raise ValueError('Invalid video ID')
    matches = [p for p in UPLOAD.glob(video_id + '.*') if p.suffix.lower() in {'.mp4','.mov','.avi','.mkv','.webm'}]
    if len(matches) != 1:
        raise FileNotFoundError('Original uploaded video unavailable; re-upload to use object-aware mode')
    return matches[0]


def run_step(module: str, *args: str) -> None:
    completed = subprocess.run([sys.executable, '-m', module, *map(str,args)], cwd=ROOT,
                               capture_output=True, text=True, timeout=900, check=False)
    if completed.returncode:
        raise RuntimeError(f'{module} failed: {(completed.stderr or completed.stdout)[-1400:]}')


def source_end(video):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise RuntimeError("Video frame count or frame rate unavailable")
        return (count - 1) / fps
    finally:
        capture.release()


def whole_video_question(question):
    return re.search(
        r"\b(?:beginning|start)\b.*\b(?:end|finish)\b|"
        r"\b(?:throughout|(?:whole|entire|full) (?:video|clip))\b",
        question, re.IGNORECASE,
    ) is not None


def boundary_context(catalog, question, first, last):
    """Keep scene views of gaps at the ends of a whole-video question."""
    if not catalog or not whole_video_question(question):
        return []
    ordered = sorted(catalog, key=lambda frame: frame["timestamp"])
    opening = [frame for frame in ordered if frame["timestamp"] < first - .35]
    closing = [frame for frame in ordered if frame["timestamp"] > last + .35]
    selected = []
    if opening:
        selected.append(opening[0])
        followups = [frame for frame in opening[1:] if frame["timestamp"] <= 6]
        if followups:
            selected.append(min(followups, key=lambda frame: abs(frame["timestamp"] - 4)))
    if closing:
        selected.append(closing[-1])
    return [{**frame, "evidence_role": "boundary_context",
             "anchor_rank": None, "similarity": None} for frame in selected]


def query_vlm(pipeline):
    if pipeline.vlm is None:
        from src.reasoning.vlm import VisionLanguageModel

        pipeline.vlm = VisionLanguageModel()
    return pipeline.vlm


def ground_and_track(video, req, encoder, semantic, work, end, *, load_vlm=None):
    from src.memory.segmented_timeline import (
        associate, choose_seeds, grounding_times, track_windows, verify_tracklets,
    )

    times = grounding_times(semantic, end)
    detected = work / "detections.json"
    run_step(
        "scripts.query_object_grounding", "--video", str(video), "--question", req.question,
        "--object", req.object_phrase, "--timestamps", ",".join(map(str, times)),
        "--output", str(detected),
    )
    detection_data = json.loads(detected.read_text(encoding="utf-8"))
    seeds = choose_seeds(detection_data["candidates"], end)
    tracklets, windows = track_windows(
        video, seeds, req.object_phrase, req.before, req.after, encoder, work
    ) if seeds else ([], [])
    checks = []
    candidates = None
    if tracklets and load_vlm is not None:
        checks = verify_tracklets(load_vlm(), req.object_phrase, tracklets, work)
        candidates = [check["tracklet"] for check in checks if check["accepted"]]
    accepted, links = associate(tracklets, candidates)
    records = [{"seed": tracklet.seed, "rank": tracklet.rank,
                "observations": [o.record() for o in tracklet.observations]}
               for tracklet in tracklets]
    trace = {"schema_version": 1, "video_interval": [0, end], "grounding_times": times,
             "windows": windows, "tracklets": records, "accepted_tracklets": accepted,
             "links": links, "tracklet_checks": checks, "evidence_view": req.evidence_view}
    return tracklets, trace, bool(seeds)


def build_evidence(video, req, encoder, semantic, work, *, catalog=(), load_vlm=None):
    from src.memory.segmented_timeline import render_evidence, select_observations

    tracklets, trace, seeded = ground_and_track(
        video, req, encoder, semantic, work, source_end(video), load_vlm=load_vlm,
    )
    accepted = trace["accepted_tracklets"]
    first = min((tracklets[i].observations[0].timestamp for i in accepted), default=math.inf)
    last = max((tracklets[i].observations[-1].timestamp for i in accepted), default=-math.inf)
    boundary = boundary_context(catalog, req.question, first, last)
    observations = select_observations(tracklets, accepted, min(req.evidence_budget, 8 - len(boundary)))
    object_frames = []
    for observation, tracklet in observations:
        path = work / f"object_{observation.frame_index:06d}.jpg"
        render_evidence(observation, req.evidence_view, path)
        object_frames.append({
            **observation.record(), "frame_id": f"O{observation.frame_index}",
            "image_path": str(path), "evidence_role": "object_track",
            "tracklet": tracklet, "anchor_rank": None, "similarity": None,
        })
    # Keep the requested timeline boundaries alongside the matched tracklets.
    context = sorted(semantic, key=lambda frame: (
        frame.get("anchor_rank") is None, frame.get("anchor_rank") or 0))
    decisions = [{"frame_id": frame["frame_id"], "decision": "selected_boundary"}
                 for frame in boundary]
    frames = object_frames + boundary
    boundary_ids = {frame["frame_id"] for frame in boundary}
    for frame in context:
        if frame["frame_id"] in boundary_ids:
            continue
        if any(abs(frame["timestamp"] - item["timestamp"]) < .35 for item in object_frames):
            decision = "covered_by_object_frame"
        elif len(frames) >= 8:
            decision = "image_budget"
        else:
            decision = "selected"
            frames.append(frame)
        decisions.append({"frame_id": frame["frame_id"], "decision": decision})
    trace.update(context_selection=decisions, frames=frames)
    status = ("object_evidence_found" if object_frames else
              "object_not_visible" if seeded else "object_not_found")
    return sorted(frames, key=lambda f: f["timestamp"]), object_frames, trace, status


def evidence_question(req, frames, trace):
    view = {
        "full": "Object evidence frames are unmarked scene views.",
        "box": "In object evidence frames a magenta box marks the candidate.",
        "mask": ("In object evidence frames a magenta outline marks the candidate. "
                 "The panel on the right is a close-up from the same timestamp."),
    }[req.evidence_view]
    labels = [
        f"Frame {i}: {frame['evidence_role']}"
        + (f", tracklet {frame['tracklet']}" if "tracklet" in frame else "")
        for i, frame in enumerate(frames, 1)
    ]
    spans = [
        f"tracklet {i}: {rows[0]['timestamp']:.2f}–{rows[-1]['timestamp']:.2f} s"
        for i in trace.get("accepted_tracklets", [])
        for rows in [trace["tracklets"][i]["observations"]]
    ]
    runtime = trace.get("video_interval")
    scope = f"The video runs from 0 to {runtime[1]:.2f} s. " if runtime else ""
    return (
        f"{req.question}\nTarget object: {req.object_phrase}. {view}\n"
        f"{scope}Sampled object intervals: {', '.join(spans) or 'none'}. "
        f"{'; '.join(labels)}.\n"
        "Describe visible actions and the starting and ending states when supported. "
        "Boundary and interval context show the unmarked scene. Use it to identify the requested "
        "object by its visible attributes and report any visible starting or ending state. "
        "Tracklets are candidate matches. Mention uncertainty about identity across gaps. "
        "Constrain claims to the frames that show them. State when a pickup, placement, "
        "destination, or intervening action cannot be determined. Cite the frame numbers "
        "that show the target and the event or resulting state."
    )


def answer_by_interval(vlm, req, frames, catalog, trace):
    """Review each part independently so a valid early event cannot crowd out later ones."""
    end = trace["video_interval"][1]
    parts = []
    used = {}
    cited = {}
    for index in range(3):
        start, stop = end * index / 3, end * (index + 1) / 3
        # Small overlap lets a transition at a boundary appear in both reviews.
        lower, upper = start - 1, stop + 1
        tracked = [frame for frame in frames if frame["evidence_role"] == "object_track"
                   and lower <= frame["timestamp"] <= upper]
        scene = sorted((frame for frame in catalog if lower <= frame["timestamp"] <= upper
                        and all(abs(frame["timestamp"] - obj["timestamp"]) >= .35
                                for obj in tracked)), key=lambda frame: frame["timestamp"])
        context_budget = 8 - len(tracked)
        # Use evenly spaced scene frames including both ends when possible.
        count = min(len(scene), context_budget)
        positions = [round(i * (len(scene) - 1) / (count - 1))
                     for i in range(count)] if count > 1 else list(range(count))
        context = [{**scene[i], "evidence_role": "interval_context",
                    "anchor_rank": None, "similarity": None} for i in positions]
        batch = sorted(tracked + context, key=lambda frame: frame["timestamp"])
        if batch:
            question = evidence_question(req, batch, trace) + (
                f"\nReview the {start:.1f}–{stop:.1f} s part of the video for this answer. "
                "Describe each distinct action involving the target and any resulting state. "
                "Identify who performs an action and what surface the object is on when visible. "
                "Use still frames to describe states. Infer a transition only when the visible "
                "sequence supports it, and qualify the inference. Mention relevant uncertainty."
            )
            result = vlm.answer_video_frames_grounded(frames=batch, question=question)
        else:
            result = {"answer": "No frames were available for this interval.",
                      "cited_frame_ids": [], "citation_parse_ok": False}
        batch_ids = {frame["frame_id"] for frame in batch}
        result["cited_frame_ids"] = [fid for fid in result["cited_frame_ids"] if fid in batch_ids]
        used.update((frame["frame_id"], frame) for frame in batch)
        cited.update(dict.fromkeys(result["cited_frame_ids"]))
        parts.append({**result, "interval": [start, stop],
                      "frame_ids": [frame["frame_id"] for frame in batch]})

    reviewed = sorted(used.values(), key=lambda frame: frame["timestamp"])
    answer = "\n".join(f"{start:.1f}–{stop:.1f} s: {part['answer']}"
                       for part in parts for start, stop in [part["interval"]])
    return {
        "answer": answer, "cited_frame_ids": [f["frame_id"] for f in reviewed if f["frame_id"] in cited],
        "grounding_status": "cited" if cited else "unverified",
        "citation_parse_ok": all(part["citation_parse_ok"] for part in parts),
        "interval_answers": parts,
    }, reviewed


def execute_event_query(video_id, video, req, pipeline, semantic, work, end):
    from src.reasoning.object_events import investigate

    tracklets, trace, seeded = ground_and_track(
        video, req, pipeline.encoder, semantic, work, end, load_vlm=lambda: query_vlm(pipeline),
    )
    trace_path = work / "evidence.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    events, reviews, frames = investigate(
        query_vlm(pipeline), req, video, work, tracklets, trace["accepted_tracklets"], semantic, end,
    )
    cited = {frame_id for event in events for frame_id in event["frame_ids"]}
    answer = "\n".join(
        f"{event['start']:.2f}–{event['end']:.2f} s: {event['description']}" for event in events
    )
    parsed_reviews = sum(review["parsed"] for review in reviews)
    if not answer and parsed_reviews < len(reviews):
        answer = (f"Event review returned no usable result. "
                  f"{len(reviews) - parsed_reviews} of {len(reviews)} windows did not yield a usable response.")
    if not answer:
        answer = "I could not establish an object interaction or state from the reviewed frames."
    if events and whole_video_question(req.question):
        answer += "\nThese are the events supported by the sampled windows. Other parts may contain actions."
    result = {
        "answer": answer, "events": events,
        "cited_frame_ids": [frame["frame_id"] for frame in frames if frame["frame_id"] in cited],
        "grounding_status": "cited" if cited else "unverified",
        "citation_parse_ok": all(review["parsed"] for review in reviews),
    }
    trace.update(reviews=reviews, frames=frames, answer=result)
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    object_frames = sum(frame["evidence_role"] == "object_track" for frame in frames)
    status = ("object_evidence_found" if object_frames else
              "object_not_visible" if seeded else "object_not_found")
    return {
        **result, "video_id": video_id, "question": req.question, "object_phrase": req.object_phrase,
        "object_status": status, "object_error": None, "object_evidence_count": object_frames,
        "frames": frames, "run_id": work.name,
        "limitations": ["Short windows sample the source video", "Cross-window identity can be uncertain",
                        "Event labels and their evidence are selected by the vision-language model"],
    }


def execute_object_query(video_id: str, req: ObjectQueryRequest, pipeline):
    from src.storage.index_store import IndexStore
    from src.retrieval.temporal_retriever import TemporalRetriever

    video = video_source(video_id)
    if req.review_mode == "events":
        from src.reasoning.object_events import MAX_CLIP_SECONDS

        end = source_end(video)
        if end > MAX_CLIP_SECONDS:
            raise ValueError(f"Event review supports clips up to {MAX_CLIP_SECONDS} seconds. "
                             "Trim the video or use Object summary.")
    idx = INDEX / video_id
    if not (idx / "manifest.json").is_file():
        raise FileNotFoundError("Index not found")
    retriever, _ = IndexStore(idx).load()
    semantic = TemporalRetriever(retriever).search_with_context(
        pipeline.encoder.encode_text([req.question]), top_k=req.top_k,
        context_before=2, context_after=2, max_frames=6,
    )
    if not semantic:
        raise RuntimeError("No semantic evidence")
    work = OUT / video_id / uuid.uuid4().hex[:12]
    work.mkdir(parents=True, exist_ok=False)
    if req.review_mode == "events":
        return execute_event_query(video_id, video, req, pipeline, semantic, work, end)
    diagnostic = None
    try:
        frames, added, trace, status = build_evidence(
            video, req, pipeline.encoder, semantic, work, catalog=retriever.metadata,
            load_vlm=lambda: query_vlm(pipeline),
        )
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        status = "object_pipeline_error"
        diagnostic = str(exc)
        frames, added = sorted(semantic, key=lambda f: f["timestamp"]), []
        trace = {"schema_version": 1, "error": diagnostic, "frames": frames}
    trace_path = work / "evidence.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    vlm = query_vlm(pipeline)
    if whole_video_question(req.question) and trace.get("video_interval", [0, 0])[1] > 0:
        answer, frames = answer_by_interval(vlm, req, frames, retriever.metadata, trace)
    else:
        answer = vlm.answer_video_frames_grounded(
            frames=frames, question=evidence_question(req, frames, trace),
        )
    trace["frames"] = frames
    trace["answer"] = answer
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return {
        **answer, "video_id": video_id, "question": req.question, "object_phrase": req.object_phrase,
        "object_status": status, "object_error": diagnostic, "object_evidence_count": len(added),
        "frames": frames, "run_id": work.name,
        "limitations": ["Object history is based on sampled windows",
                        "Cross-window matches are appearance hypotheses",
                        "Citations are selected by the answer model"],
    }


@router.post('/api/videos/{video_id}/query-object')
def query_object(video_id: str, request: ObjectQueryRequest):
    # Reuse production GPU lock and lazy pipeline; no concurrent model calls.
    from src.api.main import _query_lock, get_pipeline
    try:
        with _query_lock:
            return execute_object_query(video_id,request,get_pipeline())
    except (ValueError,FileNotFoundError) as exc:
        raise HTTPException(status_code=422,detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500,detail=f'Object-aware query failed: {exc}')
