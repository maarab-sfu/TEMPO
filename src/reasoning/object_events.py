"""Inspect bounded windows of source frames around possible object interactions."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

from scripts.experiment_sam2_tracking import box_iou
from src.memory.segmented_timeline import COLORS, render_evidence

MAX_CLIP_SECONDS = 60
WINDOW_SECONDS = 2.4
MAX_INITIAL_WINDOWS = 6
MAX_FOLLOWUPS = 2
MAX_FORMAT_RETRIES = 2
MAX_CITATION_CHECKS = 4
FRAMES_PER_WINDOW = 8
EVENT_TYPES = {"picked_up", "carried", "set_down", "moved_on_surface", "touched", "state"}
FOLLOWUP_OFFSETS = {"before": -1.2, "around": 0, "after": 1.2}


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    cue: str


def centered_window(center, duration, cue):
    width = min(duration, WINDOW_SECONDS)
    start = min(max(0., center - width / 2), duration - width)
    return Window(round(start, 5), round(start + width, 5), cue)


def propose_windows(tracklets, accepted, semantic, duration):
    """Rank local changes and retain opening, middle, and closing coverage."""
    candidates = []
    for index in accepted:
        rows = tracklets[index].observations
        for row in (rows[0], rows[-1]):
            candidates.append((.45, row.timestamp, "visibility_boundary"))
        for before, after in zip(rows, rows[1:]):
            if 0 < after.timestamp - before.timestamp <= .6:
                change = 1 - box_iou(before.box, after.box)
                if change >= .25:
                    candidates.append((change, (before.timestamp + after.timestamp) / 2, "box_change"))
    candidates.extend((.35, frame["timestamp"], "retrieval_anchor") for frame in semantic
                      if frame["evidence_role"] == "semantic_anchor")
    windows = [centered_window(0., duration, "coverage")]
    closing = centered_window(duration, duration, "coverage")
    if closing != windows[0]:
        windows.append(closing)

    # Reserve one proposal per time band so repeated motion cannot consume the clip budget.
    band_count = MAX_INITIAL_WINDOWS - 2
    for index in range(band_count):
        start, stop = duration * index / band_count, duration * (index + 1) / band_count
        midpoint = (start + stop) / 2
        ranked = sorted(
            (item for item in candidates if start <= item[1] < stop),
            key=lambda item: item[0] - abs(item[1] - midpoint) / (stop - start),
            reverse=True,
        )
        band = [(time, cue) for _, time, cue in ranked]
        band.append((midpoint, "coverage"))
        for time, cue in band:
            window = centered_window(time, duration, cue)
            if all(abs(window.start - other.start) >= WINDOW_SECONDS * .8 for other in windows):
                windows.append(window)
                break
    return sorted(windows, key=lambda window: window.start)


def decode_window_frames(video, window, observations, work, evidence_view):
    """Decode the original source and highlight only exact-frame mask matches."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or count < 1:
            raise RuntimeError("Cannot read clip frames")
        last = int(count) - 1
        start = min(last, max(0, math.ceil(window.start * fps - .001)))
        end = min(last, max(start, math.floor(window.end * fps + .001)))
        available = sorted(index for index in observations if start <= index <= end)
        indices = set()
        for i in range(FRAMES_PER_WINDOW):
            target = round(start + (end - start) * i / (FRAMES_PER_WINDOW - 1))
            if 0 < i < FRAMES_PER_WINDOW - 1 and available:
                nearest = min(available, key=lambda index: abs(index - target))
                if abs(nearest - target) <= .12 * fps:
                    target = nearest
            indices.add(target)
        for index in sorted(indices):
            path = Path(work) / f"event_{index:06d}.jpg"
            row = {"frame_id": f"E{index}", "timestamp": round(index / fps, 5),
                   "image_path": str(path), "evidence_role": "event_context"}
            if index in observations:
                observation, tracklet = observations[index]
                row.update(evidence_role="object_track", tracklet=tracklet)
                if not path.is_file():
                    render_evidence(observation, evidence_view, path)
            elif not path.is_file():
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                scale = min(1., 640 / max(height, width))
                if scale < 1:
                    image = cv2.resize(image, (max(1, round(width * scale)),
                                               max(1, round(height * scale))))
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError("Could not save event evidence image")
            frames.append(row)
    finally:
        capture.release()
    return frames


def review_payload(raw):
    """Find a review object or a bare array of proposed events."""
    decoder = json.JSONDecoder()
    for start in re.finditer(r"[{\[]", raw):
        try:
            candidate, _ = decoder.raw_decode(raw, start.start())
        except ValueError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("events"), list):
            return candidate
        if isinstance(candidate, list) and candidate:
            if all(isinstance(item, dict) and "type" in item for item in candidate):
                return {"events": candidate}
    return None


def event_claim(item):
    if not isinstance(item, dict):
        return None
    kind, description = item.get("type"), item.get("description")
    if (isinstance(kind, str) and kind in EVENT_TYPES
            and isinstance(description, str) and description.strip()):
        return kind, " ".join(description.split())[:240]
    return None


def cited_event(claim, numbers, frames):
    kind, description = claim
    if not isinstance(numbers, list) or any(
        type(value) is not int or not 1 <= value <= len(frames) for value in numbers
    ):
        return None
    positions = sorted(set(numbers))
    if len(positions) < (1 if kind == "state" else 2):
        return None
    cited = [frames[pos - 1] for pos in positions]
    return {"type": kind, "description": description, "start": cited[0]["timestamp"],
            "end": cited[-1]["timestamp"], "frame_ids": [frame["frame_id"] for frame in cited]}


def parse_review(raw, frames):
    """Map model frame numbers to real evidence; unsupported actions are omitted."""
    payload = review_payload(raw)
    if payload is None:
        return [], None, False

    events = []
    for item in payload["events"][:3]:
        claim = event_claim(item)
        if claim:
            event = cited_event(claim, item.get("frames"), frames)
            if event:
                events.append(event)

    followup = payload.get("inspect")
    if isinstance(followup, dict):
        number, direction = followup.get("frame"), followup.get("direction")
        if (type(number) is int and 1 <= number <= len(frames)
                and isinstance(direction, str) and direction in FOLLOWUP_OFFSETS):
            return events, (frames[number - 1]["timestamp"], direction), True
    return events, None, bool(events) or not payload["events"]


def review_prompt(req, frames):
    highlighted = [str(i) for i, frame in enumerate(frames, 1)
                   if frame["evidence_role"] == "object_track"]
    views = {
        "full": "Images are unmarked.",
        "box": "Magenta boxes mark candidate matches.",
        "mask": "Magenta outlines and side close-ups mark candidate matches.",
    }
    return (
        f"Target: {req.object_phrase}. User question: {req.question}\n"
        f"{views[req.evidence_view]} Tracked frame numbers: {', '.join(highlighted) or 'none'}. "
        "Track matches can be wrong. Use the visible attributes and surrounding scene to "
        "confirm the target before describing it. Read the frames in timestamp order. "
        "A different instance of the same object type does not answer the question. "
        "Report only relevant directly visible actions or states. A pickup or set-down needs "
        "visible interaction and a change between support and hand. If a hand only taps or "
        "touches the object while it stays supported, report touched. A position change alone "
        "does not establish what moved it. If a transition is unclear, report the visible state "
        "or request a neighboring window. Use 1-based frame numbers, not timestamps. "
        "Cite at least two distinct frames for an action and at least one for a state. "
        "Write a short factual description for each event, without guessing intent. "
        'Return only JSON such as {"events":[{"type":"state","description":"The object rests on a surface.",'
        '"frames":[1]}],"inspect":null}. Allowed event types: picked_up, carried, set_down, '
        "moved_on_surface, touched, state. If more frames would help, set inspect to "
        '{"frame":2,"direction":"before"} using before, around, or after. '
        'If the evidence is insufficient use {"events":[],"inspect":null}.'
    )


def check_missing_citations(vlm, req, raw, frames):
    """Ask the model to verify one uncited claim against the actual images."""
    payload = review_payload(raw) or {}
    for item in payload.get("events", [])[:3]:
        claim = event_claim(item)
        if claim and claim[0] != "picked_up" and item.get("frames") in (None, []):
            break
    else:
        return None, None, False

    kind, description = claim
    minimum = "one frame" if kind == "state" else "two distinct frames"
    response = vlm.answer_video_frames(
        frames,
        f"Target object: {req.object_phrase}. Proposed {kind} claim: {description}\n"
        "Check whether the supplied images directly show this exact claim about the target object. "
        "Reject claims about a different object, a person alone, or movement that is merely implied. "
        "For an action, the cited sequence must show the action, including contact when relevant. "
        f"Return only a JSON array of 1-based frame numbers. Cite at least {minimum} "
        "if the claim is supported, or return [] if it is unsupported or unclear.",
        max_new_tokens=64,
    )
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response.strip())
    try:
        numbers = json.loads(cleaned)
    except ValueError:
        return None, response, False
    event = cited_event(claim, numbers, frames)
    return event, response, bool(event) or numbers == []


def check_pickup(vlm, req, frames):
    """Distinguish a lift from contact while the object stays supported."""
    raw = vlm.answer_video_frames(
        frames,
        f"Target object: {req.object_phrase}. Look only at the target, its support, and the hand. "
        "Choose lift only if an earlier image clearly shows the object resting on a support "
        "and a later image clearly shows the object off that support and held by the hand. "
        "Choose touch if the hand visibly contacts the object while it stays on its support. "
        "A hand nearby, a changing camera angle, or an obscured support is insufficient. "
        'Return only one JSON array: ["lift", resting_frame, held_frame], '
        '["touch", contact_frame, supported_frame], or [] for unclear or no contact. '
        "Replace frame names with 1-based image numbers.",
        max_new_tokens=64,
    )
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip())
    try:
        result = json.loads(cleaned)
    except ValueError:
        return None, raw, False
    if result == []:
        return None, raw, True
    if not isinstance(result, list) or len(result) != 3:
        return None, raw, False
    kind, first, second = result
    if type(first) is not int or type(second) is not int:
        return None, raw, False
    if kind == "lift" and first < second:
        claim = ("picked_up", f"A hand lifts the {req.object_phrase} off its support.")
    elif kind == "touch":
        claim = ("touched", f"A hand touches the {req.object_phrase} while it stays on its support.")
    else:
        return None, raw, False
    event = cited_event(claim, [first, second], frames)
    return event, raw, bool(event)


def verify_event(vlm, req, event, frames_by_id):
    """Confirm that the cited images show this claim about the requested instance."""
    target = req.object_phrase.lower().split()
    colors = set(COLORS) | {"black", "white", "gray", "brown"}
    description = event["description"]
    words = re.findall(r"[a-z]+", description.lower().replace("grey", "gray"))
    # Reject an unambiguous color substitution such as blue toy car → yellow toy car.
    target_color = target[0].replace("grey", "gray") if target else ""
    if len(target) > 1 and target_color in colors and target_color not in words:
        for index, word in enumerate(words):
            if word not in colors:
                continue
            if words[index + 1:index + 2] != target[-1:] and words[index + 1:index + len(target)] != target[1:]:
                continue
            return {"accepted": False, "reason": "conflicting_object_color", "response": None}

    cited = [frames_by_id[frame_id] for frame_id in event["frame_ids"]]
    raw = vlm.answer_video_frames(
        cited,
        f"Requested object: {req.object_phrase}. Proposed event type: {event['type']}. "
        f"Proposed description: {description}\n"
        "Check the cited images. Is this exact action or state visibly supported for the "
        "requested object, including its stated attributes? If the relevant hand contact or "
        "movement is with a different item, the requested item is merely nearby, or the "
        "evidence is unclear, answer no. Ignore any highlight on the wrong object. "
        "Reply with one word: yes or no.",
        max_new_tokens=8,
    )
    verdict = raw.strip().lower().rstrip(".")
    return {"accepted": verdict == "yes",
            "reason": "visual_check" if verdict in {"yes", "no"} else "invalid_response",
            "response": raw}


def investigate(vlm, req, video, work, tracklets, accepted, semantic, duration,
                *, decode=decode_window_frames):
    """Review proposals, then honor up to two valid requests for additional evidence."""
    observations = {}
    for index in accepted:
        for row in tracklets[index].observations:
            observations.setdefault(row.frame_index, (row, index))
    windows = propose_windows(tracklets, accepted, semantic, duration)
    events, reviews, used = [], [], {}
    followups = 0
    retries = 0
    citation_checks = 0
    for window in windows:
        frames = decode(video, window, observations, work, req.evidence_view)
        found, request, parsed = [], None, False
        raw_responses = []
        citation_response = None
        pickup_response = None
        event_checks = []
        if frames:
            raw = vlm.answer_video_frames(frames, review_prompt(req, frames), max_new_tokens=384)
            raw_responses.append(raw)
            found, request, parsed = parse_review(raw, frames)
            if not found and citation_checks < MAX_CITATION_CHECKS:
                event, citation_response, checked = check_missing_citations(vlm, req, raw, frames)
                if citation_response is not None:
                    citation_checks += 1
                    parsed = checked
                    if event:
                        found = [event]
            if not parsed and citation_response is None and retries < MAX_FORMAT_RETRIES:
                retries += 1
                raw = vlm.answer_video_frames(
                    frames,
                    review_prompt(req, frames) + "\nGive one strongest supported event or state. "
                    "Return one short JSON object with integer frame numbers and inspect set to null.",
                    max_new_tokens=384,
                )
                raw_responses.append(raw)
                found, request, parsed = parse_review(raw, frames)
            pickup = any(
                isinstance(item, dict) and item.get("type") == "picked_up"
                for reply in raw_responses
                for item in (review_payload(reply) or {}).get("events", [])[:3]
            )
            if pickup:
                found = [event for event in found if event["type"] != "picked_up"]
                event, pickup_response, checked = check_pickup(vlm, req, frames)
                parsed = bool(found) or checked
                if event:
                    found.append(event)
            frames_by_id = {frame["frame_id"]: frame for frame in frames}
            event_checks = [{"event": event, **verify_event(vlm, req, event, frames_by_id)}
                            for event in found]
            found = [check["event"] for check in event_checks if check["accepted"]]
            if event_checks and not found and any(check["reason"] == "invalid_response"
                                                  for check in event_checks):
                parsed = False
        used.update((frame["frame_id"], frame) for frame in frames)
        reviews.append({"start": window.start, "end": window.end, "cue": window.cue,
                        "frame_ids": [frame["frame_id"] for frame in frames],
                        "parsed": parsed, "requested_followup": request, "events": found,
                        "raw_responses": raw_responses, "citation_response": citation_response,
                        "pickup_response": pickup_response, "event_checks": event_checks})
        events.extend(found)
        if request is None or followups >= MAX_FOLLOWUPS:
            continue
        time, direction = request
        additional = centered_window(time + FOLLOWUP_OFFSETS[direction], duration, "model_followup")
        if all(abs(additional.start - item.start) >= .4 for item in windows):
            windows.append(additional)
            followups += 1

    # Overlapping reviews can describe the same action. Keep one with the most evidence.
    unique = []
    for event in sorted(events, key=lambda row: (-len(row["frame_ids"]), row["start"])):
        if not any(event["type"] == other["type"]
                   and (event["type"] != "state" or event["description"] == other["description"])
                   and max(event["start"], other["start"]) <= min(event["end"], other["end"])
                   for other in unique):
            unique.append(event)
    return sorted(unique, key=lambda row: (row["start"], row["end"])), reviews, sorted(
        used.values(), key=lambda frame: frame["timestamp"],
    )
