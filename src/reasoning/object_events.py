"""Inspect bounded windows of source frames around possible object interactions."""
from dataclasses import dataclass
from pathlib import Path
import math
import re

from scripts.experiment_sam2_tracking import box_iou
from src.memory.segmented_timeline import COLORS, render_evidence
from src.reasoning.frame_labels import FRAME_LABELS, parse_frame_labels

MAX_CLIP_SECONDS = 60
WINDOW_SECONDS = 2.4
COVERAGE_SECONDS = 8
MAX_FOCUS_WINDOWS = 4
MAX_CLAIMS_PER_WINDOW = 2
MAX_SIGHTING_FRAMES = 6
FRAMES_PER_WINDOW = 8
EVENT_TYPES = {"picked_up", "carried", "set_down", "moved_on_surface", "touched", "state"}
CLIP_WIDE = re.compile(
    r"\b(?:throughout|entire (?:video|clip)|whole (?:video|clip)|"
    r"(?:from (?:the )?)?(?:start|beginning) to (?:the )?end|the whole time|"
    r"always|never)\b", re.IGNORECASE,
)
TRANSITION_WORDS = re.compile(
    r"\b(?:plac(?:e[sd]?|ing)|set(?:s|ting)? down|pick(?:s|ed|ing)? up|"
    r"lift(?:s|ed|ing)?|put(?:s|ting)? (?:back|down|on)|"
    r"mov(?:e[sd]?|ing)|touch(?:es|ed|ing)?|dropp?(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
ACTION_EVIDENCE = {
    "picked_up": "Show the target on its support earlier, then held clear of that support.",
    "set_down": "Show the target held earlier, then released and resting on the stated support.",
    "carried": "Show the same target held by the same person across a visible change of location.",
    "moved_on_surface": "Show the same target changing position relative to a fixed reference on the surface.",
    "touched": "Show actual contact with the requested object and a second view of the same object.",
}


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
    count = max(1, math.ceil(duration / COVERAGE_SECONDS))
    boundaries = [round(duration * i / count, 5) for i in range(count + 1)]
    coverage = [Window(start, stop, "coverage") for start, stop in zip(boundaries, boundaries[1:])]
    if duration <= WINDOW_SECONDS:
        return coverage

    focused = []
    for _, time, cue in sorted(candidates, key=lambda candidate: -candidate[0]):
        window = centered_window(time, duration, cue)
        if all(abs(window.start - prior.start) >= WINDOW_SECONDS * .75 for prior in focused):
            focused.append(window)
        if len(focused) == MAX_FOCUS_WINDOWS:
            break
    return sorted(coverage + focused, key=lambda window: window.start)


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
        step = (end - start) / (FRAMES_PER_WINDOW - 1)
        snap_distance = max(.12 * fps, step * .35)
        for i in range(FRAMES_PER_WINDOW):
            target = round(start + (end - start) * i / (FRAMES_PER_WINDOW - 1))
            if 0 < i < FRAMES_PER_WINDOW - 1 and available:
                nearest = min(available, key=lambda index: abs(index - target))
                if abs(nearest - target) <= snap_distance:
                    target = nearest
            indices.add(target)
        if available and not indices.intersection(available) and len(indices) > 2:
            # Preserve both endpoints, but make at least one mask available in this window.
            index = min(available, key=lambda value: abs(value - (start + end) / 2))
            replace = min(indices - {start, end}, key=lambda value: (abs(value - index), value))
            indices.remove(replace)
            indices.add(index)
        for index in sorted(indices):
            path = Path(work) / f"event_{index:06d}.jpg"
            row = {"frame_id": f"E{index}", "timestamp": round(index / fps, 5),
                   "image_path": str(path), "evidence_role": "event_context"}
            if index in observations:
                observation, tracklet = observations[index]
                row = track_frame(observation, tracklet, work, evidence_view)
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


def track_frame(observation, tracklet, work, evidence_view, *, sighting=False):
    prefix = "sighting" if sighting else "event"
    path = Path(work) / f"{prefix}_{observation.frame_index:06d}.jpg"
    if not path.is_file():
        render_evidence(observation, evidence_view, path)
    return {
        "frame_id": f"{'S' if sighting else 'E'}{observation.frame_index}",
        "timestamp": observation.timestamp,
        "image_path": str(path), "evidence_role": "object_track", "tracklet": tracklet,
    }


def review_prompt(req):
    return (
        f"Requested object: {req.object_phrase}. Question: {req.question}\n"
        "Describe up to two directly visible actions or states of that exact object. "
        "Check its type and distinguishing attributes. Other objects may appear and outlines "
        "are only hints. If the only clear evidence is a location or visible state, describe it. "
        "Describe only these images. Give no claims about what happened throughout the clip. "
        "Every line must name the target object explicitly. State lines describe what is visible "
        "in a still image. Use set_down for a placement and state for an object already resting. "
        "A hand nearby is not contact, and a change in location alone does not establish a pickup. "
        "Use one short line per observation, with a label followed by a vertical bar and the description. "
        "Labels: picked_up, carried, set_down, moved_on_surface, touched, state. "
        'Example: state | The requested object rests on the rug. '
        "If there is no clear observation of this object, write NONE."
    )


def parse_claims(raw):
    """Read typed claims. Unstructured scene narration is not an event proposal."""
    claims = []
    for line in raw.splitlines():
        label, sep, description = line.strip().partition("|")
        kind = label.strip(" *`:-").lower()
        description = description.strip(" |`")
        if sep and kind in EVENT_TYPES and description:
            claims.append((kind, description[:300]))
    return claims[:MAX_CLAIMS_PER_WINDOW]


def has_conflicting_color(phrase, description):
    colors = set(COLORS) | {"black", "white", "gray", "brown"}
    target = re.findall(r"[a-z]+", phrase.lower().replace("grey", "gray"))
    if target[:1] in (["the"], ["a"], ["an"]):
        target = target[1:]
    if len(target) < 2 or target[0] not in colors:
        return False
    words = re.findall(r"[a-z]+", description.lower().replace("grey", "gray"))
    return any(
        word in colors - {target[0]} and (words[i + 1:i + 2] == target[-1:]
                                         or words[i + 1:i + len(target)] == target[1:])
        for i, word in enumerate(words)
    )


def claim_rejection(phrase, kind, description):
    words = re.findall(r"\w+", description.casefold())
    target = re.findall(r"\w+", phrase.casefold())
    if not target or target[-1] not in words or has_conflicting_color(phrase, description):
        return "rejected_target"
    if CLIP_WIDE.search(description):
        return "unsupported_scope"
    if kind == "state" and TRANSITION_WORDS.search(description):
        return "transition_labeled_as_state"
    return None


def visible_target_frames(vlm, req, frames):
    """Locate the object before showing the verifier any proposed event."""
    raw = vlm.answer_video_frames(
        frames,
        f"Requested object: {req.object_phrase}. Return the letters of the images where the "
        "actual requested object is visible and identifiable by its shape and stated attributes. "
        "Check each image separately. Exclude images where only a different object, an obscured "
        "object, or a highlight is visible. Return letters separated by spaces, or NONE.",
        max_new_tokens=24,
        frame_labels=FRAME_LABELS[:len(frames)],
    )
    positions = parse_frame_labels(raw, len(frames))
    return [frames[position] for position in positions or []], raw, positions is not None


def verify_claim(vlm, req, frames, kind, description):
    """Attribute a claim using only frames that passed the object-presence check."""
    if len(frames) < (1 if kind == "state" else 2):
        return None, {"status": "insufficient_target_frames"}
    tracked = [FRAME_LABELS[i] for i, frame in enumerate(frames)
               if frame["evidence_role"] == "object_track"]
    if kind != "state" and not tracked:
        return None, {"status": "missing_track_anchor"}

    evidence = ACTION_EVIDENCE.get(kind, "Show the stated condition directly in the scene.")
    if kind != "state":
        evidence += (
            f" The proposal must be supported by the tracked object in at least one of images "
            f"{' '.join(tracked)}. A nearby object or an action on a different item is insufficient."
        )
    raw = vlm.answer_video_frames(
        frames,
        f"Requested object: {req.object_phrase}. Proposed {kind}: {description}\n"
        "Respond with the letters of the images that directly support this exact observation "
        "of the requested object. The object must match its stated attributes in the scene. "
        f"{evidence} Ignore any outline on a different item. "
        "Return only letters separated by spaces, such as A C, or NONE if the claim is unclear "
        "or unsupported.",
        max_new_tokens=24,
        frame_labels=FRAME_LABELS[:len(frames)],
    )
    positions = parse_frame_labels(raw, len(frames))
    if positions is None:
        return None, {"status": "invalid_citation", "response": raw}
    if len(positions) < (1 if kind == "state" else 2):
        return None, {"status": "insufficient_support", "response": raw}
    if kind != "state" and not any(FRAME_LABELS[position] in tracked for position in positions):
        return None, {"status": "missing_track_anchor", "response": raw}
    cited = [frames[position] for position in positions]
    event = {
        "type": kind, "description": description, "start": cited[0]["timestamp"],
        "end": cited[-1]["timestamp"], "frame_ids": [frame["frame_id"] for frame in cited],
    }
    return event, {"status": "supported", "frame_ids": event["frame_ids"], "response": raw}


def verified_sightings(tracklets, accepted, tracklet_checks, work, evidence_view):
    """Keep evidence of presence at the exact frames inspected by the tracker verifier."""
    checked_indices = {check["tracklet"]: set(check["frame_indices"])
                       for check in tracklet_checks if check["accepted"]}
    verified = {}
    for index in accepted:
        for row in tracklets[index].observations:
            if row.frame_index in checked_indices.get(index, ()):
                verified.setdefault(row.frame_index, (row, index))
    ordered = [verified[index] for index in sorted(verified)]
    if len(ordered) > MAX_SIGHTING_FRAMES:
        ordered = [ordered[round(i * (len(ordered) - 1) / (MAX_SIGHTING_FRAMES - 1))]
                   for i in range(MAX_SIGHTING_FRAMES)]
    return [track_frame(row, index, work, evidence_view, sighting=True) for row, index in ordered]


def investigate(vlm, req, video, work, tracklets, accepted, semantic, duration,
                *, decode=decode_window_frames):
    """Cover the clip, then independently attribute claims to source frames."""
    observations = {}
    for index in accepted:
        for row in tracklets[index].observations:
            observations.setdefault(row.frame_index, (row, index))
    windows = propose_windows(tracklets, accepted, semantic, duration)
    events, reviews, used = [], [], {}
    for window in windows:
        frames = sorted(decode(video, window, observations, work, req.evidence_view),
                        key=lambda frame: frame["timestamp"])
        used.update((frame["frame_id"], frame) for frame in frames)
        raw = vlm.answer_video_frames(frames, review_prompt(req), max_new_tokens=144) if frames else ""
        claims = [(kind, description, claim_rejection(req.object_phrase, kind, description))
                  for kind, description in parse_claims(raw)]
        visible, presence_response, presence_ok = [], None, True
        if any(rejection is None for _, _, rejection in claims):
            visible, presence_response, presence_ok = visible_target_frames(vlm, req, frames)
        checks = []
        for kind, description, rejection in claims:
            if rejection or not presence_ok:
                checks.append({"type": kind, "description": description,
                               "status": rejection or "invalid_citation"})
                continue
            event, check = verify_claim(vlm, req, visible, kind, description)
            checks.append({"type": kind, "description": description, **check})
            if event:
                events.append(event)
        reviews.append({"start": window.start, "end": window.end, "cue": window.cue,
                        "frame_ids": [frame["frame_id"] for frame in frames],
                        "target_presence_response": presence_response,
                        "target_visible_frame_ids": [frame["frame_id"] for frame in visible],
                        "observation_response": raw, "claim_checks": checks})

    # Favor denser evidence for repeated claims, while keeping separate observations.
    unique = []
    for event in sorted(events, key=lambda row: (-len(row["frame_ids"]), row["start"])):
        if not any(event["type"] == other["type"]
                   and (event["type"] != "state"
                        or event["description"].lower() == other["description"].lower())
                   and max(event["start"], other["start"]) <= min(event["end"], other["end"])
                   for other in unique):
            unique.append(event)
    return sorted(unique, key=lambda row: (row["start"], row["end"])), reviews, sorted(
        used.values(), key=lambda frame: frame["timestamp"],
    )
