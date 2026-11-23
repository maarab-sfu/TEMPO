"""Build query-specific evidence from SAM 2 tracklets and source pixels."""
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from scripts.experiment_sam2_tracking import box_iou, decode_window, load_predictor, mask_bbox, propagate
from scripts.track_grounded_object import eligible, scale_to_frame

SAM_MODEL = "facebook/sam2.1-hiera-tiny"
COLORS = {"red": 0, "orange": 1, "yellow": 2, "green": 3,
          "cyan": 4, "blue": 5, "purple": 6, "pink": 7}


@dataclass
class Observation:
    frame_index: int
    timestamp: float
    image: Image.Image
    mask: np.ndarray
    box: tuple

    def record(self):
        return {"frame_index": self.frame_index, "timestamp": self.timestamp,
                "box": list(self.box), "mask_pixels": int(self.mask.sum())}


@dataclass
class Tracklet:
    seed: dict
    observations: list[Observation]
    embedding: np.ndarray
    colors: np.ndarray
    rank: float


def grounding_times(semantic, duration):
    """Cover the video at roughly 2 s resolution, capped for long uploads."""
    count = min(32, max(2, int(np.ceil(duration / 2)) + 1))
    return sorted(set(np.linspace(0, duration, count).round(3))
                  | {round(float(frame["timestamp"]), 3) for frame in semantic})


def choose_seeds(candidates, duration):
    """Keep a strong seed per time band and nearby competing instances."""
    bands = {}
    for candidate in eligible(candidates):
        band = min(5, int(candidate["timestamp"] / max(duration, .001) * 6))
        bands.setdefault(band, []).append(candidate)

    seeds = []
    for group in bands.values():
        best = group[0]
        distinct = [best]
        for candidate in group[1:]:
            if candidate["frame_index"] != best["frame_index"]:
                continue
            if all(box_iou(candidate["box"], prior["box"]) < .3 for prior in distinct):
                distinct.append(candidate)
                if len(distinct) == 3:
                    break
        seeds.extend(distinct)
    return sorted(seeds, key=lambda seed: seed["timestamp"])


def masked_crop(observation):
    box = observation.box
    image = observation.image.crop(box)
    mask = Image.fromarray(observation.mask).crop(box)
    return Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)


def color_distribution(observation):
    hsv = np.asarray(observation.image.convert("HSV"))[observation.mask]
    hue, saturation, value = hsv.T
    # PIL represents hue in [0,255]. Separate neutral pixels from hue.
    edges = [8, 20, 50, 110, 140, 185, 215, 245]
    bins = np.digitize(hue, edges) % 8
    bins[(saturation < 65) | (value < 45)] = 8
    return np.bincount(bins, minlength=9) / len(bins)


def visible_runs(frames, indices, masks, fps, seed_pos):
    runs = []
    current = []
    for position, frame_index in enumerate(indices):
        mask = masks.get(position)
        box = mask_bbox(mask) if mask is not None else None
        if box is None or mask.sum() < 16 or mask.mean() > .35:
            if current:
                runs.append(current)
                current = []
            continue
        current.append((position, Observation(
            frame_index, round(frame_index / fps, 5),
            Image.fromarray(frames[position][:, :, ::-1]),
            mask, tuple(box))))
    if current:
        runs.append(current)
    return [(min(run, key=lambda point: abs(point[0] - seed_pos))[1],
             [observation for _, observation in run]) for run in runs
            if len(run) >= 2 or any(pos == seed_pos for pos, _ in run)]


def describe_run(seed, reference, observations, encoder, text_embedding, color, directory):
    # Sample the seed-side observation and both ends of the uninterrupted run.
    samples = {o.frame_index: o for o in (reference, observations[0], observations[-1])}
    paths = []
    for observation in samples.values():
        path = directory / f"crop_{observation.frame_index}.png"
        masked_crop(observation).save(path)
        paths.append(path)
    embedding = encoder.encode_images(paths).mean(axis=0)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
    colors = np.mean([color_distribution(o) for o in samples.values()], axis=0)
    rank = .55 * seed["score"] + .45 * float(embedding @ text_embedding)
    if color is not None:
        rank += .65 * float(colors[color])
    if reference.frame_index != seed["frame_index"]:
        rank -= .2
    return Tracklet(seed, observations, embedding, colors, rank)


def track_windows(video, seeds, phrase, before, after, encoder, work):
    device = encoder.device
    predictor = load_predictor(SAM_MODEL, device)
    text_embedding = encoder.encode_text([phrase])[0]
    color_words = set(re.findall(r"[a-z]+", phrase.lower())) & COLORS.keys()
    color = COLORS[next(iter(color_words))] if len(color_words) == 1 else None
    tracklets, windows = [], []
    for seed in seeds:
        with TemporaryDirectory(dir=work) as directory:
            directory = Path(directory)
            frames, indices, seed_pos, fps, width, height = decode_window(
                video, seed["timestamp"], before, after, 5, 640)
            target = frames[seed_pos]
            box = scale_to_frame(seed["box"], width, height, target.shape[1], target.shape[0])
            backward = propagate(frames, seed_pos, box, SAM_MODEL, device, directory, True,
                                 predictor=predictor)
            forward = propagate(frames, seed_pos, box, SAM_MODEL, device, directory, False,
                                predictor=predictor)
            masks = {**backward, **forward}
            runs = visible_runs(frames, indices, masks, fps, seed_pos)
            windows.append({"seed": seed, "sampled_interval": [indices[0] / fps, indices[-1] / fps],
                            "sampled_frames": len(indices), "visible_runs": len(runs)})
            for reference, observations in runs:
                tracklets.append(describe_run(seed, reference, observations, encoder,
                                              text_embedding, color, directory))
    return tracklets, windows


def link_score(left, right):
    # Nearby samples may have different strides because each window is seed-relative.
    overlaps = []
    right_index = 0
    for observation in left.observations:
        while (right_index + 1 < len(right.observations)
               and abs(right.observations[right_index + 1].timestamp - observation.timestamp)
               < abs(right.observations[right_index].timestamp - observation.timestamp)):
            right_index += 1
        other = right.observations[right_index]
        if abs(observation.timestamp - other.timestamp) <= .11:
            overlaps.append(box_iou(observation.box, other.box))
    if overlaps:
        overlap = float(np.median(overlaps))
        return (overlap, "overlap", overlap >= .5)
    appearance = float(left.embedding @ right.embedding)
    color = float(np.minimum(left.colors, right.colors).sum())
    score = .7 * appearance + .3 * color
    return score, "appearance", appearance >= .76 and color >= .48 and score >= .73


def verify_tracklets(vlm, phrase, tracklets, work):
    """Check the masked object class and attributes before choosing an identity."""
    checks = []
    for index, tracklet in enumerate(tracklets):
        rows = tracklet.observations
        reference = min(rows, key=lambda row: abs(row.frame_index - tracklet.seed["frame_index"]))
        samples = {row.frame_index: row for row in (rows[0], reference, rows[-1])}
        sheet = Image.new("RGB", (192 * len(samples), 192), (127, 127, 127))
        for panel, row in enumerate(samples.values()):
            crop = ImageOps.contain(masked_crop(row), (184, 184))
            sheet.paste(crop, (panel * 192 + (192 - crop.width) // 2, (192 - crop.height) // 2))
        path = work / f"tracklet_check_{index:02d}.png"
        sheet.save(path)
        # Check shape/category as well as color, which can match an unrelated object.
        reply = vlm.answer_image(
            path,
            f"Requested object: {phrase}. These panels show segmented candidates. "
            "Does the segmented item in every panel clearly match the requested object type "
            "and all its stated attributes? Matching color alone is insufficient. "
            "Reply with one word: yes, no, or unclear.",
            max_new_tokens=8,
        )
        verdict = reply.strip().lower().rstrip(".")
        accepted = verdict == "yes"
        checks.append({"tracklet": index, "frame_indices": list(samples), "image_path": str(path),
                       "response": reply, "accepted": accepted})
    return checks


def associate(tracklets, candidates=None):
    if candidates is None:
        candidates = list(range(len(tracklets)))
    if not candidates:
        return [], []
    root = max(candidates, key=lambda index: tracklets[index].rank)
    selected = [root]
    links = []
    for index in candidates:
        if index == root:
            continue
        candidate = tracklets[index]
        score, cue, matches = link_score(tracklets[root], candidate)
        # A similarly plausible, simultaneous alternative makes an appearance-only link ambiguous.
        ambiguous = False
        if matches and cue == "appearance":
            for alt_index in candidates:
                if alt_index in (root, index):
                    continue
                _, comparison, same_instance = link_score(candidate, tracklets[alt_index])
                if comparison != "overlap" or same_instance:
                    continue
                alt_score, _, alt_matches = link_score(tracklets[root], tracklets[alt_index])
                if alt_matches and alt_score >= score - .07:
                    ambiguous = True
                    break
        conflicts = any(link_score(tracklets[other], candidate)[1:] == ("overlap", False)
                        for other in selected if other != root)
        accepted = matches and not ambiguous and not conflicts
        reason = ("accepted" if accepted else "ambiguous" if ambiguous else
                  "simultaneous_conflict" if conflicts else "insufficient_support")
        if accepted:
            selected.append(index)
        links.append({"source": root, "target": index, "score": score, "cue": cue,
                      "decision": reason})
    return selected, links


def select_observations(tracklets, selected, budget):
    """Cover the endpoints and include nearby before/after views of a position change."""
    observations = {}
    for index in selected:
        for observation in tracklets[index].observations:
            observations.setdefault(observation.frame_index, (observation, index))
    ordered = sorted(observations.values(), key=lambda row: row[0].frame_index)
    if len(ordered) < 2 or budget == 1:
        return ordered[:budget]
    chosen = {0, len(ordered) - 1}
    if budget >= 4:
        pairs = [(i, 1 - box_iou(row[0].box, following[0].box))
                 for i, (row, following) in enumerate(zip(ordered, ordered[1:]))
                 if row[1] == following[1] and following[0].timestamp - row[0].timestamp <= .6]
        if pairs:
            pair, _ = max(pairs, key=lambda item: item[1])
            chosen.update((pair, pair + 1))
    while len(chosen) < min(budget, len(ordered)):
        index = max((i for i in range(len(ordered)) if i not in chosen),
                    key=lambda i: min(abs(ordered[i][0].timestamp - ordered[j][0].timestamp)
                                      for j in chosen))
        chosen.add(index)
    return [ordered[i] for i in sorted(chosen)]


def render_evidence(observation, view, path):
    scene = observation.image.copy()
    if view == "box":
        ImageDraw.Draw(scene).rectangle(observation.box, outline=(255, 30, 190), width=3)
    elif view == "mask":
        mask = Image.fromarray(observation.mask)
        edge = np.asarray(mask.filter(ImageFilter.MaxFilter(5))) & ~np.asarray(mask)
        scene.paste((255, 30, 190), mask=Image.fromarray(edge))
        # Add the close-up beside the scene so it never obscures a person or surface.
        inset = ImageOps.contain(masked_crop(observation), (128, 128))
        panel = Image.new("RGB", (scene.width + 132, scene.height), (45, 45, 45))
        panel.paste(scene)
        panel.paste(inset, (scene.width + 2, 2))
        scene = panel
    scene.save(path, quality=93)
