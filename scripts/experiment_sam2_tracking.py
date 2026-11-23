"""Experimental SAM 2.1 object propagation seeded from M4.5 detections.

No changes to VideoMind production indexing, retrieval, or answering.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path


def choose_seed(tracks: dict, track_id: str, target_time: float | None) -> dict:
    matching = [track for track in tracks["tracks"] if track["track_id"] == track_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one track named {track_id!r}; found {len(matching)}")
    observations = matching[0]["observations"]
    if not observations:
        raise ValueError(f"Track {track_id} has no observations")
    if target_time is None:
        return max(observations, key=lambda obs: float(obs["score"]))
    return min(observations, key=lambda obs: (abs(float(obs["timestamp"]) - target_time),
                                                     -float(obs["score"])))


def scale_box(box: list[float], source_width: int, source_height: int,
              target_width: int, target_height: int, detector_max_side: int = 800) -> list[float]:
    """M4.2/M4.5 boxes use frames resized to at most detector_max_side."""
    factor = min(1.0, detector_max_side / max(source_width, source_height))
    detector_width = round(source_width * factor)
    detector_height = round(source_height * factor)
    sx, sy = target_width / detector_width, target_height / detector_height
    x1, y1, x2, y2 = box
    return [max(0.0, min(target_width - 1.0, x1 * sx)),
            max(0.0, min(target_height - 1.0, y1 * sy)),
            max(0.0, min(target_width - 1.0, x2 * sx)),
            max(0.0, min(target_height - 1.0, y2 * sy))]


def mask_bbox(mask):
    import numpy as np
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def box_iou(a, b):
    if a is None or b is None:
        return None
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_window(video_path: Path, center_seconds: float, before: float, after: float,
                  sample_fps: float, max_side: int):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video metadata invalid; this pilot requires constant-FPS video")
    start = max(0, int(round((center_seconds - before) * fps)))
    stop = min(count - 1, int(round((center_seconds + after) * fps)))
    stride = max(1, int(round(fps / sample_fps)))
    indices = list(range(start, stop + 1, stride))
    seed_index = max(0, min(count - 1, int(round(center_seconds * fps))))
    indices = sorted(set(indices + [seed_index]))
    frames, kept = [], []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        if max(width, height) > max_side:
            factor = max_side / max(width, height)
            frame = cv2.resize(frame, (round(width * factor), round(height * factor)))
        frames.append(frame)
        kept.append(index)
    cap.release()
    if seed_index not in kept:
        raise RuntimeError("Could not decode the selected seed frame")
    return frames, kept, kept.index(seed_index), fps, width, height


def load_predictor(model_id, device):
    try:
        from sam2.build_sam import build_sam2_video_predictor_hf
    except ImportError as exc:
        raise RuntimeError("SAM 2 not installed. See README_M48.md") from exc

    return build_sam2_video_predictor_hf(model_id, device=device)


def propagate(frames, seed_position: int, box, model_id: str, device: str,
              workdir: Path, backwards: bool, *, predictor=None):
    """Run SAM2 independently in each direction, starting from seed at position zero."""
    import cv2
    import numpy as np
    import torch
    order = list(range(seed_position, -1, -1)) if backwards else list(range(seed_position, len(frames)))
    image_dir = workdir / ("reverse" if backwards else "forward")
    image_dir.mkdir(parents=True, exist_ok=True)
    for local_index, original_index in enumerate(order):
        cv2.imwrite(str(image_dir / f"{local_index:05d}.jpg"), frames[original_index],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    owns_predictor = predictor is None
    if owns_predictor:
        predictor = load_predictor(model_id, device)
    state = predictor.init_state(video_path=str(image_dir), offload_video_to_cpu=True)
    results = {}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1,
            box=np.asarray(box, dtype=np.float32),
        )
        for local_idx, ids, logits in predictor.propagate_in_video(state):
            if 1 not in ids:
                continue
            position = ids.index(1) if isinstance(ids, list) else list(ids).index(1)
            mask = (logits[position] > 0.0).squeeze().cpu().numpy().astype(bool)
            results[order[local_idx]] = mask
    predictor.reset_state(state)
    if owns_predictor:
        del predictor
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--track-id", default="person_0002")
    parser.add_argument("--seed-time", type=float, default=None,
                        help="Nearest M4.5 observation; omit for highest-confidence detection")
    parser.add_argument("--before", type=float, default=5.0)
    parser.add_argument("--after", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--detector-max-side", type=int, default=800)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/m48_sam2_pilot"))
    args = parser.parse_args()
    if args.before < 0 or args.after < 0 or args.sample_fps <= 0 or args.max_side < 64:
        parser.error("Invalid time window, sampling rate, or image size")

    import cv2
    import numpy as np
    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    seed = choose_seed(tracks, args.track_id, args.seed_time)
    frames, indices, seed_pos, fps, source_w, source_h = decode_window(
        args.video, float(seed["timestamp"]), args.before, args.after,
        args.sample_fps, args.max_side,
    )
    box = scale_box(seed["box"], source_w, source_h,
                    frames[0].shape[1], frames[0].shape[0], args.detector_max_side)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid seed box after scaling: {box}")
    print(f"Seed: {args.track_id}, t={seed['timestamp']}s, box={box}", flush=True)
    print(f"Frames: {len(frames)}, sample_fps={args.sample_fps}", flush=True)

    with tempfile.TemporaryDirectory(prefix="videomind_sam2_") as temp:
        workdir = Path(temp)
        forward = propagate(frames, seed_pos, box, args.model, args.device, workdir, False)
        backward = propagate(frames, seed_pos, box, args.model, args.device, workdir, True)
    masks = {**backward, **forward}

    args.output.mkdir(parents=True, exist_ok=True)
    annotated = args.output / "annotated"
    annotated.mkdir(exist_ok=True)
    rows = []
    track = next(t for t in tracks["tracks"] if t["track_id"] == args.track_id)
    for position, (frame, frame_idx) in enumerate(zip(frames, indices)):
        mask = masks.get(position)
        predicted = mask_bbox(mask) if mask is not None else None
        t = frame_idx / fps
        nearest = min(track["observations"], key=lambda obs: abs(float(obs["timestamp"]) - t))
        delta = abs(float(nearest["timestamp"]) - t)
        old_box = scale_box(nearest["box"], source_w, source_h,
                            frame.shape[1], frame.shape[0], args.detector_max_side)
        comparison = box_iou(predicted, old_box) if delta <= 0.3 else None
        rows.append({"frame_index": frame_idx, "timestamp": round(t, 4),
                     "sam2_box": predicted, "sam2_mask_pixels": int(mask.sum()) if mask is not None else 0,
                     "m45_nearest_observation_seconds": nearest["timestamp"],
                     "m45_box_iou": round(comparison, 4) if comparison is not None else None})
        overlay = frame.copy()
        if mask is not None:
            overlay[mask] = (0.65 * overlay[mask] + 0.35 * np.array([0, 210, 0])).astype(np.uint8)
        if predicted:
            cv2.rectangle(overlay, (predicted[0], predicted[1]),
                          (predicted[2], predicted[3]), (0, 255, 0), 2)
        if delta <= 0.3:
            cv2.rectangle(overlay, (round(old_box[0]), round(old_box[1])),
                          (round(old_box[2]), round(old_box[3])), (0, 150, 255), 2)
        cv2.putText(overlay, f"{t:.2f}s SAM2=green M45=orange", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(str(annotated / f"{position:05d}.jpg"), overlay)
    compared = [r["m45_box_iou"] for r in rows if r["m45_box_iou"] is not None]
    summary = {"video": str(args.video), "track_id": args.track_id,
               "model": args.model, "seed": seed, "seed_box_rescaled": box,
               "sample_fps_requested": args.sample_fps, "frames": len(rows),
               "masks_present": sum(r["sam2_box"] is not None for r in rows),
               "mean_m45_box_iou_when_near": round(sum(compared) / len(compared), 4) if compared else None,
               "compared_frames": len(compared),
               "warning": "M4.5 boxes are NOT ground truth. IoU measures agreement, not accuracy. "
                          "Mask propagation can drift or swap objects."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "trajectory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Annotated frames: {annotated}")


if __name__ == "__main__":
    main()
