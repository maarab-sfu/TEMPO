"""Select diverse, high-motion evidence frames from a SAM2 trajectory.

This is an experimental export utility. It does not change VideoMind's
production retriever, citations, or VLM prompts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def trajectory_motion(rows):
    """Normalized displacement between successive valid boxes (pixels/frame)."""
    result = [0.0] * len(rows)
    previous = None
    for index, row in enumerate(rows):
        box = row.get("sam2_box")
        if box is None:
            previous = None
            continue
        if previous is not None:
            previous_index, previous_box = previous
            elapsed = float(row["timestamp"]) - float(rows[previous_index]["timestamp"])
            if elapsed > 0:
                p, q = center(box), center(previous_box)
                result[index] = math.dist(p, q) / elapsed
        previous = (index, box)
    return result


def select_indices(rows, count, min_spacing=0.5):
    """Select endpoints and motion peaks with a minimum temporal separation."""
    if count < 1 or min_spacing < 0:
        raise ValueError("count must be positive and min_spacing nonnegative")
    valid = [i for i, row in enumerate(rows) if row.get("sam2_box") is not None]
    if not valid:
        return []
    motion = trajectory_motion(rows)
    chosen = []

    def add(index):
        if len(chosen) >= count or index in chosen:
            return
        timestamp = float(rows[index]["timestamp"])
        if all(abs(timestamp - float(rows[j]["timestamp"])) >= min_spacing for j in chosen):
            chosen.append(index)

    add(valid[0])
    add(valid[-1])
    for index in sorted(valid, key=lambda i: (-motion[i], i)):
        add(index)
    # Fill remaining slots with temporally distributed candidates, not only motion.
    while len(chosen) < min(count, len(valid)):
        eligible = [i for i in valid if i not in chosen and all(
            abs(float(rows[i]["timestamp"]) - float(rows[j]["timestamp"])) >= min_spacing
            for j in chosen
        )]
        if not eligible:
            break
        if chosen:
            index = max(eligible, key=lambda i: (
                min(abs(float(rows[i]["timestamp"]) - float(rows[j]["timestamp"])) for j in chosen),
                -i,
            ))
        else:
            index = eligible[0]
        add(index)
    return sorted(chosen, key=lambda i: float(rows[i]["timestamp"]))


def export_evidence(video_path, rows, selected, output_dir):
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid source FPS")
    manifest = []
    try:
        for number, index in enumerate(selected):
            row = rows[index]
            frame_index = int(row["frame_index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {frame_index}")
            filename = f"evidence_{number:02d}_{frame_index:06d}.jpg"
            destination = output_dir / filename
            if not cv2.imwrite(str(destination), frame):
                raise RuntimeError(f"Could not save {destination}")
            manifest.append({
                "evidence_id": f"track_{number:02d}",
                "timestamp": float(row["timestamp"]),
                "frame_index": frame_index,
                "image_path": str(destination),
                "sam2_box": row["sam2_box"],
                "source": "sam2_trajectory",
            })
    finally:
        cap.release()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--min-spacing", type=float, default=0.5)
    args = parser.parse_args()
    if args.max_frames < 1 or args.min_spacing < 0:
        parser.error("max-frames must be >= 1 and min-spacing >= 0")
    rows = json.loads(args.trajectory.read_text(encoding="utf-8"))
    indices = select_indices(rows, args.max_frames, args.min_spacing)
    if not indices:
        raise RuntimeError("No valid SAM2 bounding boxes in trajectory")
    manifest = export_evidence(args.video, rows, indices, args.output)
    output_path = args.output / "evidence_manifest.json"
    output_path.write_text(json.dumps({
        "video": str(args.video),
        "trajectory": str(args.trajectory),
        "selection": "endpoints_and_motion_diversity",
        "frames": manifest,
        "caution": "Motion-selected frames are not yet query-relevant evidence; no QA evaluation has been run.",
    }, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest)} frames to {args.output}")
    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    main()
