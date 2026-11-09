import json

from pathlib import Path

from src.perception.object_tracker import ObjectTracker


def main():
    video_path = Path(
        "data/raw/youcook2/--bv0V6ZjWI.mp4"
    )

    output_dir = Path(
        "data/processed/youcook2/--bv0V6ZjWI"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ObjectTracker(
        model_name="yolo11n.pt",
        confidence=0.25,
    )

    results = tracker.track_video(
        video_path=video_path,
        frame_stride=1,
    )

    output_path = output_dir / "tracks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    unique_ids = set()

    detection_count = 0

    for frame in results:
        for detection in frame["detections"]:
            unique_ids.add(detection["track_id"])
            detection_count += 1

    print(f"Processed frames: {len(results)}")
    print(f"Total detections: {detection_count}")
    print(f"Unique track IDs: {len(unique_ids)}")
    print(f"Saved tracking results: {output_path}")


if __name__ == "__main__":
    main()