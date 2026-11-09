import cv2
import json
import numpy as np

from pathlib import Path


VIDEO_PATH = Path(
    "data/raw/youcook2/--bv0V6ZjWI.mp4"
)

TRACKS_PATH = Path(
    "data/processed/youcook2/--bv0V6ZjWI/tracks.json"
)

OUTPUT_PATH = Path(
    "results/youcook2/tracking_preview.mp4"
)

START_TIME = 60
DURATION = 30


def color_for_id(track_id):
    rng = np.random.default_rng(track_id)
    return tuple(int(x) for x in rng.integers(50, 255, 3))


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TRACKS_PATH, encoding="utf-8") as f:
        tracking_data = json.load(f)

    tracking_by_frame = {
        item["frame_index"]: item
        for item in tracking_data
    }

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(START_TIME * fps)
    end_frame = int((START_TIME + DURATION) * fps)

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_index in range(start_frame, end_frame):
        success, frame = cap.read()

        if not success:
            break

        frame_data = tracking_by_frame.get(frame_index)

        if frame_data:
            for detection in frame_data["detections"]:
                x1, y1, x2, y2 = map(
                    int, detection["bbox"]
                )

                track_id = detection["track_id"]
                class_name = detection["class_name"]

                color = color_for_id(track_id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                label = f"{class_name} ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Saved visualization: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()