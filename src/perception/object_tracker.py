import cv2

from pathlib import Path
from ultralytics import YOLO


class ObjectTracker:
    def __init__(
        self,
        model_name="yolo11n.pt",
        confidence=0.25,
        tracker_config="bytetrack.yaml",
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.tracker_config = tracker_config

    def track_video(
        self,
        video_path,
        output_path=None,
        frame_stride=1,
    ):
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            cap.release()
            raise RuntimeError("Invalid video FPS")

        results = []

        frame_index = 0

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                if frame_index % frame_stride != 0:
                    frame_index += 1
                    continue

                timestamp = frame_index / fps

                prediction = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.confidence,
                    verbose=False,
                )[0]

                detections = []

                boxes = prediction.boxes

                if boxes is not None and boxes.id is not None:
                    for box in boxes:
                        track_id = int(box.id.item())
                        class_id = int(box.cls.item())

                        detections.append({
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": self.model.names[class_id],
                            "confidence": float(box.conf.item()),
                            "bbox": box.xyxy[0].tolist(),
                        })

                results.append({
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "detections": detections,
                })

                frame_index += 1

        finally:
            cap.release()

        return results