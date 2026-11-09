import cv2
import numpy as np

from collections import defaultdict
from pathlib import Path


class TrackletEncoder:
    def __init__(
        self,
        clip_encoder,
        samples_per_track=5,
        min_track_length=10,
        crop_padding=0.1,
    ):
        self.encoder = clip_encoder
        self.samples_per_track = samples_per_track
        self.min_track_length = min_track_length
        self.crop_padding = crop_padding

    def group_tracks(self, tracking_data):
        tracks = defaultdict(list)

        for frame in tracking_data:
            for detection in frame["detections"]:
                tracks[detection["track_id"]].append({
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "bbox": detection["bbox"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                })

        return tracks

    def encode_tracklets(
        self,
        video_path,
        tracking_data,
        output_dir,
    ):
        output_dir = Path(output_dir)
        crop_dir = output_dir / "tracklet_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        tracks = self.group_tracks(tracking_data)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        tracklet_embeddings = {}
        tracklet_metadata = {}

        try:
            for track_id, observations in tracks.items():

                if len(observations) < self.min_track_length:
                    continue

                # Sample observations across the track.
                indices = np.linspace(
                    0,
                    len(observations) - 1,
                    min(self.samples_per_track, len(observations)),
                    dtype=int,
                )

                crop_paths = []

                for idx in indices:
                    obs = observations[idx]

                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        obs["frame_index"],
                    )

                    success, frame = cap.read()

                    if not success:
                        continue

                    height, width = frame.shape[:2]

                    x1, y1, x2, y2 = obs["bbox"]

                    box_width = x2 - x1
                    box_height = y2 - y1

                    padding_x = box_width * self.crop_padding
                    padding_y = box_height * self.crop_padding

                    x1 = max(0, int(x1 - padding_x))
                    y1 = max(0, int(y1 - padding_y))
                    x2 = min(width, int(x2 + padding_x))
                    y2 = min(height, int(y2 + padding_y))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    crop_path = (
                        crop_dir /
                        f"track_{track_id}_frame_{obs['frame_index']}.jpg"
                    )

                    cv2.imwrite(str(crop_path), crop)
                    crop_paths.append(str(crop_path))

                if not crop_paths:
                    continue

                # Encode object crops.
                embeddings = self.encoder.encode_images(
                    crop_paths,
                    batch_size=8,
                )

                # Average appearance across observations.
                mean_embedding = np.mean(embeddings, axis=0)

                mean_embedding /= (
                    np.linalg.norm(mean_embedding) + 1e-12
                )

                tracklet_embeddings[track_id] = mean_embedding

                tracklet_metadata[track_id] = {
                    "start": observations[0]["timestamp"],
                    "end": observations[-1]["timestamp"],
                    "num_observations": len(observations),
                    "class_name": observations[0]["class_name"],
                    "crop_paths": crop_paths,
                }

        finally:
            cap.release()

        return tracklet_embeddings, tracklet_metadata