import cv2
import json

from pathlib import Path


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        sampling_interval: float = 1.0,
        max_resolution: int = 640,
    ):
        """
        Args:
            video_path: Path to the input video.
            output_dir: Directory for extracted frames.
            sampling_interval: Seconds between sampled frames.
            max_resolution: Maximum width or height of saved frames.
        """

        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be positive")

        if max_resolution <= 0:
            raise ValueError("max_resolution must be positive")

        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.sampling_interval = sampling_interval
        self.max_resolution = max_resolution

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self):

        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {self.video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            cap.release()
            raise RuntimeError("Invalid video FPS")

        duration = total_frames / fps

        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {duration:.2f} seconds")

        metadata = []

        sample_index = 0
        next_timestamp = 0.0

        try:
            while next_timestamp < duration:

                # Seek to the requested timestamp.
                cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    next_timestamp * 1000
                )

                success, frame = cap.read()

                if not success:
                    break

                # Retrieve the actual decoded timestamp.
                actual_timestamp = (
                    cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                )

                height, width = frame.shape[:2]

                # Resize while preserving aspect ratio.
                scale = min(
                    1.0,
                    self.max_resolution / max(width, height)
                )

                if scale < 1.0:
                    new_width = round(width * scale)
                    new_height = round(height * scale)

                    frame = cv2.resize(
                        frame,
                        (new_width, new_height),
                        interpolation=cv2.INTER_AREA
                    )

                filename = f"frame_{sample_index:06d}.jpg"

                output_path = self.output_dir / filename

                saved = cv2.imwrite(str(output_path), frame)

                if not saved:
                    raise RuntimeError(
                        f"Failed to save frame: {output_path}"
                    )

                metadata.append({
                    "frame_id": sample_index,
                    "requested_timestamp": round(
                        next_timestamp, 3
                    ),
                    "timestamp": round(
                        actual_timestamp, 3
                    ),
                    "image_path": str(output_path),
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                })

                sample_index += 1
                next_timestamp = (
                    sample_index * self.sampling_interval
                )

        finally:
            cap.release()

        metadata_path = self.output_dir / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Extracted {len(metadata)} frames")
        print(f"Metadata saved to {metadata_path}")

        return metadata