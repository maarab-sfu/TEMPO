from pathlib import Path
from src.perception.video_processor import VideoProcessor


def main():
    video_path = Path("data/raw/sample_long.mp4")
    output_dir = Path("data/processed/sample_long")

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    processor = VideoProcessor(
        video_path=str(video_path),
        output_dir=str(output_dir),
        sampling_interval=2.0,
        max_resolution=640
    )

    metadata = processor.extract_frames()

    print(f"\nSuccessfully extracted {len(metadata)} frames")


if __name__ == "__main__":
    main()