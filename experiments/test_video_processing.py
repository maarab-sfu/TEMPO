from src.perception.video_processor import VideoProcessor


def main():

    processor = VideoProcessor(
        video_path="data/raw/sample.mp4",
        output_dir="data/processed/sample",
        sampling_interval=2.0,
        max_resolution=640,
    )

    metadata = processor.extract_frames()

    print("\nFirst five extracted frames:")

    for item in metadata[:5]:
        print(
            f"Frame {item['frame_id']} | "
            f"Time: {item['timestamp']:.2f}s | "
            f"Path: {item['image_path']}"
        )


if __name__ == "__main__":
    main()