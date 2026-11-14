from pathlib import Path

from src.pipeline.video_rag import VideoRAGPipeline


def main():
    index_dir = Path("data/processed/sample_long")

    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"Existing FAISS index not found in {index_dir}"
        )

    pipeline = VideoRAGPipeline()

    result = pipeline.query(
        index_dir=index_dir,
        question="What is the person doing in this video?",
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSUPPORTING FRAMES:")
    for frame in result["frames"]:
        print(
            f'{frame["timestamp"]:.2f}s '
            f'{frame["image_path"]}'
        )


if __name__ == "__main__":
    main()
