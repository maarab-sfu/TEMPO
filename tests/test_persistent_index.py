from pathlib import Path
import json

from src.pipeline.video_rag import VideoRAGPipeline
from src.storage.index_store import IndexStore


def main():
    video_path = Path("data/raw/sample_long.mp4")
    index_dir = Path("data/indexes/sample_long")

    pipeline = VideoRAGPipeline()

    print("\n=== INDEXING ===")

    result = pipeline.index_video(
        video_path=video_path,
        output_dir=index_dir,
        sampling_interval=2.0,
    )

    print(result)

    print("\n=== VALIDATING STORAGE ===")

    store = IndexStore(index_dir)
    retriever, manifest = store.load()

    print(json.dumps(manifest, indent=2))

    assert retriever.index.ntotal == manifest["frame_count"]
    assert all(
        Path(frame["image_path"]).is_file()
        for frame in retriever.metadata
    )

    print("\n=== QUERYING ===")

    result = pipeline.query(
        index_dir=index_dir,
        question="What is the person doing in this video?",
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nTIMESTAMPS:")
    for frame in result["frames"]:
        print(f'{frame["timestamp"]:.2f}s')


if __name__ == "__main__":
    main()
