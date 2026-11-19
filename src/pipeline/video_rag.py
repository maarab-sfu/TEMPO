from pathlib import Path
import shutil
import uuid

from src.perception.video_processor import VideoProcessor
from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.temporal_retriever import TemporalRetriever
from src.reasoning.vlm import VisionLanguageModel
from src.storage.index_store import IndexStore


class VideoRAGPipeline:
    def __init__(self):
        # Load CLIP once. Qwen is loaded lazily.
        self.encoder = CLIPEncoder()
        self.vlm = None

    def index_video(
        self,
        video_path,
        output_dir,
        sampling_interval=2.0,
        max_resolution=640,
    ):
        video_path = Path(video_path)
        output_dir = Path(output_dir).resolve()

        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        if output_dir.exists():
            raise FileExistsError(
                f"Index already exists: {output_dir}"
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)

        staging_dir = (
            output_dir.parent
            / f".staging_{uuid.uuid4().hex}"
        )

        staging_dir.mkdir()

        try:
            processor = VideoProcessor(
                video_path=str(video_path),
                output_dir=str(staging_dir / "frames"),
                sampling_interval=sampling_interval,
                max_resolution=max_resolution,
            )

            metadata = processor.extract_frames()

            if not metadata:
                raise RuntimeError("No frames extracted.")

            image_paths = [
                frame["image_path"]
                for frame in metadata
            ]

            embeddings = self.encoder.encode_images(
                image_paths,
                batch_size=16,
            )

            retriever = FAISSRetriever(
                embedding_dim=embeddings.shape[1]
            )

            retriever.build_index(embeddings, metadata)

            store = IndexStore(staging_dir)

            store.save(
                retriever=retriever,
                metadata=metadata,
                video_id=output_dir.name,
                source_filename=video_path.name,
                sampling_interval=sampling_interval,
                max_resolution=max_resolution,
            )

            # Validate before publication.
            validated_retriever, manifest = store.load()

            if validated_retriever.index.ntotal == 0:
                raise RuntimeError("Validated index is empty.")

            # Publish on the same filesystem.
            # Destination must not already exist.
            staging_dir.rename(output_dir)

            return {
                "frame_count": manifest["frame_count"],
                "index_dir": str(output_dir),
                "embedding_dimension": manifest["embedding_dimension"],
            }

        finally:
            # Clean up failed or abandoned staging directories.
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def query(
        self,
        index_dir,
        question,
        top_k=2,
        context_before=2,
        context_after=2,
        max_frames=8,
    ):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        index_dir = Path(index_dir)

        store = IndexStore(index_dir)
        retriever, manifest = store.load()

        temporal_retriever = TemporalRetriever(retriever)

        query_embedding = self.encoder.encode_text(
            [question]
        )

        frames = temporal_retriever.search_with_context(
            query_embedding,
            top_k=top_k,
            context_before=context_before,
            context_after=context_after,
            max_frames=max_frames,
        )

        if not frames:
            raise RuntimeError("No frames retrieved.")

        if self.vlm is None:
            self.vlm = VisionLanguageModel()

        grounded = self.vlm.answer_video_frames_grounded(
            frames=frames,
            question=question,
        )

        return {
            "question": question,
            "answer": grounded["answer"],
            "cited_frame_ids": grounded["cited_frame_ids"],
            "grounding_status": grounded["grounding_status"],
            "citation_parse_ok": grounded["citation_parse_ok"],
            "frames": [
                {
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": frame["image_path"],
                    "evidence_role": frame["evidence_role"],
                    "anchor_rank": frame["anchor_rank"],
                    "similarity": frame["similarity"],
                }
                for frame in frames
            ],
        }
