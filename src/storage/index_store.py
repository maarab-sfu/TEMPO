import json
from pathlib import Path

from src.retrieval.faiss_retriever import FAISSRetriever


SCHEMA_VERSION = 1
CLIP_MODEL = "openai/clip-vit-base-patch32"


class IndexStore:
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir).resolve()

    def save(
        self,
        retriever,
        metadata,
        video_id,
        source_filename,
        sampling_interval,
        max_resolution,
    ):
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Store portable frame paths.
        portable_metadata = []

        for frame in metadata:
            item = frame.copy()

            image_path = Path(item["image_path"]).resolve()
            relative_path = image_path.relative_to(
                self.index_dir
            )

            item["image_path"] = relative_path.as_posix()
            portable_metadata.append(item)

        # Existing FAISS serialization.
        original_metadata = retriever.metadata

        try:
            retriever.metadata = portable_metadata
            retriever.save(self.index_dir)
        finally:
            retriever.metadata = original_metadata

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "video_id": video_id,
            "source_filename": source_filename,
            "status": "ready",
            "embedding_model": CLIP_MODEL,
            "embedding_dimension": retriever.embedding_dim,
            "frame_count": len(portable_metadata),
            "sampling_interval": sampling_interval,
            "max_resolution": max_resolution,
        }

        with open(
            self.index_dir / "manifest.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def load(self):
        manifest_path = self.index_dir / "manifest.json"

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported index schema version")

        if manifest["embedding_model"] != CLIP_MODEL:
            raise ValueError("Incompatible CLIP model")

        retriever = FAISSRetriever()
        retriever.load(self.index_dir)

        if retriever.embedding_dim != manifest["embedding_dimension"]:
            raise ValueError("Embedding dimension mismatch")

        if retriever.index.ntotal != manifest["frame_count"]:
            raise ValueError("Frame count mismatch")

        # Restore absolute paths for the existing ML modules.
        for frame in retriever.metadata:
            relative_path = Path(frame["image_path"])

            if relative_path.is_absolute():
                raise ValueError("Expected relative frame path")

            resolved_path = (
                self.index_dir / relative_path
            ).resolve()

            if not resolved_path.is_relative_to(self.index_dir):
                raise ValueError("Frame path escapes index directory")

            if not resolved_path.is_file():
                raise FileNotFoundError(resolved_path)

            frame["image_path"] = str(resolved_path)

        return retriever, manifest