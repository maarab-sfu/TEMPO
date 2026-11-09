import json
import numpy as np

from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever


def main():
    data_dir = Path("data/processed/sample_long")

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    image_paths = [
        frame["image_path"]
        for frame in metadata
    ]

    encoder = CLIPEncoder()

    embeddings = encoder.encode_images(
        image_paths,
        batch_size=16
    )

    print("Embeddings shape:", embeddings.shape)

    np.save(
        data_dir / "clip_embeddings.npy",
        embeddings
    )

    retriever = FAISSRetriever(
        embedding_dim=embeddings.shape[1]
    )

    retriever.build_index(
        embeddings,
        metadata
    )

    retriever.save(data_dir)

    print(f"Indexed {len(metadata)} frames")


if __name__ == "__main__":
    main()