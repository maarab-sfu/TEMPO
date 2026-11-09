import json
import numpy as np

from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever


def main():

    data_dir = Path("data/processed/sample")

    embeddings = np.load(
        data_dir / "clip_embeddings.npy"
    )

    with open(
        data_dir / "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    # Build and save the index
    retriever = FAISSRetriever(
        embedding_dim=embeddings.shape[1]
    )

    retriever.build_index(
        embeddings,
        metadata
    )

    retriever.save(data_dir)

    # Verify that the saved index can be loaded
    retriever = FAISSRetriever()
    retriever.load(data_dir)

    # Load CLIP for text encoding
    encoder = CLIPEncoder()

    query_embedding = encoder.encode_text(
        ["A person climbing a ladder"]
    )

    # NumPy baseline
    scores_numpy = query_embedding @ embeddings.T

    top_numpy = np.argsort(
        scores_numpy[0]
    )[::-1][:5]

    # FAISS
    scores_faiss, top_faiss = retriever.index.search(
        np.ascontiguousarray(query_embedding, dtype=np.float32),
        5
    )

    print("NumPy:", top_numpy)
    print("FAISS:", top_faiss[0])

    assert np.array_equal(top_numpy, top_faiss[0])

    # while True:

    #     query = input(
    #         "\nEnter a query (or 'exit'): "
    #     ).strip()

    #     if query.lower() == "exit":
    #         break

    #     if not query:
    #         continue

    #     query_embedding = encoder.encode_text([query])

    #     results = retriever.search(
    #         query_embedding,
    #         top_k=5
    #     )

    #     print(f"\nTop results for: {query}")

    #     for rank, item in enumerate(results, start=1):

    #         print(
    #             f"{rank}. "
    #             f"Time: {item['timestamp']:.2f}s | "
    #             f"Similarity: {item['similarity']:.4f} | "
    #             f"Frame: {item['image_path']}"
    #         )


if __name__ == "__main__":
    main()