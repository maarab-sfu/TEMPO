import faiss
import json
import numpy as np

from pathlib import Path


class FAISSRetriever:

    def __init__(self, embedding_dim=512):

        self.embedding_dim = embedding_dim

        self.index = faiss.IndexFlatIP(embedding_dim)

        self.metadata = []

    def build_index(self, embeddings, metadata):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings and metadata entries must match"
            )

        # Ensure vectors are normalized
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

        self.index.reset()
        self.index.add(embeddings)

        self.metadata = metadata

        print(f"Indexed {self.index.ntotal} frames")

    def search(self, query_embedding, top_k=5):

        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.embedding_dim):
            raise ValueError("Expected one query embedding")

        query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            item = self.metadata[int(idx)].copy()

            item["similarity"] = float(score)

            results.append(item)

        return results

    def save(self, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(output_dir / "index.faiss")
        )

        with open(
            output_dir / "index_metadata.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, output_dir):

        output_dir = Path(output_dir)

        self.index = faiss.read_index(
            str(output_dir / "index.faiss")
        )

        self.embedding_dim = self.index.d

        with open(
            output_dir / "index_metadata.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "Index size and metadata count do not match"
            )