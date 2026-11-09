from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.temporal_retriever import TemporalRetriever
from src.reasoning.vlm import VisionLanguageModel
import numpy as np


def main():

    data_dir = Path("data/processed/sample_long")

    print("Loading CLIP...")
    encoder = CLIPEncoder()

    print("Loading FAISS...")
    retriever = FAISSRetriever()
    retriever.load(data_dir)

    temporal_retriever = TemporalRetriever(
        retriever
    )

    print("Loading VLM...")
    vlm = VisionLanguageModel()

    while True:

        question = input(
            "\nAsk a video question (or 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # Encode query.
        query_embedding = encoder.encode_text(
            [question]
        )

        # Retrieve relevant frames + temporal context.
        frames = temporal_retriever.search_with_context(
            query_embedding,
            top_k=2,
            context_before=2,
            context_after=2
        )

        max_frames = 8

        if len(frames) > max_frames:
            indices = np.linspace(
                0,
                len(frames) - 1,
                max_frames,
                dtype=int
            )

            frames = [frames[i] for i in indices]

        print("\nRetrieved timestamps:")

        for frame in frames:
            print(
                f"  {frame['timestamp']:.2f}s"
            )

        # Generate answer from multiple frames.
        answer = vlm.answer_video_frames(
            frames=frames,
            question=question
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()