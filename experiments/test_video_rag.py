from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.reasoning.vlm import VisionLanguageModel


def main():

    data_dir = Path("data/processed/sample")

    print("Loading retrieval models...")

    encoder = CLIPEncoder()

    retriever = FAISSRetriever()
    retriever.load(data_dir)

    print("Loading VLM...")

    vlm = VisionLanguageModel()

    while True:

        question = input(
            "\nAsk a question about the video "
            "(or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # Step 1: Encode the question.
        query_embedding = encoder.encode_text(
            [question]
        )

        # Step 2: Retrieve the most relevant frame.
        results = retriever.search(
            query_embedding,
            top_k=1,
        )

        if not results:
            print("No relevant frames found.")
            continue

        best_frame = results[0]

        print(
            f"\nRetrieved frame at "
            f"{best_frame['timestamp']:.2f}s"
        )

        print(
            f"CLIP similarity: "
            f"{best_frame['similarity']:.4f}"
        )

        # Step 3: Ask the VLM about the retrieved frame.
        answer = vlm.answer_image(
            image_path=best_frame["image_path"],
            question=question,
        )

        print("\nAnswer:", answer)

        print(
            f"Evidence timestamp: "
            f"{best_frame['timestamp']:.2f}s"
        )


if __name__ == "__main__":
    main()