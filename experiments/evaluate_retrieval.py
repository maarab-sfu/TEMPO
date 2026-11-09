import json
import numpy as np

from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever


def evaluate_query(
    retriever,
    encoder,
    query,
    ground_truth_start,
    ground_truth_end,
    k_values=(1, 3, 5, 10),
):
    query_embedding = encoder.encode_text([query])

    results = retriever.search(
        query_embedding,
        top_k=max(k_values)
    )

    metrics = {}

    for k in k_values:
        top_k_results = results[:k]

        hit = any(
            ground_truth_start <= item["timestamp"] <= ground_truth_end
            for item in top_k_results
        )

        metrics[f"hit@{k}"] = int(hit)

    return metrics, results


def main():

    data_dir = Path("data/processed/sample_long")

    encoder = CLIPEncoder()

    retriever = FAISSRetriever()
    retriever.load(data_dir)

    # Replace these with actual events and timestamps
    # observed in your YouCook2 video.
    annotations = [
        {
            "query": "The person adds shredded cheese",
            "start": 144.0,
            "end": 164.0,
        },
        {
            "query": "The person adds tomotoes",
            "start": 190.0,
            "end": 196.0,
        },
        {
            "query": "The person puts the pizza in the oven",
            "start": 240.0,
            "end": 244.0,
        },
    ]

    all_metrics = []

    for annotation in annotations:

        metrics, results = evaluate_query(
            retriever,
            encoder,
            annotation["query"],
            annotation["start"],
            annotation["end"],
        )

        all_metrics.append(metrics)

        print(f"\nQuery: {annotation['query']}")
        print(f"Ground truth: {annotation['start']}–{annotation['end']}s")

        for item in results[:5]:
            print(
                f"  {item['timestamp']:.1f}s "
                f"(score={item['similarity']:.3f})"
            )

        print("Metrics:", metrics)

    print("\n=== Overall Results ===")

    for k in (1, 3, 5, 10):

        key = f"hit@{k}"

        recall = np.mean([
            m[key] for m in all_metrics
        ])

        print(f"Recall@{k}: {recall:.3f}")


if __name__ == "__main__":
    main()