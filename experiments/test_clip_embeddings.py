import json
import numpy as np

from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder


def main():

    metadata_path = Path(
        "data/processed/sample/metadata.json"
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_paths = [
        item["image_path"]
        for item in metadata
    ]

    encoder = CLIPEncoder()

    # Generate image embeddings
    image_embeddings = encoder.encode_images(
        image_paths,
        batch_size=16
    )

    print("Image embeddings shape:", image_embeddings.shape)

    # Save embeddings
    output_path = Path(
        "data/processed/sample/clip_embeddings.npy"
    )

    np.save(output_path, image_embeddings)

    print(f"Saved embeddings to {output_path}")

    # Example natural-language queries
    queries = [
        "A person climbing a ladder",
        "A person carrying an umbrella",
        "A car driving on the road",
    ]

    text_embeddings = encoder.encode_text(queries)

    print("Text embeddings shape:", text_embeddings.shape)

    # Compute cosine similarity
    similarities = text_embeddings @ image_embeddings.T

    for query_idx, query in enumerate(queries):

        print(f"\nQuery: {query}")

        top_indices = np.argsort(
            similarities[query_idx]
        )[::-1][:5]

        for idx in top_indices:

            item = metadata[idx]

            print(
                f"Time: {item['timestamp']:.2f}s | "
                f"Similarity: {similarities[query_idx, idx]:.4f} | "
                f"Frame: {item['image_path']}"
            )


if __name__ == "__main__":
    main()