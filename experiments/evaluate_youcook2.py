import json
import csv
import numpy as np

from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever


VIDEO_ID = "--bv0V6ZjWI"

DATA_DIR = Path("data/processed/sample_long")
ANNOTATION_FILE = Path("data/annotations/youcook2.json")
OUTPUT_DIR = Path("results/youcook2")

K_VALUES = [1, 3, 5, 10]

def random_hit_probability(
    timestamps,
    start,
    end,
    k,
):
    """
    Exact probability of retrieving at least one relevant
    frame when sampling k distinct frames uniformly.
    """

    n = len(timestamps)

    relevant = sum(
        start <= t <= end
        for t in timestamps
    )

    k = min(k, n)

    if relevant == 0:
        return 0.0

    if k > n - relevant:
        return 1.0

    probability_no_hit = 1.0

    for i in range(k):
        probability_no_hit *= (
            (n - relevant - i) / (n - i)
        )

    return 1.0 - probability_no_hit


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load annotations
    # --------------------------------------------------

    with open(ANNOTATION_FILE, encoding="utf-8") as f:
        database = json.load(f)["database"]

    if VIDEO_ID not in database:
        raise ValueError(
            f"Video {VIDEO_ID} not found in annotations"
        )

    video_data = database[VIDEO_ID]
    annotations = video_data["annotations"]

    print(f"Video ID: {VIDEO_ID}")
    print(f"Subset: {video_data['subset']}")
    print(f"Duration: {video_data['duration']} seconds")
    print(f"Annotated events: {len(annotations)}")

    # --------------------------------------------------
    # 2. Load retrieval system
    # --------------------------------------------------

    encoder = CLIPEncoder()

    retriever = FAISSRetriever()
    retriever.load(DATA_DIR)

    # --------------------------------------------------
    # 3. Evaluate each annotated event
    # --------------------------------------------------

    results = []

    for annotation in annotations:

        query = annotation["sentence"]

        start, end = annotation["segment"]

        query_embedding = encoder.encode_text([query])

        retrieved = retriever.search(
            query_embedding,
            top_k=max(K_VALUES)
        )

        row = {
            "video_id": VIDEO_ID,
            "event_id": annotation["id"],
            "query": query,
            "start": start,
            "end": end,
        }

        for k in K_VALUES:

            top_k = retrieved[:k]

            hit = any(
                start <= frame["timestamp"] <= end
                for frame in top_k
            )

            tolerance = 2.0

            hit_tolerant = any(
                start - tolerance <= frame["timestamp"] <= end + tolerance
                for frame in top_k
            )

            row[f"hit@{k}"] = int(hit)
            row[f"hit@{k}_tol2s"] = int(hit_tolerant)

        row["top1_timestamp"] = (
            retrieved[0]["timestamp"]
            if retrieved else None
        )

        row["top1_similarity"] = (
            retrieved[0]["similarity"]
            if retrieved else None
        )

        results.append(row)

        print(f"\nEvent {annotation['id']}: {query}")
        print(f"Ground truth: {start}–{end}s")
        print(
            "Retrieved:",
            [round(x["timestamp"], 1) for x in retrieved[:5]]
        )

        for k in K_VALUES:
            print(f"Hit@{k}: {row[f'hit@{k}']}")

    # --------------------------------------------------
    # 4. Aggregate metrics
    # --------------------------------------------------

    print("\n========== RESULTS ==========")

    summary = {
        "video_id": VIDEO_ID,
        "num_events": len(results),
    }

    for k in K_VALUES:

        key = f"hit@{k}"

        recall = np.mean([
            row[key] for row in results
        ])

        summary[f"recall@{k}"] = float(recall)

        print(f"Recall@{k}: {recall:.4f}")

    print("\n=== Boundary-Tolerant Results (±2 seconds) ===")

    for k in K_VALUES:
        key = f"hit@{k}_tol2s"

        recall = np.mean([
            row[key] for row in results
        ])

        summary[f"recall@{k}_tol2s"] = float(recall)

        print(f"Recall@{k} (±2s): {recall:.4f}")

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------

    csv_path = OUTPUT_DIR / f"{VIDEO_ID}_retrieval.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / f"{VIDEO_ID}_summary.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()