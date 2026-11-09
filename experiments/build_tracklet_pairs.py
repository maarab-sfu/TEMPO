import csv
import json
import numpy as np

from pathlib import Path


VIDEO_ID = "--bv0V6ZjWI"

DATA_DIR = Path(f"data/processed/youcook2/{VIDEO_ID}")
OUTPUT_DIR = Path("results/youcook2/association")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def temporal_gap(a, b):
    if a["end"] < b["start"]:
        return b["start"] - a["end"]

    if b["end"] < a["start"]:
        return a["start"] - b["end"]

    return 0.0


def main():

    data = np.load(DATA_DIR / "tracklet_embeddings.npz")

    track_ids = data["track_ids"]
    embeddings = data["embeddings"]

    with open(
        DATA_DIR / "tracklet_metadata.json",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    similarities = embeddings @ embeddings.T

    candidates = []

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):

            track_a = int(track_ids[i])
            track_b = int(track_ids[j])

            info_a = metadata[str(track_a)]
            info_b = metadata[str(track_b)]

            gap = temporal_gap(info_a, info_b)

            # Exclude overlapping tracks.
            if gap == 0:
                continue

            candidates.append({
                "track_a": track_a,
                "track_b": track_b,
                "similarity": float(similarities[i, j]),
                "gap": gap,
                "class_a": info_a["class_name"],
                "class_b": info_b["class_name"],
                "label": "",
            })

    # High-similarity pairs across different time gaps.
    short_gap = [
        c for c in candidates
        if 0 < c["gap"] <= 2
    ]

    medium_gap = [
        c for c in candidates
        if 2 < c["gap"] <= 15
    ]

    long_gap = [
        c for c in candidates
        if c["gap"] > 15
    ]

    selected = []

    for group in (short_gap, medium_gap, long_gap):
        group.sort(
            key=lambda x: x["similarity"],
            reverse=True,
        )

        selected.extend(group[:30])

    # Add lower-similarity examples for comparison.
    remaining = [
        c for c in candidates
        if c not in selected
    ]

    rng = np.random.default_rng(42)

    if remaining:
        indices = rng.choice(
            len(remaining),
            size=min(30, len(remaining)),
            replace=False,
        )

        selected.extend(remaining[i] for i in indices)

    output_path = OUTPUT_DIR / "candidate_pairs.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=selected[0].keys(),
        )

        writer.writeheader()
        writer.writerows(selected)

    print(f"Selected {len(selected)} candidate pairs")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()