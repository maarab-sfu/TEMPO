import json
import numpy as np
import cv2

from pathlib import Path


VIDEO_ID = "--bv0V6ZjWI"

DATA_DIR = Path(f"data/processed/youcook2/{VIDEO_ID}")
OUTPUT_DIR = Path("results/youcook2/tracklet_pairs")

TOP_K = 20


def load_data():
    data = np.load(DATA_DIR / "tracklet_embeddings.npz")

    track_ids = data["track_ids"]
    embeddings = data["embeddings"]

    with open(
        DATA_DIR / "tracklet_metadata.json",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return track_ids, embeddings, metadata


def load_crop(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(path)

    return image


def resize_with_padding(image, size=256):
    height, width = image.shape[:2]

    scale = min(size / width, size / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
    )

    canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    x = (size - new_width) // 2
    y = (size - new_height) // 2

    canvas[y:y + new_height, x:x + new_width] = resized

    return canvas


def create_pair_image(track_a, track_b, similarity, metadata):
    info_a = metadata[str(track_a)]
    info_b = metadata[str(track_b)]

    crop_a = load_crop(info_a["crop_paths"][2])
    crop_b = load_crop(info_b["crop_paths"][2])

    crop_a = resize_with_padding(crop_a)
    crop_b = resize_with_padding(crop_b)

    combined = np.hstack([crop_a, crop_b])

    header = np.zeros(
        (80, combined.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        f"Tracks {track_a} / {track_b} | Similarity: {similarity:.4f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        header,
        f"{info_a['class_name']} / {info_b['class_name']}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    return np.vstack([header, combined])

def temporal_gap(info_a, info_b):
    if info_a["end"] < info_b["start"]:
        return info_b["start"] - info_a["end"]

    if info_b["end"] < info_a["start"]:
        return info_a["start"] - info_b["end"]

    return 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    track_ids, embeddings, metadata = load_data()

    similarities = embeddings @ embeddings.T

    # Exclude self-comparisons.
    np.fill_diagonal(similarities, -np.inf)

    pairs = []

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            pairs.append((
                similarities[i, j],
                int(track_ids[i]),
                int(track_ids[j]),
            ))

    pairs.sort(reverse=True)

    print(f"Total candidate pairs: {len(pairs)}")

    for rank, (similarity, track_a, track_b) in enumerate(
        pairs[:TOP_K], start=1
    ):
        info_a = metadata[str(track_a)]
        info_b = metadata[str(track_b)]

        overlap = max(
            0,
            min(info_a["end"], info_b["end"])
            - max(info_a["start"], info_b["start"]),
        )

        gap = temporal_gap(info_a, info_b)

        print(
            f"{rank:02d}. "
            f"Tracks {track_a} / {track_b} | "
            f"Similarity={similarity:.4f} | "
            f"Gap={gap:.2f}s | "
            f"A=[{info_a['start']:.1f}, {info_a['end']:.1f}] | "
            f"B=[{info_b['start']:.1f}, {info_b['end']:.1f}]"
        )

        image = create_pair_image(
            track_a,
            track_b,
            similarity,
            metadata,
        )

        output_path = OUTPUT_DIR / f"pair_{rank:02d}.jpg"

        cv2.imwrite(str(output_path), image)

    print(f"\nSaved images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()