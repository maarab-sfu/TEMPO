import csv
import json
import cv2
import numpy as np

from pathlib import Path


VIDEO_ID = "--bv0V6ZjWI"

DATA_DIR = Path(f"data/processed/youcook2/{VIDEO_ID}")
ASSOCIATION_DIR = Path("results/youcook2/association")

CSV_PATH = ASSOCIATION_DIR / "candidate_pairs.csv"
OUTPUT_DIR = ASSOCIATION_DIR / "contact_sheets"

PAIRS_PER_SHEET = 10
CROP_SIZE = 180


def load_crop(path):
    image = cv2.imread(str(path))

    if image is None:
        return np.zeros(
            (CROP_SIZE, CROP_SIZE, 3),
            dtype=np.uint8,
        )

    h, w = image.shape[:2]

    scale = min(CROP_SIZE / w, CROP_SIZE / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros(
        (CROP_SIZE, CROP_SIZE, 3),
        dtype=np.uint8,
    )

    x = (CROP_SIZE - new_w) // 2
    y = (CROP_SIZE - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    with open(
        DATA_DIR / "tracklet_metadata.json",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    row_height = 230
    row_width = 800

    for start in range(0, len(pairs), PAIRS_PER_SHEET):

        batch = pairs[start:start + PAIRS_PER_SHEET]

        sheet = np.full(
            (row_height * len(batch), row_width, 3),
            35,
            dtype=np.uint8,
        )

        for local_idx, pair in enumerate(batch):

            global_idx = start + local_idx

            track_a = pair["track_a"]
            track_b = pair["track_b"]

            info_a = metadata[track_a]
            info_b = metadata[track_b]

            crop_a = load_crop(info_a["crop_paths"][2])
            crop_b = load_crop(info_b["crop_paths"][2])

            y = local_idx * row_height

            sheet[
                y + 35:y + 35 + CROP_SIZE,
                10:10 + CROP_SIZE,
            ] = crop_a

            sheet[
                y + 35:y + 35 + CROP_SIZE,
                210:210 + CROP_SIZE,
            ] = crop_b

            text_lines = [
                f"Row {global_idx + 1}",
                f"Tracks: {track_a} / {track_b}",
                f"Similarity: {float(pair['similarity']):.4f}",
                f"Gap: {float(pair['gap']):.2f}s",
                f"Classes: {pair['class_a']} / {pair['class_b']}",
                "Label: 1=same, 0=different, -1=unknown",
            ]

            for line_idx, text in enumerate(text_lines):

                cv2.putText(
                    sheet,
                    text,
                    (410, y + 45 + line_idx * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (255, 255, 255),
                    1,
                )

        output_path = OUTPUT_DIR / (
            f"sheet_{start // PAIRS_PER_SHEET + 1:03d}.jpg"
        )

        cv2.imwrite(str(output_path), sheet)

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()