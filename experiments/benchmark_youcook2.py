import csv
import json
import numpy as np

from pathlib import Path

from src.perception.video_processor import VideoProcessor
from src.embeddings.clip_encoder import CLIPEncoder


# Configuration
RAW_DIR = Path("data/raw/youcook2")
PROCESSED_DIR = Path("data/processed/youcook2")
ANNOTATIONS_FILE = Path("data/annotations/youcook2.json")
RESULTS_DIR = Path("results/youcook2/batch")

MAX_VIDEOS = 20
SAMPLING_INTERVAL = 2.0
K_VALUES = (1, 3, 5, 10)
TOLERANCE = 2.0


def normalize(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    return embeddings / np.maximum(norms, 1e-12)


def random_hit_probability(timestamps, start, end, k):
    """
    Exact expected Hit@K for uniformly sampled
    distinct frames.
    """
    n = len(timestamps)

    if n == 0:
        return 0.0

    relevant = np.count_nonzero(
        (timestamps >= start) & (timestamps <= end)
    )

    k = min(k, n)

    if relevant == 0:
        return 0.0

    if k > n - relevant:
        return 1.0

    p_no_hit = 1.0

    for i in range(k):
        p_no_hit *= (n - relevant - i) / (n - i)

    return 1.0 - p_no_hit


def prepare_video(video_path, video_id, encoder):
    """
    Extract frames and compute embeddings.
    Reuse cached results when available.
    """
    output_dir = PROCESSED_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    embeddings_path = output_dir / "clip_embeddings.npy"

    if not metadata_path.exists():
        processor = VideoProcessor(
            video_path=str(video_path),
            output_dir=str(output_dir),
            sampling_interval=SAMPLING_INTERVAL,
            max_resolution=640,
        )

        metadata = processor.extract_frames()
    else:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    if not metadata:
        raise RuntimeError(f"No frames extracted: {video_id}")

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        image_paths = [
            frame["image_path"]
            for frame in metadata
        ]

        embeddings = encoder.encode_images(
            image_paths,
            batch_size=16,
        )

        embeddings = normalize(embeddings)
        np.save(embeddings_path, embeddings)

    if len(metadata) != len(embeddings):
        raise RuntimeError(
            f"Metadata/embedding mismatch: {video_id}"
        )

    timestamps = np.array(
        [frame["timestamp"] for frame in metadata],
        dtype=np.float64,
    )

    return timestamps, normalize(embeddings)


def evaluate_video(video_id, video_path, video_data, encoder):
    timestamps, image_embeddings = prepare_video(
        video_path, video_id, encoder
    )

    annotations = video_data["annotations"]

    if not annotations:
        return []

    queries = [
        annotation["sentence"]
        for annotation in annotations
    ]

    text_embeddings = normalize(
        encoder.encode_text(queries)
    )

    # CLIP cosine similarity, since both sets
    # of embeddings are L2-normalized.
    similarities = text_embeddings @ image_embeddings.T

    rows = []

    for annotation, scores in zip(annotations, similarities):
        start, end = annotation["segment"]

        ranking = np.argsort(-scores)

        row = {
            "video_id": video_id,
            "subset": video_data["subset"],
            "event_id": annotation["id"],
            "query": annotation["sentence"],
            "start": start,
            "end": end,
            "num_frames": len(timestamps),
            "top1_timestamp": float(timestamps[ranking[0]]),
        }

        for k in K_VALUES:
            top_indices = ranking[:k]
            retrieved_times = timestamps[top_indices]

            strict_hit = np.any(
                (retrieved_times >= start)
                & (retrieved_times <= end)
            )

            tolerant_hit = np.any(
                (retrieved_times >= start - TOLERANCE)
                & (retrieved_times <= end + TOLERANCE)
            )

            row[f"hit@{k}"] = int(strict_hit)
            row[f"hit@{k}_tol2s"] = int(tolerant_hit)

            row[f"random@{k}"] = random_hit_probability(
                timestamps, start, end, k
            )

        rows.append(row)

    return rows


def aggregate(rows):
    """
    Macro-average over videos:
    first average events within each video,
    then average the video-level scores.
    """
    video_ids = sorted(set(row["video_id"] for row in rows))

    summary = {
        "num_videos": len(video_ids),
        "num_events": len(rows),
    }

    for k in K_VALUES:
        for metric in (
            f"hit@{k}",
            f"hit@{k}_tol2s",
            f"random@{k}",
        ):
            per_video = []

            for video_id in video_ids:
                values = [
                    row[metric]
                    for row in rows
                    if row["video_id"] == video_id
                ]

                per_video.append(np.mean(values))

            summary[metric] = float(np.mean(per_video))

    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        database = json.load(f)["database"]

    video_paths = sorted(RAW_DIR.glob("*.mp4"))

    # Only include videos with annotations.
    video_paths = [
        path for path in video_paths
        if path.stem in database
    ]

    video_paths = video_paths[:MAX_VIDEOS]

    if not video_paths:
        raise RuntimeError(
            "No annotated MP4 files found in data/raw/youcook2"
        )

    print(f"Found {len(video_paths)} annotated videos")

    encoder = CLIPEncoder()

    all_rows = []

    for i, video_path in enumerate(video_paths, 1):
        video_id = video_path.stem

        print(
            f"\n[{i}/{len(video_paths)}] "
            f"Processing {video_id}"
        )

        try:
            rows = evaluate_video(
                video_id,
                video_path,
                database[video_id],
                encoder,
            )

            all_rows.extend(rows)

            print(f"Evaluated {len(rows)} events")

        except Exception as exc:
            print(f"FAILED: {video_id}: {exc}")

    if not all_rows:
        raise RuntimeError("No videos evaluated successfully")

    summary = aggregate(all_rows)

    validation_rows = [
        row for row in all_rows
        if row["subset"] == "validation"
    ]

    validation_summary = (
        aggregate(validation_rows)
        if validation_rows else None
    )

    csv_path = RESULTS_DIR / "event_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = RESULTS_DIR / "summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========== BATCH RESULTS ==========")
    print(f"Videos: {summary['num_videos']}")
    print(f"Events: {summary['num_events']}")

    for k in K_VALUES:
        print(
            f"Recall@{k}: {summary[f'hit@{k}']:.4f} | "
            f"Tolerant: {summary[f'hit@{k}_tol2s']:.4f} | "
            f"Random: {summary[f'random@{k}']:.4f}"
        )

    print(f"\nSaved results to {RESULTS_DIR}")

    if validation_summary is not None:
        print("\n========== VALIDATION RESULTS ==========")
        print(f"Videos: {validation_summary['num_videos']}")
        print(f"Events: {validation_summary['num_events']}")

        for k in K_VALUES:
            print(
                f"Recall@{k}: "
                f"{validation_summary[f'hit@{k}']:.4f} | "
                f"Tolerant: "
                f"{validation_summary[f'hit@{k}_tol2s']:.4f} | "
                f"Random: "
                f"{validation_summary[f'random@{k}']:.4f}"
            )

        with open(
            RESULTS_DIR / "validation_summary.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(validation_summary, f, indent=2)


if __name__ == "__main__":
    main()