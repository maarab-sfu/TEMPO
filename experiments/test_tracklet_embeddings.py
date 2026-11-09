import json
import numpy as np

from pathlib import Path

from src.embeddings.clip_encoder import CLIPEncoder
from src.perception.tracklet_encoder import TrackletEncoder


VIDEO_ID = "--bv0V6ZjWI"

VIDEO_PATH = Path(
    f"data/raw/youcook2/{VIDEO_ID}.mp4"
)

OUTPUT_DIR = Path(
    f"data/processed/youcook2/{VIDEO_ID}"
)


def main():

    with open(
        OUTPUT_DIR / "tracks.json",
        encoding="utf-8",
    ) as f:
        tracking_data = json.load(f)

    clip_encoder = CLIPEncoder()

    tracklet_encoder = TrackletEncoder(
        clip_encoder=clip_encoder,
        samples_per_track=5,
        min_track_length=10,
    )

    embeddings, metadata = tracklet_encoder.encode_tracklets(
        video_path=VIDEO_PATH,
        tracking_data=tracking_data,
        output_dir=OUTPUT_DIR,
    )

    track_ids = sorted(embeddings.keys())

    embedding_matrix = np.stack([
        embeddings[track_id]
        for track_id in track_ids
    ])

    np.savez_compressed(
        OUTPUT_DIR / "tracklet_embeddings.npz",
        track_ids=np.array(track_ids),
        embeddings=embedding_matrix,
    )

    with open(
        OUTPUT_DIR / "tracklet_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(f"Encoded tracklets: {len(track_ids)}")
    print(f"Embedding shape: {embedding_matrix.shape}")

    # Compute tracklet-to-tracklet similarity.
    similarity = embedding_matrix @ embedding_matrix.T

    np.fill_diagonal(similarity, -1)

    # Find the most similar pair.
    i, j = np.unravel_index(
        np.argmax(similarity),
        similarity.shape,
    )

    print("\nMost similar tracklets:")
    print(f"Track {track_ids[i]} and Track {track_ids[j]}")
    print(f"Cosine similarity: {similarity[i, j]:.4f}")

    print("Track A:", metadata[track_ids[i]])
    print("Track B:", metadata[track_ids[j]])


if __name__ == "__main__":
    main()