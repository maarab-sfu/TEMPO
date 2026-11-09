import json
import numpy as np

from collections import defaultdict
from pathlib import Path


TRACKS_PATH = Path(
    "data/processed/youcook2/--bv0V6ZjWI/tracks.json"
)


def main():
    with open(TRACKS_PATH, encoding="utf-8") as f:
        frames = json.load(f)

    tracks = defaultdict(list)

    for frame in frames:
        for detection in frame["detections"]:
            tracks[detection["track_id"]].append({
                "timestamp": frame["timestamp"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
            })

    lengths = []
    durations = []
    short_tracks = 0

    for track_id, observations in tracks.items():
        timestamps = [
            obs["timestamp"]
            for obs in observations
        ]

        lengths.append(len(observations))

        duration = max(timestamps) - min(timestamps)
        durations.append(duration)

        if len(observations) < 10:
            short_tracks += 1

    print(f"Total track IDs: {len(tracks)}")

    print(
        f"Mean observations per track: "
        f"{np.mean(lengths):.2f}"
    )

    print(
        f"Median observations per track: "
        f"{np.median(lengths):.2f}"
    )

    print(
        f"Mean track duration: "
        f"{np.mean(durations):.2f}s"
    )

    print(
        f"Median track duration: "
        f"{np.median(durations):.2f}s"
    )

    print(
        f"Tracks shorter than 10 observations: "
        f"{short_tracks} "
        f"({100 * short_tracks / len(tracks):.1f}%)"
    )


if __name__ == "__main__":
    main()