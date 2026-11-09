import csv
import json
import random
from pathlib import Path

import requests


ANNOTATIONS_FILE = Path("data/annotations/youcook2.json")
OUTPUT_DIR = Path("data/raw/youcook2")
MANIFEST_FILE = Path("data/annotations/youcook2_batch_manifest.csv")

BASE_URL = (
    "https://huggingface.co/buckets/TornadoLabs/staging/"
    "resolve/videos/youcook2"
)

TARGET_NEW_VIDEOS = 19
RANDOM_SEED = 42
TIMEOUT = (20, 120)


def download_video(video_id):
    destination = OUTPUT_DIR / f"{video_id}.mp4"
    temporary = OUTPUT_DIR / f"{video_id}.mp4.part"

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {video_id}")
        return True

    url = f"{BASE_URL}/{video_id}.mp4"

    try:
        with requests.get(
            url,
            stream=True,
            timeout=TIMEOUT,
        ) as response:

            response.raise_for_status()

            with open(temporary, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        if temporary.stat().st_size == 0:
            raise RuntimeError("Empty download")

        temporary.replace(destination)

        print(
            f"Downloaded {video_id}: "
            f"{destination.stat().st_size / 1e6:.1f} MB"
        )

        return True

    except (requests.RequestException, OSError, RuntimeError) as exc:
        print(f"Failed {video_id}: {exc}")
        temporary.unlink(missing_ok=True)
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
        database = json.load(f)["database"]

    candidates = [
        video_id
        for video_id, info in database.items()
        if info["subset"] == "validation"
        and len(info["annotations"]) >= 5
    ]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(candidates)

    print(f"Eligible validation videos: {len(candidates)}")

    successful = []
    attempted = []

    for video_id in candidates:
        if len(successful) >= TARGET_NEW_VIDEOS:
            break

        print(
            f"\n[{len(successful) + 1}/{TARGET_NEW_VIDEOS}] "
            f"{video_id}"
        )

        ok = download_video(video_id)

        attempted.append({
            "video_id": video_id,
            "subset": database[video_id]["subset"],
            "num_events": len(database[video_id]["annotations"]),
            "downloaded": ok,
        })

        if ok:
            successful.append(video_id)

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "subset",
                "num_events",
                "downloaded",
            ],
        )
        writer.writeheader()
        writer.writerows(attempted)

    print("\n========== DOWNLOAD SUMMARY ==========")
    print(f"Downloaded: {len(successful)}")
    print(f"Failed attempts: {len(attempted) - len(successful)}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()