from datasets import load_dataset
from pathlib import Path
import requests

# Load metadata containing direct video URLs
ds = load_dataset(
    "TornadoLabs/youcook2",
    "videos",
    split="train"
)

print(f"Available videos: {len(ds)}")

# Select the first video
video = ds[0]

print("Video ID:", video["video_id"])
print("Download URL:", video["video_url"])

# Output directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "sample_long.mp4"

# Download video
with requests.get(
    video["video_url"],
    stream=True,
    timeout=60
) as response:

    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                f.write(chunk)

print(f"Saved to: {output_path}")