import subprocess
from pathlib import Path


def convert_for_browser(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    if not output_path.is_file():
        raise RuntimeError("Video conversion produced no file")

    return output_path