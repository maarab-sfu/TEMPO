import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def convert_for_browser(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        # H.264 with yuv420p needs even dimensions. Preserve all source pixels.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
    ]

    with TemporaryDirectory(dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        try:
            subprocess.run(
                [*command, str(staged)],
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[-2000:] or f"exit code {exc.returncode}"
            raise RuntimeError(f"Video conversion failed: {detail}") from exc

        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError("Video conversion produced no file")
        staged.replace(output_path)

    return output_path
