import json
from pathlib import Path

from fastapi import FastAPI
import threading

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.pipeline.video_rag import VideoRAGPipeline

import shutil
import uuid

from fastapi import BackgroundTasks, File, UploadFile

from src.api.jobs import jobs
from fastapi.responses import FileResponse
from src.storage.index_store import IndexStore

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.media.video_converter import convert_for_browser


app = FastAPI(
    title="VideoMind API",
    description="Multimodal Video Retrieval and Question Answering",
    version="0.1.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes"

UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

PLAYBACK_ROOT = PROJECT_ROOT / "data" / "playback"
PLAYBACK_ROOT.mkdir(parents=True, exist_ok=True)

FRONTEND_ROOT = PROJECT_ROOT / "src" / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_ROOT),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
def frontend():
    return (
        FRONTEND_ROOT / "index.html"
    ).read_text(encoding="utf-8")

# Initialize the pipeline only when the first query arrives.
_pipeline = None
_pipeline_lock = threading.Lock()

# Prevent concurrent GPU inference in this initial version.
_query_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=2, ge=1, le=10)
    context_before: int = Field(default=2, ge=0, le=10)
    context_after: int = Field(default=2, ge=0, le=10)


def get_pipeline():
    global _pipeline

    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = VideoRAGPipeline()

        return _pipeline


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VideoMind",
        "version": "0.1.0",
    }


@app.get("/api/videos")
def list_videos():
    videos = []

    if not INDEX_ROOT.exists():
        return {"videos": []}

    for index_dir in sorted(INDEX_ROOT.iterdir()):
        if not index_dir.is_dir():
            continue

        manifest_path = index_dir / "manifest.json"

        if not manifest_path.is_file():
            continue

        try:
            with manifest_path.open(
                "r", encoding="utf-8"
            ) as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            continue

        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "ready"
            or manifest.get("video_id") != index_dir.name
            or not (index_dir / "index.faiss").is_file()
            or not (index_dir / "index_metadata.json").is_file()
        ):
            continue

        videos.append({
            "video_id": manifest["video_id"],
            "source_filename": manifest.get("source_filename"),
            "frame_count": manifest.get("frame_count"),
            "sampling_interval": manifest.get("sampling_interval"),
            "status": manifest["status"],
        })

    return {"videos": videos}

@app.post("/api/videos/{video_id}/query")
def query_video(video_id: str, request: QueryRequest):
    # Only allow simple video identifiers.
    if (
        not video_id
        or not all(
            c.isalnum() or c in "-_"
            for c in video_id
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid video ID",
        )

    index_dir = INDEX_ROOT / video_id

    if not (index_dir / "manifest.json").is_file():
        raise HTTPException(
            status_code=404,
            detail="Video index not found",
        )

    try:
        with _query_lock:
            pipeline = get_pipeline()

            result = pipeline.query(
                index_dir=index_dir,
                question=request.question,
                top_k=request.top_k,
                context_before=request.context_before,
                context_after=request.context_after,
            )

        return {
            "video_id": video_id,
            **result,
        }

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )


def index_uploaded_video(video_id: str, video_path: Path):
    jobs.update(video_id, "processing")

    playback_path = PLAYBACK_ROOT / f"{video_id}.mp4"

    try:
        # Generate a browser-compatible playback copy.
        convert_for_browser(
            input_path=video_path,
            output_path=playback_path,
        )

        # Build the AI index from the original video.
        with _query_lock:
            pipeline = get_pipeline()

            pipeline.index_video(
                video_path=video_path,
                output_dir=INDEX_ROOT / video_id,
                sampling_interval=2.0,
            )

        jobs.update(video_id, "ready")

    except Exception as exc:
        jobs.update(video_id, "failed", str(exc))


@app.post("/api/videos", status_code=202)
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() not in {
        ".mp4", ".mov", ".mkv", ".avi", ".webm"
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format",
        )

    video_id = uuid.uuid4().hex[:12]
    video_path = UPLOAD_ROOT / f"{video_id}{Path(filename).suffix.lower()}"

    try:
        with video_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
    finally:
        file.file.close()

    job = jobs.create(video_id, filename)

    background_tasks.add_task(
        index_uploaded_video,
        video_id,
        video_path,
    )

    return job


@app.get("/api/videos/{video_id}/status")
def get_video_status(video_id: str):
    job = jobs.get(video_id)

    if job is not None:
        return job

    index_dir = INDEX_ROOT / video_id
    manifest_path = index_dir / "manifest.json"

    if manifest_path.is_file():
        with manifest_path.open(
            "r", encoding="utf-8"
        ) as f:
            manifest = json.load(f)

        if manifest.get("status") == "ready":
            return {
                "video_id": video_id,
                "status": "ready",
                "filename": manifest.get("source_filename"),
                "error": None,
            }

    raise HTTPException(
        status_code=404,
        detail="Video not found",
    )

def get_ready_index_dir(video_id: str) -> Path:
    """Find a published, ready index for a valid video ID."""
    if (
        not video_id
        or not all(c.isalnum() or c in "-_" for c in video_id)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid video ID",
        )

    index_dir = INDEX_ROOT / video_id
    manifest_path = index_dir / "manifest.json"

    if not manifest_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video index not found",
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Invalid index manifest",
        )

    if (
        manifest.get("status") != "ready"
        or manifest.get("schema_version") != 1
        or manifest.get("video_id") != video_id
    ):
        raise HTTPException(
            status_code=422,
            detail="Index is not ready or is incompatible",
        )

    return index_dir


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str):
    get_ready_index_dir(video_id)

    playback_path = PLAYBACK_ROOT / f"{video_id}.mp4"

    if playback_path.is_file():
        return FileResponse(
            playback_path,
            media_type="video/mp4",
            content_disposition_type="inline",
        )

    # Compatibility with videos uploaded before conversion existed.
    manifest_path = INDEX_ROOT / video_id / "manifest.json"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    suffix = Path(
        manifest.get("source_filename", "")
    ).suffix.lower()

    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(
            status_code=404,
            detail="Unsupported source video format",
        )

    original_path = UPLOAD_ROOT / f"{video_id}{suffix}"

    if not original_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original video unavailable",
        )

    return FileResponse(
        original_path,
        media_type={
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".avi": "video/x-msvideo",
            ".webm": "video/webm",
        }[suffix],
        content_disposition_type="inline",
    )


@app.get("/api/videos/{video_id}/frames/{frame_id}")
def get_video_frame(video_id: str, frame_id: int):
    index_dir = get_ready_index_dir(video_id)

    try:
        retriever, _ = IndexStore(index_dir).load()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot load video index: {exc}",
        )

    frame = next(
        (
            item for item in retriever.metadata
            if item["frame_id"] == frame_id
        ),
        None,
    )

    if frame is None:
        raise HTTPException(
            status_code=404,
            detail="Frame not found",
        )

    return FileResponse(
        frame["image_path"],
        media_type="image/jpeg",
    )


# M5.4 experimental object-aware endpoint
from src.api.object_query import router as object_query_router
app.include_router(object_query_router)
