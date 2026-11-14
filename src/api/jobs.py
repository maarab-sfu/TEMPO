import threading
from datetime import datetime, timezone


class JobRegistry:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self, video_id, filename):
        job = {
            "video_id": video_id,
            "filename": filename,
            "status": "queued",
            "error": None,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        with self._lock:
            self._jobs[video_id] = job

        return job.copy()

    def update(self, video_id, status, error=None):
        with self._lock:
            job = self._jobs[video_id]
            job["status"] = status
            job["error"] = error

    def get(self, video_id):
        with self._lock:
            job = self._jobs.get(video_id)
            return job.copy() if job else None


jobs = JobRegistry()