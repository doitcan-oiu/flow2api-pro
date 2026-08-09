"""In-memory video job store for the ``/v1/videos`` protocol.

The OpenAI videos API is asynchronous: ``POST /v1/videos`` returns a job in
``queued`` state and clients poll ``GET /v1/videos/{id}`` until the status is
``completed``, then fetch bytes from ``GET /v1/videos/{id}/content``.

The upstream Flow pipeline is a long-running call, so this store runs the
generation as a background task and tracks lifecycle state. Jobs are kept in
memory only: they expire alongside the cached media files, and a restart simply
drops in-flight jobs (clients see 404 and retry), which matches how the file
cache already behaves.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.logger import debug_logger
from ..core.media_models import (
    VIDEO_STATUS_COMPLETED,
    VIDEO_STATUS_FAILED,
    VIDEO_STATUS_IN_PROGRESS,
    VIDEO_STATUS_QUEUED,
)

# Keep finished jobs around long enough for clients to poll and download.
JOB_RETENTION_SECONDS = 24 * 60 * 60
MAX_JOBS = 500


@dataclass
class VideoJob:
    """Lifecycle state for a single video generation job."""

    id: str
    model: str
    prompt: str
    created_at: int
    status: str = VIDEO_STATUS_QUEUED
    progress: int = 0
    seconds: Optional[str] = None
    size: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    url: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_resource(self) -> Dict[str, Any]:
        """Serialize to the OpenAI ``video`` object shape."""
        payload: Dict[str, Any] = {
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "size": self.size,
            "seconds": self.seconds,
            "remixed_from_video_id": self.remixed_from_video_id,
        }
        if self.status == VIDEO_STATUS_FAILED:
            payload["error"] = {
                "code": self.error_code or "generation_failed",
                "message": self.error_message or "Video generation failed",
            }
        else:
            payload["error"] = None
        return payload


class VideoJobStore:
    """Async-safe registry of video generation jobs."""

    def __init__(self, retention_seconds: int = JOB_RETENTION_SECONDS, max_jobs: int = MAX_JOBS):
        self._jobs: Dict[str, VideoJob] = {}
        self._lock = asyncio.Lock()
        self._retention_seconds = retention_seconds
        self._max_jobs = max_jobs

    async def create(
        self,
        *,
        model: str,
        prompt: str,
        seconds: Optional[str] = None,
        size: Optional[str] = None,
        remixed_from_video_id: Optional[str] = None,
    ) -> VideoJob:
        job = VideoJob(
            id=f"video_{uuid.uuid4().hex}",
            model=model,
            prompt=prompt,
            created_at=int(time.time()),
            seconds=seconds,
            size=size,
            remixed_from_video_id=remixed_from_video_id,
        )
        async with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        return job

    async def get(self, job_id: str) -> Optional[VideoJob]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if self._is_expired(job):
                self._jobs.pop(job_id, None)
                return None
            return job

    async def list(self, limit: int = 20, order: str = "desc") -> List[VideoJob]:
        async with self._lock:
            self._purge_expired_locked()
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=(order != "asc"),
            )
        if limit > 0:
            jobs = jobs[:limit]
        return jobs

    async def delete(self, job_id: str) -> Optional[VideoJob]:
        async with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None and job.task is not None and not job.task.done():
            job.task.cancel()
        return job

    def mark_running(self, job: VideoJob, progress: int = 10) -> None:
        if job.status == VIDEO_STATUS_QUEUED:
            job.status = VIDEO_STATUS_IN_PROGRESS
        job.progress = max(job.progress, progress)

    def mark_completed(self, job: VideoJob, url: str) -> None:
        job.status = VIDEO_STATUS_COMPLETED
        job.progress = 100
        job.url = url
        job.completed_at = int(time.time())
        job.expires_at = job.completed_at + self._retention_seconds
        job.error_code = None
        job.error_message = None

    def mark_failed(self, job: VideoJob, message: str, code: str = "generation_failed") -> None:
        job.status = VIDEO_STATUS_FAILED
        job.completed_at = int(time.time())
        job.expires_at = job.completed_at + self._retention_seconds
        job.error_code = code
        job.error_message = message
        debug_logger.log_error(f"[VIDEO_JOB] {job.id} 失败: {message}")

    def _is_expired(self, job: VideoJob) -> bool:
        return bool(job.expires_at) and time.time() > job.expires_at

    def _purge_expired_locked(self) -> None:
        for job_id in [jid for jid, job in self._jobs.items() if self._is_expired(job)]:
            self._jobs.pop(job_id, None)

    def _evict_locked(self) -> None:
        self._purge_expired_locked()
        if len(self._jobs) <= self._max_jobs:
            return
        # Drop the oldest finished jobs first so active work keeps running.
        finished = sorted(
            (job for job in self._jobs.values() if job.status in {VIDEO_STATUS_COMPLETED, VIDEO_STATUS_FAILED}),
            key=lambda item: item.created_at,
        )
        for job in finished:
            if len(self._jobs) <= self._max_jobs:
                break
            self._jobs.pop(job.id, None)


video_job_store = VideoJobStore()
