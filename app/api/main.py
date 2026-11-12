"""FastAPI app: upload a clip -> bark detector, audio / body / scene branches, fusion -> a
step-by-step explanation per detected vocalization.

Run from the repo root:
    uvicorn app.api.main:app
then open http://127.0.0.1:8000

POST /api/jobs                 upload a video or audio file -> {"job_id": ...}
GET  /api/jobs/{job_id}        status, stage and progress; the results once done
GET  /api/jobs/{job_id}/video  the uploaded file, for the page's player (supports seeking)

Models load once at startup and jobs run one at a time (one GPU). Job files are deleted
JOB_TTL_S after upload.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.serialize import segment_json
from woofdecoder.pipeline import Pipeline
from woofdecoder.preprocess.media import describe_media

logger = logging.getLogger("woofdecoder.app")

WEB_DIR: Path = Path(__file__).resolve().parents[1] / "web"
JOBS_DIR: Path = Path(tempfile.gettempdir()) / "woofdecoder_jobs"
MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES: int = 1024 * 1024
JOB_TTL_S: float = 3600.0  # delete a job (and its file) 1 hour after upload
CLEANUP_EVERY_S: float = 300.0
MEDIA_SUFFIXES: set[str] = {
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
}  # fmt: skip


@dataclass
class Job:
    id: str
    path: Path
    filename: str
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    done: int = 0
    total: int = 0
    error: str | None = None
    results: list[dict[str, Any]] | None = None

    def as_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
        }
        if self.error is not None:
            data["error"] = self.error
        if self.status == "done":
            data["results"] = self.results
        return data


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def run_job(pipeline: Pipeline, job: Job) -> None:
    """Runs on the single worker thread."""

    def progress(stage: str, done: int, total: int) -> None:
        job.stage, job.done, job.total = stage, done, total

    job.status = "running"
    try:
        results = pipeline.run(job.path, progress=progress)
        job.results = [segment_json(number, result) for number, result in enumerate(results, 1)]
        job.stage, job.done, job.total = "done", len(results), len(results)
        job.status = "done"
    except Exception as error:  # report any failure to the page instead of hanging
        logger.exception("job %s failed", job.id)
        job.error = f"The analysis failed: {error}"
        job.status = "error"


def delete_old_jobs(now: float) -> None:
    with jobs_lock:
        old = [
            job
            for job in jobs.values()
            if now - job.created > JOB_TTL_S and job.status in ("done", "error")
        ]
        for job in old:
            del jobs[job.id]
    for job in old:
        shutil.rmtree(job.path.parent, ignore_errors=True)


async def cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_EVERY_S)
        delete_old_jobs(time.time())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline = Pipeline()  # load every model once
    app.state.executor = ThreadPoolExecutor(max_workers=1)  # one job at a time on the GPU
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    cleaner = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleaner.cancel()
        with suppress(asyncio.CancelledError):
            await cleaner
        app.state.executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(JOBS_DIR, ignore_errors=True)


app = FastAPI(title="WoofDecoder", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def looks_like_media(upload: UploadFile) -> bool:
    content_type = (upload.content_type or "").lower()
    if content_type.startswith(("video/", "audio/")):
        return True
    return Path(upload.filename or "").suffix.lower() in MEDIA_SUFFIXES


def save_upload(upload: UploadFile, path: Path) -> None:
    """Copy the upload to path, stopping with 413 once it passes MAX_UPLOAD_BYTES."""
    size = 0
    with open(path, "wb") as out:
        while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413, f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                )
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "The file is empty.")


def check_media(path: Path) -> None:
    try:
        info = describe_media(path, verbose=False)
    except Exception as error:
        raise HTTPException(415, "This is not a video or audio file the app can read.") from error
    if not info["has_audio"]:
        raise HTTPException(
            422, "This file has no audio track, so there are no dog sounds to detect."
        )


@app.post("/api/jobs")
def create_job(request: Request, clip: Annotated[UploadFile, File()]) -> dict[str, str]:
    if not looks_like_media(clip):
        raise HTTPException(415, "Please upload a video or audio file (e.g. .mp4, .mov, .wav).")
    if clip.size is not None and clip.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    suffix = Path(clip.filename or "").suffix.lower()
    path = job_dir / f"upload{suffix if suffix in MEDIA_SUFFIXES else ''}"
    try:
        save_upload(clip, path)
        check_media(path)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    job = Job(job_id, path, clip.filename or "clip")
    with jobs_lock:
        jobs[job_id] = job
    request.app.state.executor.submit(run_job, request.app.state.pipeline, job)
    return {"job_id": job_id}


def get_job(job_id: str) -> Job:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This job was not found. Results are kept for 1 hour.")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    return get_job(job_id).as_json()


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = get_job(job_id)
    media_type = mimetypes.guess_type(job.filename)[0] or "application/octet-stream"
    return FileResponse(job.path, media_type=media_type)
