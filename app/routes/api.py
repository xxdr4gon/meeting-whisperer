import os
import uuid
import json
from pathlib import Path
from typing import Optional
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
import logging
import threading
from fastapi.responses import JSONResponse, PlainTextResponse

from ..config import settings
from ..auth import get_current_user
from ..tasks import process_video_task
from ..celery_app import celery_app

router = APIRouter()


logger = logging.getLogger(__name__)


def _enqueue_async(video_path: str, job_id: str, email: str | None):
    """Fire-and-forget enqueue with small retry loop to avoid failing HTTP."""
    import time
    attempts = 0
    last_err: Exception | None = None
    while attempts < 5:
        try:
            celery_app.send_task(
                name="process_video_task",
                args=[video_path, job_id, email],
                queue="celery",
            )
            logger.info("Enqueued job %s", job_id)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Enqueue attempt %s failed for %s: %s", attempts + 1, job_id, e)
            time.sleep(1 + attempts)
            attempts += 1
    if last_err:
        logger.error("Failed to enqueue job %s after retries: %s", job_id, last_err)


@router.post("/upload", status_code=202)
async def upload_video(file: UploadFile = File(...), user=Depends(get_current_user)):
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    video_path = upload_dir / f"{job_id}_{file.filename}"

    # Stream upload to disk to avoid loading entire file into memory
    try:
        with open(video_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MiB
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    # Enqueue in a background thread with retries; always return 202
    # persist a simple task mapping file so /status can find the task later if needed
    task_map_dir = Path(settings.transcript_dir) / "_tasks"
    task_map_dir.mkdir(parents=True, exist_ok=True)
    (task_map_dir / f"{job_id}.task").write_text("pending", encoding="utf-8")

    threading.Thread(target=_enqueue_async, args=(str(video_path), job_id, user.get("email")), daemon=True).start()
    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def job_status(job_id: str, user=Depends(get_current_user)):
    transcript_json = Path(settings.transcript_dir) / f"{job_id}.json"
    if transcript_json.exists():
        return {"job_id": job_id, "status": "completed", "percent": 100, "stage": "done"}
    # read task meta if available
    task_map = Path(settings.transcript_dir) / "_tasks" / f"{job_id}.task"
    stage = "transcribe"
    percent = 10
    if task_map.exists():
        try:
            raw = task_map.read_text(encoding="utf-8").strip()
            parts = raw.split("|", 2)
            if len(parts) >= 2:
                stage = parts[0]
                percent = int(parts[1])
        except Exception:
            pass
    return {"job_id": job_id, "status": "processing", "percent": percent, "stage": stage}

@router.get("/logs/{job_id}")
async def job_logs(job_id: str, user=Depends(get_current_user)):
    log_path = Path(settings.transcript_dir) / "_tasks" / f"{job_id}.log"
    if not log_path.exists():
        return PlainTextResponse("", status_code=200)
    return PlainTextResponse(log_path.read_text(encoding="utf-8"))


@router.get("/transcript/{job_id}")
async def get_transcript(job_id: str, format: Optional[str] = "json", user=Depends(get_current_user)):
	transcript_dir = Path(settings.transcript_dir)
	json_path = transcript_dir / f"{job_id}.json"
	txt_path = transcript_dir / f"{job_id}.txt"
	sum_path = transcript_dir / f"{job_id}.summary.txt"
	if format == "json":
		if not json_path.exists():
			raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
		return JSONResponse(content=json.loads(json_path.read_text(encoding="utf-8")))
	elif format == "txt":
		if not txt_path.exists():
			raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
		return PlainTextResponse(content=txt_path.read_text(encoding="utf-8"))
	elif format == "summary":
		if not sum_path.exists():
			raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
		return PlainTextResponse(content=sum_path.read_text(encoding="utf-8"))
	else:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported format")
