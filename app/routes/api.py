import os
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from ..config import settings
from ..auth import get_current_user
from ..tasks import process_video_task
from ..celery_app import celery_app

router = APIRouter()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...), user=Depends(get_current_user)):
	upload_dir = Path(settings.upload_dir)
	upload_dir.mkdir(parents=True, exist_ok=True)
	job_id = str(uuid.uuid4())
	video_path = upload_dir / f"{job_id}_{file.filename}"
	with open(video_path, "wb") as f:
		f.write(await file.read())
	# enqueue background job
	async_result = process_video_task.delay(str(video_path), job_id, user.get("email"))
	return {"job_id": job_id, "task_id": async_result.id}


@router.get("/status/{job_id}")
async def job_status(job_id: str, user=Depends(get_current_user)):
	# If we knew task_id, we could query by task id; since we only have job_id, check files and active tasks
	# Simple approach: search tasks for meta with matching job (out of scope). Return file-based completion and best-effort percent.
	transcript_json = Path(settings.transcript_dir) / f"{job_id}.json"
	if transcript_json.exists():
		return {"job_id": job_id, "status": "completed", "percent": 100, "stage": "done"}
	# No persistent task lookup here; return processing
	return {"job_id": job_id, "status": "processing", "percent": 50, "stage": "transcribe"}


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
