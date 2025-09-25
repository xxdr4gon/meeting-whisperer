import json
import os
import subprocess
import tempfile
from pathlib import Path

from celery import states
from .celery_app import celery_app
from .config import settings
from .services.transcription import transcribe_audio
from .services.diarization import diarize_speakers
from .services.emailer import send_transcript_email
from .services.summarize import summarize_text


def _extract_audio(video_path: str) -> str:
	out_wav = tempfile.mktemp(suffix=".wav")
	cmd = [
		"ffmpeg", "-y", "-i", video_path,
		"-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
		out_wav,
	]
	subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return out_wav


def _update_progress(current_task, stage: str, percent: int):
	meta = current_task.info or {}
	meta["stage"] = stage
	meta["percent"] = percent
	current_task.update_state(state=states.STARTED, meta=meta)


@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, video_path: str, job_id: str, user_email: str | None) -> dict:
	transcript_dir = Path(settings.transcript_dir)
	transcript_dir.mkdir(parents=True, exist_ok=True)
	_update_progress(self, "extract", 5)
	audio_path = _extract_audio(video_path)
	try:
		_update_progress(self, "transcribe", 20)
		transcript = transcribe_audio(audio_path)
		_update_progress(self, "diarize", 60)
		diar = diarize_speakers(audio_path)
		# Merge diarization speakers into transcript segments when possible (simple placeholder)
		_update_progress(self, "summarize", 80)
		plain_text = "\n".join(seg.get("text", "") for seg in transcript.get("segments", []))
		summary = summarize_text(plain_text, max_sentences=7)
		result = {
			"job_id": job_id,
			"segments": transcript.get("segments", []),
			"language": transcript.get("language"),
			"speakers": diar.get("speakers", []),
			"summary": summary,
		}
		json_path = transcript_dir / f"{job_id}.json"
		txt_path = transcript_dir / f"{job_id}.txt"
		sum_path = transcript_dir / f"{job_id}.summary.txt"
		json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
		# Simple TXT join
		lines = []
		for seg in result["segments"]:
			spk = seg.get("speaker", "")
			text = seg.get("text", "")
			lines.append((f"[{spk}] " if spk else "") + text)
		txt_path.write_text("\n".join(lines), encoding="utf-8")
		sum_path.write_text(summary or "", encoding="utf-8")
		_update_progress(self, "email", 95)
		if user_email:
			send_transcript_email(user_email, job_id, txt_path.read_text(encoding="utf-8"))
		_update_progress(self, "done", 100)
		return {"status": "ok", "job_id": job_id}
	finally:
		try:
			os.remove(audio_path)
		except Exception:
			pass
