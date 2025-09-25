import json
import os
import subprocess
import tempfile
from pathlib import Path

from .celery_app import celery_app
from .config import settings
from .services.transcription import transcribe_audio
from .services.diarization import diarize_speakers
from .services.emailer import send_transcript_email


def _extract_audio(video_path: str) -> str:
	out_wav = tempfile.mktemp(suffix=".wav")
	cmd = [
		"ffmpeg", "-y", "-i", video_path,
		"-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
		out_wav,
	]
	subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return out_wav


@celery_app.task(name="process_video_task")
def process_video_task(video_path: str, job_id: str, user_email: str | None) -> dict:
	transcript_dir = Path(settings.transcript_dir)
	transcript_dir.mkdir(parents=True, exist_ok=True)
	audio_path = _extract_audio(video_path)
	try:
		transcript = transcribe_audio(audio_path)
		diar = diarize_speakers(audio_path)
		# Merge diarization speakers into transcript segments when possible
		result = {
			"job_id": job_id,
			"segments": transcript.get("segments", []),
			"language": transcript.get("language"),
			"speakers": diar.get("speakers", []),
		}
		json_path = transcript_dir / f"{job_id}.json"
		txt_path = transcript_dir / f"{job_id}.txt"
		json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
		# Simple TXT join
		lines = []
		for seg in result["segments"]:
			spk = seg.get("speaker", "")
			text = seg.get("text", "")
			lines.append((f"[{spk}] " if spk else "") + text)
		txt_path.write_text("\n".join(lines), encoding="utf-8")
		if user_email:
			send_transcript_email(user_email, job_id, txt_path.read_text(encoding="utf-8"))
		return {"status": "ok", "job_id": job_id}
	finally:
		try:
			os.remove(audio_path)
		except Exception:
			pass
