import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from ..config import settings


def _run_whisper_cpp(audio_path: str) -> Dict:
	model_path = settings.model_path
	tmp_json = tempfile.mktemp(suffix=".json")
	cmd = [
		"whisper", audio_path,
		"--model", model_path,
		"--language", "auto",
		"--output_format", "json",
		"--output_file", tmp_json,
	]
	subprocess.run(cmd, check=True)
	data = json.loads(Path(tmp_json).read_text(encoding="utf-8"))
	segments = []
	for seg in data.get("segments", []):
		segments.append({
			"start": seg.get("start"),
			"end": seg.get("end"),
			"text": seg.get("text", "").strip(),
		})
	return {"segments": segments, "language": data.get("language", "en")}


def _run_faster_whisper(audio_path: str) -> Dict:
	from faster_whisper import WhisperModel
	compute_type = "float16" if settings.enable_gpu else "int8"
	device = "cuda" if settings.enable_gpu else "cpu"
	model = WhisperModel(settings.model_path, device=device, compute_type=compute_type)
	segments, info = model.transcribe(audio_path)
	out_segments = []
	for s in segments:
		out_segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
	return {"segments": out_segments, "language": info.language}


def _run_openai_whisper(audio_path: str) -> Dict:
	import whisper as ow
	device = "cuda" if settings.enable_gpu else "cpu"
	# MODEL_PATH can be a model name (e.g., "base") or a local directory containing the model
	model_ref = settings.model_path if settings.model_path else "base"
	model = ow.load_model(model_ref, device=device)
	result = model.transcribe(audio_path)
	segments = []
	for seg in result.get("segments", []) or []:
		segments.append({"start": seg.get("start"), "end": seg.get("end"), "text": (seg.get("text") or "").strip()})
	return {"segments": segments, "language": result.get("language", "en")}


def transcribe_audio(audio_path: str) -> Dict:
	model_type = (settings.model_type or "whisper").lower()
	if model_type == "openai-whisper":
		return _run_openai_whisper(audio_path)
	if settings.enable_gpu:
		return _run_faster_whisper(audio_path)
	return _run_whisper_cpp(audio_path)
