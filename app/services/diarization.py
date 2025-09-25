from typing import Dict, List
from pathlib import Path

from ..config import settings


DEFAULT_PYANNOTE_CACHE = "/models/pyannote"


def diarize_speakers(audio_path: str) -> Dict:
	# Prefer local path if provided
	local_path = settings.pyannote_local_path or None
	try:
		from pyannote.audio import Pipeline
		if local_path and Path(local_path).exists():
			pipeline = Pipeline.from_pretrained(local_path)
		else:
			# Download/cache under DEFAULT_PYANNOTE_CACHE if token is available
			if not settings.pyannote_auth_token:
				return {"speakers": []}
			pipeline = Pipeline.from_pretrained(
				settings.pyannote_diarization_model,
				use_auth_token=settings.pyannote_auth_token,
				cache_dir=DEFAULT_PYANNOTE_CACHE,
			)
		diarization = pipeline(audio_path)
		spk_segments: List[Dict] = []
		for turn, _, speaker in diarization.itertracks(yield_label=True):
			spk_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})
		return {"speakers": spk_segments}
	except Exception:
		return {"speakers": []}
