from typing import Dict, List

from ..config import settings


def diarize_speakers(audio_path: str) -> Dict:
	# Minimal placeholder to avoid heavy runtime during bootstrap; integrate real model when token is provided
	if not settings.pyannote_auth_token:
		# Return empty or a dummy single-speaker diarization
		return {"speakers": []}
	try:
		from pyannote.audio import Pipeline
		pipeline = Pipeline.from_pretrained(settings.pyannote_diarization_model, use_auth_token=settings.pyannote_auth_token)
		diarization = pipeline(audio_path)
		spk_segments: List[Dict] = []
		for turn, _, speaker in diarization.itertracks(yield_label=True):
			spk_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})
		return {"speakers": spk_segments}
	except Exception:
		return {"speakers": []}
