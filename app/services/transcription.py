import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from ..config import settings


WHISPER_CPP_BIN = "whisper"  # Use Python whisper CLI instead


def _run_whisper_cpp(audio_path: str) -> Dict:
	# Python whisper CLI expects model names like "base", not file paths
	model_name = "base"  # Default to base model
	tmp_prefix = tempfile.mktemp(prefix="wcpp_")
	# Python whisper CLI: audio file is positional argument
	cmd = [
		WHISPER_CPP_BIN,
		audio_path,
		"--model", model_name,
		"--output_format", "json",
		"--output_dir", os.path.dirname(tmp_prefix),
	]
	subprocess.run(cmd, check=True)
	# Python whisper CLI creates output with the same name as input file
	audio_name = Path(audio_path).stem
	json_path = Path(os.path.dirname(tmp_prefix)) / f"{audio_name}.json"
	data = json.loads(json_path.read_text(encoding="utf-8"))
	segments = []
	for seg in data.get("segments", []):
		segments.append({
			"start": seg.get("t0"),
			"end": seg.get("t1"),
			"text": (seg.get("text") or "").strip(),
		})
	for s in segments:
		if isinstance(s.get("start"), (int, float)):
			s["start"] = s["start"] / 100.0 if s["start"] > 1000 else s["start"]
		if isinstance(s.get("end"), (int, float)):
			s["end"] = s["end"] / 100.0 if s["end"] > 1000 else s["end"]
	return {"segments": segments, "language": data.get("language", "en")}


from threading import Lock
from huggingface_hub import snapshot_download
from transformers import pipeline as hf_pipeline

_ET_ASR = None
_ET_LOCK: Lock = Lock()


def _get_et_pipeline(model_name: str, use_gpu: bool):
    global _ET_ASR
    if _ET_ASR is not None:
        return _ET_ASR
    with _ET_LOCK:
        if _ET_ASR is not None:
            return _ET_ASR
        
        # Check if model is already downloaded locally
        local_dir = "/models/hf_et_model"
        if not os.path.exists(local_dir) or not os.listdir(local_dir):
            print("Preparing Estonian model snapshot...")
            try:
                local_dir = snapshot_download(
                    repo_id=model_name,
                    local_dir=local_dir,
                )
                print(f"Model downloaded to {local_dir}")
            except Exception as e:
                print(f"Failed to download Estonian model: {e}")
                raise e
        else:
            print(f"Using pre-downloaded Estonian model from {local_dir}")
        # Verify the model files exist (Estonian model uses vocab.json + merges.txt instead of tokenizer.json)
        required_files = ["config.json", "preprocessor_config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]  # Either pytorch or safetensors format
        tokenizer_files = ["vocab.json", "merges.txt", "tokenizer_config.json"]  # Estonian model uses BPE tokenizer
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(local_dir, f))]
        missing_model = not any(os.path.exists(os.path.join(local_dir, f)) for f in model_files)
        missing_tokenizer = [f for f in tokenizer_files if not os.path.exists(os.path.join(local_dir, f))]
        
        if missing_files or missing_model or missing_tokenizer:
            missing_list = missing_files + (["model files"] if missing_model else []) + missing_tokenizer
            print(f"Missing required files: {missing_list}")
            print("Re-downloading model...")
            local_dir = snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
            )
            
        device_index = 0 if use_gpu else -1
        print("Initializing HF ASR pipeline...")
        
        # Load model and processor explicitly as Whisper classes
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch
        
        print("Loading Whisper model and processor...")
        try:
            # Try loading from local directory first
            model = WhisperForConditionalGeneration.from_pretrained(
                local_dir,
                dtype=torch.float32,  # Use dtype instead of torch_dtype
                low_cpu_mem_usage=True,
                use_safetensors=True  # Use safetensors to avoid torch.load vulnerability
            )
            processor = WhisperProcessor.from_pretrained(local_dir)
            print("✅ Loaded model and processor from local directory")
        except Exception as e:
            print(f"Failed to load from local directory: {e}")
            print("Downloading model directly from Hugging Face...")
            try:
                # Download directly from Hugging Face as fallback
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=torch.float32,  # Use dtype instead of torch_dtype
                    low_cpu_mem_usage=True,
                    use_safetensors=True  # Use safetensors to avoid torch.load vulnerability
                )
                processor = WhisperProcessor.from_pretrained(model_name)
                # Save to local directory for future use
                model.save_pretrained(local_dir)
                processor.save_pretrained(local_dir)
                print(f"✅ Model downloaded and saved to {local_dir}")
            except Exception as e2:
                print(f"❌ Failed to download model: {e2}")
                raise e2
        
        # Create pipeline with explicit model and processor
        _ET_ASR = hf_pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=60,
            stride_length_s=10,
            device=device_index,
            return_timestamps=True,
            generate_kwargs={
                "language": "et",
                "task": "transcribe",
                "max_new_tokens": 300,
                "no_repeat_ngram_size": 3,
                "do_sample": False,
                "temperature": 0.0,
            },
        )
        print("HF ASR pipeline ready")
        return _ET_ASR


def _run_estonian_whisper(audio_path: str) -> Dict:
    """Estonian ASR via TalTechNLP with memoized HF pipeline and chunking."""
    model_name = getattr(settings, 'estonian_model', 'TalTechNLP/whisper-large-v3-turbo-et-verbatim')
    print(f"Using Estonian model: {model_name}")
    try:
        print("Loading ASR pipeline (chunked)...")
        asr = _get_et_pipeline(model_name, settings.enable_gpu)
        print("Transcribing with chunking...")
        result = asr(audio_path)

        segments = []
        if isinstance(result, dict) and isinstance(result.get("chunks"), list) and result["chunks"]:
            for ch in result["chunks"]:
                ts = ch.get("timestamp") or [0.0, 0.0]
                start = float(ts[0] or 0.0)
                end = float(ts[1] or start)
                text = (ch.get("text") or "").strip()
                if text:
                    segments.append({"start": start, "end": end, "text": text})
        else:
            text = (result.get("text") if isinstance(result, dict) else str(result)).strip()
            if text:
                segments.append({"start": 0.0, "end": 0.0, "text": text})

        print(f"Estonian transcription completed with {len(segments)} segments")
        return {"segments": segments, "language": "et"}
    except Exception as e:
        print(f"Estonian model failed, falling back to regular whisper: {e}")
        return _run_whisper_cpp(audio_path)


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
	model_ref = settings.model_path if settings.model_path else "base"
	model = ow.load_model(model_ref, device=device)
	result = model.transcribe(audio_path)
	segments = []
	for seg in result.get("segments", []) or []:
		segments.append({"start": seg.get("start"), "end": seg.get("end"), "text": (seg.get("text") or "").strip()})
	return {"segments": segments, "language": result.get("language", "en")}


def transcribe_audio(audio_path: str) -> Dict:
	model_type = (settings.model_type or "whisper").lower()
	
	print(f"=== TRANSCRIPTION DEBUG ===")
	print(f"Model type: {model_type}")
	print(f"Settings use_estonian_model: {getattr(settings, 'use_estonian_model', 'NOT_SET')}")
	
	# First, detect language to determine if we should use Estonian model
	detected_language = _detect_language(audio_path)
	
	# Use Estonian-optimized model if Estonian is detected and enabled
	use_estonian = getattr(settings, 'use_estonian_model', False)
	print(f"Use Estonian: {use_estonian}, Detected language: {detected_language}")
	
	if use_estonian and detected_language == "et":
		print("Using Estonian model!")
		return _run_estonian_whisper(audio_path)
	else:
		print("Using regular Whisper model")
	
	if model_type == "openai-whisper":
		return _run_openai_whisper(audio_path)
	if model_type == "faster-whisper":
		return _run_faster_whisper(audio_path)
	if settings.enable_gpu:
		return _run_faster_whisper(audio_path)
	return _run_whisper_cpp(audio_path)


def _detect_language(audio_path: str) -> str:
	"""Detect language of audio file using a small sample"""
	import whisper as ow
	device = "cuda" if settings.enable_gpu else "cpu"
	
	try:
		print("Detecting language...")
		# Use tiny model for fast language detection
		model = ow.load_model("tiny", device=device)
		# Transcribe only first 30 seconds for language detection
		result = model.transcribe(audio_path, language=None, fp16=False)
		detected_lang = result.get("language", "en")
		print(f"Detected language: {detected_lang}")
		return detected_lang
	except Exception as e:
		print(f"Language detection failed: {e}")
		return "en"  # Default to English
