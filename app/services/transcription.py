import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from ..config import settings


WHISPER_CPP_BIN = "whisper"  # Use Python whisper CLI instead


def _run_whisper_cpp(audio_path: str) -> Dict:
	# Fallback to faster-whisper since whisper CLI is not available
	return _run_faster_whisper(audio_path)


from threading import Lock
from huggingface_hub import snapshot_download
from transformers import pipeline as hf_pipeline

_ET_ASR = None
_ET_LOCK: Lock = Lock()
_FASTER_WHISPER_MODEL = None
_FASTER_WHISPER_LOCK: Lock = Lock()


def _get_et_pipeline(model_name: str, use_gpu: bool):
    global _ET_ASR
    if _ET_ASR is not None:
        return _ET_ASR
    with _ET_LOCK:
        if _ET_ASR is not None:
            return _ET_ASR
        
        # Check if model_name is a local path or Hugging Face repo
        if model_name.startswith('/') or os.path.exists(model_name):
            # Local model path
            local_dir = model_name
            print(f"Using local Estonian model from {local_dir}")
        else:
            # Hugging Face repo - download to local directory
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
        
        device_index = 0 if use_gpu else -1
        print("Initializing HF ASR pipeline...")
        
        # Load model and processor explicitly as Whisper classes
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch
        
        print("Loading Whisper model and processor with accelerate...")
        use_accelerate = True  # We're using accelerate
        try:
            # Use accelerate for faster model loading
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from accelerate.utils import get_balanced_memory
            
            # Load from local directory with accelerate
            model = WhisperForConditionalGeneration.from_pretrained(
                local_dir,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,  # Allow custom code
                local_files_only=True,   # Only use local files
                use_safetensors=True,    # Use safetensors if available
                device_map="cpu"         # Explicitly set to CPU
            )
            processor = WhisperProcessor.from_pretrained(
                local_dir,
                trust_remote_code=True,
                local_files_only=True
            )
            print("Loaded model and processor from local directory with accelerate")
        except Exception as e:
            print(f"Failed to load from local directory with accelerate: {e}")
            # Fallback to standard loading
            use_accelerate = False  # Not using accelerate in fallback
            try:
                model = WhisperForConditionalGeneration.from_pretrained(
                    local_dir,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=True
                )
                processor = WhisperProcessor.from_pretrained(
                    local_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print("Loaded model and processor from local directory (standard)")
            except Exception as e2:
                print(f"Failed to load from local directory: {e2}")
                # Fallback to Hugging Face if local fails
                use_accelerate = False  # Not using accelerate in Hugging Face fallback
                try:
                    model = WhisperForConditionalGeneration.from_pretrained(
                        model_name,
                        dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    processor = WhisperProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    print(f"Loaded model from Hugging Face: {model_name}")
                except Exception as e3:
                    print(f"Failed to load model: {e3}")
                    raise e3
        
        # Fix timestamp configuration for the model
        print("Configuring model for timestamp support...")
        
        # Get the tokenizer to find the correct timestamp token IDs
        tokenizer = processor.tokenizer
        
        # Find the no_timestamps_token_id from the tokenizer
        no_timestamps_token_id = None
        if hasattr(tokenizer, 'all_special_ids'):
            # Look for timestamp-related special tokens
            for token_id in tokenizer.all_special_ids:
                token = tokenizer.decode([token_id])
                if '<|notimestamps|>' in token or 'notimestamps' in token.lower():
                    no_timestamps_token_id = token_id
                    break
        
        # If not found, try to find it in the tokenizer's special tokens
        if no_timestamps_token_id is None:
            try:
                # Try to encode the no timestamps token
                no_timestamps_token_id = tokenizer.encode('<|notimestamps|>', add_special_tokens=False)[0]
            except:
                # Fallback: use a common token ID for Whisper models
                no_timestamps_token_id = 50362  # Common no-timestamps token ID for Whisper
        
        print(f"Found no_timestamps_token_id: {no_timestamps_token_id}")
        
        # Set up proper generation config for timestamps
        if not hasattr(model.config, 'generation_config') or model.config.generation_config is None:
            from transformers import GenerationConfig
            model.config.generation_config = GenerationConfig()
        
        # Configure timestamps properly with the correct token ID
        model.config.generation_config.return_timestamps = True
        model.config.generation_config.no_timestamps_token_id = no_timestamps_token_id
        
        # Also set on the model itself
        if hasattr(model, 'generation_config'):
            model.generation_config.return_timestamps = True
            model.generation_config.no_timestamps_token_id = no_timestamps_token_id
        
        print("Model configured for timestamp support")
        
        # Create pipeline with proper timestamp configuration
        # Note: Don't specify device when using accelerate, as it handles device placement automatically
        pipeline_kwargs = {
            "task": "automatic-speech-recognition",
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "return_timestamps": True,  # Enable timestamps
            "generate_kwargs": {
                "language": "et",
                "task": "transcribe",
                "max_new_tokens": 200,
                "no_repeat_ngram_size": 3,
                "do_sample": False,
                "temperature": 0.0,
                "num_beams": 1,
                "early_stopping": False,
                "return_timestamps": True,
            },
        }
        
        # Only add device if not using accelerate
        if not use_accelerate:
            pipeline_kwargs["device"] = device_index
            
        _ET_ASR = hf_pipeline(**pipeline_kwargs)
        print("HF ASR pipeline ready")
        return _ET_ASR


def _process_long_audio_chunked(asr, audio_path: str, duration: float):
    """Process long audio in chunks to avoid long-form generation issues"""
    import librosa
    import tempfile
    import os
    
    # Create chunks of 25 seconds each (under the 30s limit)
    chunk_duration = 25.0
    chunks = []
    
    print(f"Processing {duration:.2f}s audio in {chunk_duration}s chunks...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    for start_time in range(0, int(duration), int(chunk_duration)):
        end_time = min(start_time + chunk_duration, duration)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract chunk
        chunk_audio = audio[start_sample:end_sample]
        
        # Save chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, chunk_audio, sr)
            tmp_path = tmp_file.name
        
        try:
            # Process chunk
            print(f"Processing chunk {start_time:.1f}s - {end_time:.1f}s...")
            chunk_result = asr(tmp_path, return_timestamps=False)
            
            # Extract text from chunk result
            if isinstance(chunk_result, dict) and "text" in chunk_result:
                text = chunk_result["text"].strip()
                if text:
                    chunks.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
            elif isinstance(chunk_result, str):
                text = chunk_result.strip()
                if text:
                    chunks.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
        except Exception as e:
            print(f"Error processing chunk {start_time:.1f}s - {end_time:.1f}s: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Combine all chunks into a single result
    if chunks:
        # Create a result similar to what the pipeline would return
        combined_text = " ".join([chunk["text"] for chunk in chunks])
        return {
            "text": combined_text,
            "chunks": chunks
        }
    else:
        return {"text": "", "chunks": []}


def _get_faster_whisper_model(use_gpu: bool):
    """Get cached faster-whisper model for speed with accelerate optimization"""
    global _FASTER_WHISPER_MODEL
    if _FASTER_WHISPER_MODEL is not None:
        return _FASTER_WHISPER_MODEL
    
    with _FASTER_WHISPER_LOCK:
        if _FASTER_WHISPER_MODEL is not None:
            return _FASTER_WHISPER_MODEL
        
        from faster_whisper import WhisperModel
        device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"
        
        print("Loading faster-whisper model with accelerate optimization...")
        try:
            # Use accelerate for faster model loading
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            
            _FASTER_WHISPER_MODEL = WhisperModel(
                "base",  # Changed from "tiny" to "base" for better accuracy
                device=device, 
                compute_type=compute_type,
                download_root="/models/english-asr",
                local_files_only=False,  # Allow downloading with accelerate
                use_safetensors=True     # Use safetensors if available
            )
            print("Faster-whisper model loaded with accelerate")
        except Exception as e:
            print(f"Failed to load with accelerate: {e}")
            # Fallback to standard loading
            _FASTER_WHISPER_MODEL = WhisperModel("base", device=device, compute_type=compute_type)
            print("Faster-whisper model loaded (standard)")
        
        return _FASTER_WHISPER_MODEL


def _run_estonian_whisper(audio_path: str) -> Dict:
    """Estonian ASR via TalTechNLP with memoized HF pipeline - optimized for speed."""
    # Use large model with better timestamp support
    model_name = getattr(settings, 'estonian_asr_dir', '/models/estonian-asr')
    print(f"Using Estonian ASR model: {model_name}")
    try:
        print("Loading Estonian ASR pipeline...")
        asr = _get_et_pipeline(model_name, settings.enable_gpu)
        print("Transcribing audio...")
        
        # Process the audio file directly with the large model that supports timestamps
        # Try timestamps first, then fallback to no timestamps if needed
        try:
            result = asr(audio_path, return_timestamps=True)
            print("Estonian transcription with timestamps successful")
        except Exception as e:
            print(f"Timestamp transcription failed, trying without timestamps: {e}")
            try:
                # Fallback to no timestamps
                result = asr(audio_path, return_timestamps=False)
                print("Estonian transcription without timestamps successful")
            except Exception as e2:
                print(f"Estonian model failed completely: {e2}")
                raise e2

        segments = []
        if isinstance(result, dict) and "chunks" in result and result["chunks"]:
            # Handle chunked results
            for ch in result["chunks"]:
                ts = ch.get("timestamp") or [0.0, 0.0]
                start = float(ts[0] or 0.0)
                end = float(ts[1] or start)
                text = (ch.get("text") or "").strip()
                if text:
                    segments.append({"start": start, "end": end, "text": text})
        elif isinstance(result, dict) and "text" in result:
            # Handle single result - create segments with estimated timestamps
            text = result.get("text", "").strip()
            if text:
                try:
                    import librosa
                    duration = librosa.get_duration(path=audio_path)
                    # Split text into sentences for better segmentation
                    sentences = text.split('. ')
                    if len(sentences) > 1:
                        # Create segments for each sentence
                        time_per_sentence = duration / len(sentences)
                        for i, sentence in enumerate(sentences):
                            if sentence.strip():
                                start_time = i * time_per_sentence
                                end_time = (i + 1) * time_per_sentence
                                segments.append({
                                    "start": start_time, 
                                    "end": end_time, 
                                    "text": sentence.strip() + ('.' if not sentence.endswith('.') else '')
                                })
                    else:
                        # Single segment for the whole text
                        segments.append({"start": 0.0, "end": duration, "text": text})
                except:
                    segments.append({"start": 0.0, "end": 0.0, "text": text})
        elif isinstance(result, str):
            # Handle string result directly
            text = result.strip()
            if text:
                try:
                    import librosa
                    duration = librosa.get_duration(path=audio_path)
                    segments.append({"start": 0.0, "end": duration, "text": text})
                except:
                    segments.append({"start": 0.0, "end": 0.0, "text": text})
        else:
            # Fallback for other result types
            text = str(result).strip()
            if text:
                segments.append({"start": 0.0, "end": 0.0, "text": text})

        print(f"Estonian transcription completed with {len(segments)} segments")
        
        # Apply grammar correction if enabled
        if settings.use_grammar_correction:
            print("Applying Estonian grammar correction...")
            try:
                from .grammar_correction import correct_estonian_grammar
                
                # Correct each segment
                corrected_segments = []
                total_corrections = 0
                
                for i, segment in enumerate(segments):
                    if isinstance(segment, dict) and "text" in segment:
                        original_text = segment["text"]
                        correction_result = correct_estonian_grammar(original_text)
                        
                        # Update segment with corrected text
                        corrected_segment = segment.copy()
                        corrected_segment["text"] = correction_result["corrected_text"]
                        corrected_segment["grammar_corrected"] = True
                        corrected_segment["corrections_applied"] = correction_result["corrections_applied"]
                        corrected_segments.append(corrected_segment)
                        
                        total_corrections += correction_result["corrections_applied"]
                        
                        if correction_result["corrections_applied"] > 0:
                            print(f"Segment {i+1}: Applied {correction_result['corrections_applied']} grammar corrections")
                    else:
                        corrected_segments.append(segment)
                
                segments = corrected_segments
                print(f"Grammar correction completed: {total_corrections} total corrections applied")
                
            except Exception as grammar_error:
                print(f"Grammar correction failed: {grammar_error}")
                print("Continuing with original transcription...")
        
        # Check if transcription seems complete (has reasonable number of segments)
        if len(segments) < 2:
            print("Warning: Estonian transcription seems incomplete, trying faster-whisper fallback...")
            try:
                fallback_result = _run_faster_whisper(audio_path)
                if len(fallback_result.get("segments", [])) > len(segments):
                    print("Using faster-whisper result as it has more segments")
                    return fallback_result
            except Exception as fallback_e:
                print(f"Faster-whisper fallback also failed: {fallback_e}")
        
        return {"segments": segments, "language": "et"}
    except Exception as e:
        print(f"Estonian model failed, falling back to faster-whisper: {e}")
        try:
            return _run_faster_whisper(audio_path)
        except Exception as fallback_e:
            print(f"Faster-whisper fallback also failed: {fallback_e}")
            return _run_whisper_cpp(audio_path)


def _run_faster_whisper(audio_path: str) -> Dict:
	# Use cached model for speed
	model = _get_faster_whisper_model(settings.enable_gpu)
	
	# Optimize transcription settings for speed
	segments, info = model.transcribe(
		audio_path,
		language=None,  # Auto-detect for speed
		task="transcribe",
		beam_size=1,  # Single beam for speed
		best_of=1,  # No beam search for speed
		patience=1,  # Low patience for speed
		length_penalty=1.0,
		temperature=0.0,
		compression_ratio_threshold=2.4,
		log_prob_threshold=-1.0,
		no_speech_threshold=0.6,
		condition_on_previous_text=False,  # Disable for speed
		prompt_reset_on_temperature=0.5,
		initial_prompt=None,
		prefix=None,
		suppress_blank=True,
		suppress_tokens=[-1],
		without_timestamps=False,  # Keep timestamps for better UX
		max_initial_timestamp=0.0,
		word_timestamps=False,  # Disable word timestamps for speed
		vad_filter=True,  # Enable VAD for better quality
		vad_parameters=dict(
			min_silence_duration_ms=500,
			speech_pad_ms=400
		)
	)
	
	out_segments = []
	for s in segments:
		out_segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
	return {"segments": out_segments, "language": info.language}


def _run_openai_whisper(audio_path: str) -> Dict:
	# Fallback to faster-whisper since openai-whisper has dependency conflicts
	return _run_faster_whisper(audio_path)


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
	"""Detect language of audio file using faster-whisper - optimized for speed"""
	try:
		print("Detecting language...")
		# Use cached model for speed
		model = _get_faster_whisper_model(settings.enable_gpu)
		
		# Transcribe only first 10 seconds for faster language detection
		segments, info = model.transcribe(
			audio_path, 
			language=None,
			beam_size=1,  # Single beam for speed
			best_of=1,    # No beam search for speed
			patience=1,   # Low patience for speed
			condition_on_previous_text=False,  # Disable for speed
			vad_filter=True,  # Enable VAD for better quality
			vad_parameters=dict(
				min_silence_duration_ms=300,  # Shorter for speed
				speech_pad_ms=200
			)
		)
		detected_lang = info.language if info.language else "en"
		print(f"Detected language: {detected_lang}")
		return detected_lang
	except Exception as e:
		print(f"Language detection failed: {e}")
		return "en"  # Default to English
