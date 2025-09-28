import json
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from celery import states
from .celery_app import celery_app
from .config import settings
from .services.transcription import transcribe_audio
from .services.diarization import diarize_speakers
from .services.emailer import send_transcript_email
from .services.summarize import summarize_text
from .services.ai_summarize import create_ai_meeting_summary


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
    meta = getattr(current_task, "_progress_meta", {})
    meta["stage"] = stage
    meta["percent"] = percent
    setattr(current_task, "_progress_meta", meta)
    current_task.update_state(state=states.STARTED, meta=meta)
    # also persist to a simple task file for UI polling and logs
    job_id = meta.get("job_id")
    if job_id:
        try:
            task_dir = Path(settings.transcript_dir) / "_tasks"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / f"{job_id}.task").write_text(f"{stage}|{percent}", encoding="utf-8")
        except Exception:
            pass


def _log(job_id: str, message: str):
    try:
        task_dir = Path(settings.transcript_dir) / "_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        with open(task_dir / f"{job_id}.log", "a", encoding="utf-8") as fh:
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass


@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, video_path: str, job_id: str, user_email: str | None) -> dict:
    transcript_dir = Path(settings.transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract original filename from video_path (remove job_id prefix)
    video_path_obj = Path(video_path)
    original_filename = video_path_obj.name
    if '_' in original_filename:
        # Remove job_id prefix: "uuid_filename.ext" -> "filename.ext"
        original_filename = '_'.join(original_filename.split('_')[1:])
    
    # Use original filename for transcript files
    base_name = Path(original_filename).stem  # Remove extension
    
    # seed job_id in meta for persistence
    setattr(self, "_progress_meta", {"job_id": job_id})
    _update_progress(self, "extract", 5)
    _log(job_id, f"Extracting audio from {video_path}")
    audio_path = _extract_audio(video_path)
    try:
        _update_progress(self, "transcribe", 20)
        _log(job_id, "Starting transcription")
        transcript = transcribe_audio(audio_path)
        _update_progress(self, "diarize", 60)
        _log(job_id, "Running diarization")
        diar = diarize_speakers(audio_path)
        # Merge diarization speakers into transcript segments
        speaker_segments = diar.get("speakers", [])
        for segment in transcript.get("segments", []):
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            # Find the speaker that overlaps most with this segment
            best_speaker = None
            best_overlap = 0
            for spk_seg in speaker_segments:
                spk_start = spk_seg.get("start", 0)
                spk_end = spk_seg.get("end", 0)
                # Calculate overlap
                overlap_start = max(segment_start, spk_start)
                overlap_end = min(segment_end, spk_end)
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = spk_seg.get("speaker", "Speaker")
            if best_speaker:
                segment["speaker"] = best_speaker
        _update_progress(self, "summarize", 80)
        _log(job_id, "Summarizing transcript")
        plain_text = "\n".join(seg.get("text", "") for seg in transcript.get("segments", []))
        
        # Unload Estonian model to free GPU memory for AI summarization
        from app.services.transcription import unload_estonian_model
        unload_estonian_model()
        
        # Additional aggressive memory cleanup
        import torch
        import time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            # Small delay to ensure memory is fully freed
            time.sleep(2)
        
        # Use AI summarization if enabled, otherwise fallback to rule-based
        if settings.use_ai_summarization:
            try:
                detected_language = transcript.get("language", "et")
                _log(job_id, f"Starting AI summarization for language: {detected_language}")
                summary = create_ai_meeting_summary(plain_text, detected_language)
                _log(job_id, "AI summarization completed")
                
                # Unload AI summarizer to free GPU memory
                from app.services.ai_summarize import unload_ai_summarizer
                unload_ai_summarizer()
                _log(job_id, "AI summarizer unloaded from GPU memory")
                
            except Exception as e:
                _log(job_id, f"AI summarization failed: {e}, using fallback")
                _log(job_id, "AI summarization failed - will use rule-based summarization")
                summary = summarize_text(plain_text, max_sentences=7)
        else:
            _log(job_id, "Using rule-based summarization")
            summary = summarize_text(plain_text, max_sentences=7)
        result = {
            "job_id": job_id,
            "original_filename": original_filename,
            "segments": transcript.get("segments", []),
            "language": transcript.get("language"),
            "speakers": diar.get("speakers", []),
            "summary": summary,
        }
        json_path = transcript_dir / f"{base_name}.json"
        txt_path = transcript_dir / f"{base_name}.txt"
        sum_path = transcript_dir / f"{base_name}.summary.txt"
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
        _log(job_id, "Sending transcript email")
        if user_email:
            send_transcript_email(user_email, job_id, txt_path.read_text(encoding="utf-8"))
        _update_progress(self, "done", 100)
        _log(job_id, "Job complete")
        _log(job_id, f"Final progress: stage=done, percent=100")
        return {"status": "ok", "job_id": job_id}
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass