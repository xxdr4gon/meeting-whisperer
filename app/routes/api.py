import os
import uuid
import json
from pathlib import Path
from typing import Optional
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
import logging
import threading
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse

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


@router.get("/export/{job_id}")
async def export_transcript(job_id: str, format: str = "pdf", user=Depends(get_current_user)):
	"""Export transcript in various formats: pdf, docx, pptx"""
	transcript_dir = Path(settings.transcript_dir)
	json_path = transcript_dir / f"{job_id}.json"
	
	if not json_path.exists():
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")
	
	data = json.loads(json_path.read_text(encoding="utf-8"))
	
	if format == "pdf":
		return await _export_pdf(data, job_id)
	elif format == "docx":
		return await _export_docx(data, job_id)
	elif format == "pptx":
		return await _export_pptx(data, job_id)
	else:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported export format")


async def _export_pdf(data, job_id):
	"""Export transcript as PDF"""
	try:
		from reportlab.lib.pagesizes import letter, A4
		from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
		from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
		from reportlab.lib.units import inch
		from reportlab.lib import colors
		from io import BytesIO
		
		buffer = BytesIO()
		doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
		story = []
		styles = getSampleStyleSheet()
		
		# Title
		title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30)
		story.append(Paragraph(f"Meeting Transcript - {job_id}", title_style))
		story.append(Spacer(1, 12))
		
		# Summary if available
		if 'summary' in data and data['summary']:
			story.append(Paragraph("Summary", styles['Heading2']))
			story.append(Paragraph(data['summary'].replace('\n', '<br/>'), styles['Normal']))
			story.append(Spacer(1, 12))
		
		# Transcript segments
		story.append(Paragraph("Transcript", styles['Heading2']))
		story.append(Spacer(1, 12))
		
		for segment in data.get('segments', []):
			start_time = segment.get('start', 0)
			end_time = segment.get('end', 0)
			text = segment.get('text', '')
			speaker = segment.get('speaker', 'Unknown')
			
			time_str = f"[{start_time:.1f}s - {end_time:.1f}s]"
			speaker_str = f"<b>{speaker}:</b>" if speaker != 'Unknown' else ""
			
			story.append(Paragraph(f"{time_str} {speaker_str} {text}", styles['Normal']))
			story.append(Spacer(1, 6))
		
		doc.build(story)
		buffer.seek(0)
		
		return FileResponse(
			path=buffer,
			filename=f"transcript_{job_id}.pdf",
			media_type="application/pdf"
		)
	except ImportError:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="PDF export not available - missing reportlab")


async def _export_docx(data, job_id):
	"""Export transcript as DOCX"""
	try:
		from docx import Document
		from docx.shared import Inches
		from io import BytesIO
		
		doc = Document()
		doc.add_heading(f'Meeting Transcript - {job_id}', 0)
		
		# Summary
		if 'summary' in data and data['summary']:
			doc.add_heading('Summary', level=1)
			doc.add_paragraph(data['summary'])
		
		# Transcript
		doc.add_heading('Transcript', level=1)
		
		for segment in data.get('segments', []):
			start_time = segment.get('start', 0)
			end_time = segment.get('end', 0)
			text = segment.get('text', '')
			speaker = segment.get('speaker', 'Unknown')
			
			time_str = f"[{start_time:.1f}s - {end_time:.1f}s]"
			speaker_str = f"{speaker}:" if speaker != 'Unknown' else ""
			
			doc.add_paragraph(f"{time_str} {speaker_str} {text}")
		
		buffer = BytesIO()
		doc.save(buffer)
		buffer.seek(0)
		
		return FileResponse(
			path=buffer,
			filename=f"transcript_{job_id}.docx",
			media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
		)
	except ImportError:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DOCX export not available - missing python-docx")


async def _export_pptx(data, job_id):
	"""Export transcript as PPTX"""
	try:
		from pptx import Presentation
		from pptx.util import Inches, Pt
		from io import BytesIO
		
		prs = Presentation()
		
		# Title slide
		title_slide_layout = prs.slide_layouts[0]
		slide = prs.slides.add_slide(title_slide_layout)
		title = slide.shapes.title
		subtitle = slide.placeholders[1]
		title.text = f"Meeting Transcript"
		subtitle.text = f"Job ID: {job_id}"
		
		# Summary slide
		if 'summary' in data and data['summary']:
			summary_slide = prs.slides.add_slide(prs.slide_layouts[1])
			summary_slide.shapes.title.text = "Summary"
			summary_slide.placeholders[1].text = data['summary']
		
		# Transcript slides (group segments)
		segments = data.get('segments', [])
		segments_per_slide = 5
		
		for i in range(0, len(segments), segments_per_slide):
			slide = prs.slides.add_slide(prs.slide_layouts[1])
			slide.shapes.title.text = f"Transcript (Part {i//segments_per_slide + 1})"
			
			textbox = slide.placeholders[1]
			text_frame = textbox.text_frame
			text_frame.clear()
			
			for segment in segments[i:i+segments_per_slide]:
				start_time = segment.get('start', 0)
				end_time = segment.get('end', 0)
				text = segment.get('text', '')
				speaker = segment.get('speaker', 'Unknown')
				
				time_str = f"[{start_time:.1f}s - {end_time:.1f}s]"
				speaker_str = f"{speaker}:" if speaker != 'Unknown' else ""
				
				p = text_frame.add_paragraph()
				p.text = f"{time_str} {speaker_str} {text}"
				p.font.size = Pt(12)
		
		buffer = BytesIO()
		prs.save(buffer)
		buffer.seek(0)
		
		return FileResponse(
			path=buffer,
			filename=f"transcript_{job_id}.pptx",
			media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
		)
	except ImportError:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="PPTX export not available - missing python-pptx")


@router.put("/transcript/{job_id}/speakers")
async def update_speakers(job_id: str, speakers: dict, user=Depends(get_current_user)):
	"""Update speaker names for a transcript"""
	transcript_dir = Path(settings.transcript_dir)
	json_path = transcript_dir / f"{job_id}.json"
	
	if not json_path.exists():
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")
	
	# Load current transcript
	data = json.loads(json_path.read_text(encoding="utf-8"))
	
	# Update speaker names in segments
	for segment in data.get('segments', []):
		old_speaker = segment.get('speaker', '')
		if old_speaker in speakers:
			segment['speaker'] = speakers[old_speaker]
	
	# Save updated transcript
	json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
	
	# Update TXT file
	txt_path = transcript_dir / f"{job_id}.txt"
	if txt_path.exists():
		lines = []
		for seg in data["segments"]:
			spk = seg.get("speaker", "")
			text = seg.get("text", "")
			lines.append((f"[{spk}] " if spk else "") + text)
		txt_path.write_text("\n".join(lines), encoding="utf-8")
	
	return {"status": "success", "message": "Speaker names updated"}


@router.get("/search")
async def search_transcripts(query: str, user=Depends(get_current_user)):
	"""Search across all transcripts"""
	transcript_dir = Path(settings.transcript_dir)
	results = []
	
	# Search through all JSON files
	for json_file in transcript_dir.glob("*.json"):
		try:
			data = json.loads(json_file.read_text(encoding="utf-8"))
			job_id = json_file.stem
			
			# Search in segments
			matching_segments = []
			for segment in data.get('segments', []):
				text = segment.get('text', '').lower()
				if query.lower() in text:
					matching_segments.append(segment)
			
			# Search in summary
			summary_matches = []
			if 'summary' in data and data['summary']:
				if query.lower() in data['summary'].lower():
					summary_matches.append(data['summary'])
			
			if matching_segments or summary_matches:
				results.append({
					"job_id": job_id,
					"segments": matching_segments,
					"summary_matches": summary_matches,
					"total_segments": len(data.get('segments', [])),
					"language": data.get('language', 'unknown')
				})
		except Exception as e:
			continue
	
	return {"query": query, "results": results, "total": len(results)}


@router.get("/analytics")
async def get_analytics(user=Depends(get_current_user)):
	"""Get usage statistics and performance metrics"""
	transcript_dir = Path(settings.transcript_dir)
	
	# Count total transcripts
	total_transcripts = len(list(transcript_dir.glob("*.json")))
	
	# Calculate total duration
	total_duration = 0
	language_stats = {}
	file_sizes = []
	
	for json_file in transcript_dir.glob("*.json"):
		try:
			data = json.loads(json_file.read_text(encoding="utf-8"))
			
			# Calculate duration
			segments = data.get('segments', [])
			if segments:
				last_segment = max(segments, key=lambda x: x.get('end', 0))
				total_duration += last_segment.get('end', 0)
			
			# Language stats
			lang = data.get('language', 'unknown')
			language_stats[lang] = language_stats.get(lang, 0) + 1
			
			# File size
			file_sizes.append(json_file.stat().st_size)
		except Exception:
			continue
	
	# Calculate averages
	avg_duration = total_duration / total_transcripts if total_transcripts > 0 else 0
	avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
	
	return {
		"total_transcripts": total_transcripts,
		"total_duration_hours": total_duration / 3600,
		"average_duration_minutes": avg_duration / 60,
		"language_distribution": language_stats,
		"average_file_size_mb": avg_file_size / (1024 * 1024),
		"storage_used_mb": sum(file_sizes) / (1024 * 1024)
	}
