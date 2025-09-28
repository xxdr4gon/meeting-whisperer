# Meeting Whisperer

A FastAPI-based application for transcribing, summarizing, and correcting Estonian audio/video content using AI models.

## Features

- **Audio/Video Transcription**: Supports multiple formats with automatic language detection
- **AI-Powered Summarization**: Uses Qwen2.5-3B-Instruct for intelligent content summarization
- **Grammar Correction**: Estonian grammar and spelling correction with custom dictionary support
- **Speaker Diarization**: Identifies different speakers in conversations
- **Web Interface**: Clean dashboard for uploading, monitoring, and accessing transcripts
- **GPU Acceleration**: Optional GPU support for faster processing

## Models Used

- **Summarization**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Estonian ASR**: [TalTechNLP/whisper-large-v3-turbo-et-verbatim](https://huggingface.co/TalTechNLP/whisper-large-v3-turbo-et-verbatim)
- **Grammar Correction**: [TalTechNLP/grammar_et](https://huggingface.co/datasets/TalTechNLP/grammar_et) + custom datasets

## Quick Start

### CPU Only
```bash
docker compose up --build -d
```

### GPU Acceleration (Recommended)
```bash
docker build -f Dockerfile.gpu -t meeting-whisperer-gpu
docker-compose -f docker-compose.gpu.yml up --build
```

## Usage

1. **Upload**: Drag and drop audio/video files through the web interface
2. **Process**: The application automatically transcribes, corrects grammar, and generates summaries
3. **Access**: View transcripts, summaries, and download results in various formats

## Grammar Correction

The application includes comprehensive Estonian grammar correction with:
- Automatic dataset discovery in `models/grammar-correction/` and subdirectories
- Custom dictionary support for domain-specific corrections
- Rule-based fallbacks for common Estonian grammar patterns

## API Endpoints

- `POST /api/upload` - Upload audio/video files
- `GET /api/jobs` - List processing jobs
- `GET /api/transcripts/{job_id}` - Get transcript data
- `GET /api/export/{job_id}` - Export transcript in various formats

## Development

The application is built with:
- **Backend**: FastAPI, Celery, Redis
- **AI/ML**: Transformers, faster-whisper, PyTorch
- **Frontend**: HTML/CSS/JavaScript
- **Infrastructure**: Docker, Docker Compose