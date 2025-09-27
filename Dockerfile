# syntax=docker/dockerfile:1

FROM debian:stable-slim AS whisper_builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake libopenblas-dev curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /build/whisper.cpp
# Build with BLAS (faster CPU inference)
RUN make -j$(nproc) GGML_BLAS=1
# Download a default CPU-friendly model (base)
RUN bash -c "cd models && ./download-ggml-model.sh base"
# Normalize output binary path to /out/whisper-cli regardless of repo layout
RUN mkdir -p /out && \
    if [ -f main ]; then cp main /out/whisper-cli; \
    elif [ -f bin/whisper ]; then cp bin/whisper /out/whisper-cli; \
    elif [ -f bin/main ]; then cp bin/main /out/whisper-cli; \
    elif [ -f ./build/bin/whisper ]; then cp ./build/bin/whisper /out/whisper-cli; \
    elif [ -f ./build/bin/main ]; then cp ./build/bin/main /out/whisper-cli; \
    else echo "whisper binary not found after build" && ls -la && exit 1; fi

# Base image with all dependencies
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_LOAD_SAFE_MODE=1

# Install system dependencies in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    libopenblas0 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install compiled whisper.cpp binary
COPY --from=whisper_builder /out/whisper-cli /usr/local/bin/whisper-cli
RUN chmod +x /usr/local/bin/whisper-cli

# Install default model to /models
RUN mkdir -p /models
COPY --from=whisper_builder /build/whisper.cpp/models/ggml-base.bin /models/ggml-base.bin

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with memory optimization
# Install PyTorch first (largest dependency) - CPU only version
RUN pip install --no-cache-dir --no-deps torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --no-deps torchaudio==2.8.0

# Install transformers and accelerate
RUN pip install --no-cache-dir --no-deps "transformers>=4.40.0,<5.0.0" "accelerate>=0.20.0"

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify PyTorch installation (CPU only)
RUN python -c "import torch; import torchaudio; print(f'PyTorch version: {torch.__version__}'); print(f'TorchAudio version: {torchaudio.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); assert not torch.cuda.is_available(), 'CUDA should not be available in CPU-only build'"

# Create models directories
RUN mkdir -p /models/hf_et_model /models/estonian-ai

COPY . .

# App service (web server)
FROM base AS app
ENV APP_PORT=8080 \
    APP_HOST=0.0.0.0 \
    ENABLE_GPU=false \
    GPU_BACKEND= \
    GPU_DEVICE_ID=0 \
    MODEL_PATH=/models/ggml-base.bin
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Worker service (background tasks)
FROM base AS worker
ENV PYTORCH_NUM_THREADS=20 \
    OMP_NUM_THREADS=20 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_LOAD_SAFE_MODE=1 \
    TRANSFORMERS_OFFLINE=0
CMD ["celery", "-A", "app.celery_app.celery_app", "worker", "--loglevel=INFO", "-c", "20"]
