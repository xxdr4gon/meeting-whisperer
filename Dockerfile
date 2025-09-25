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


FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Install compiled whisper.cpp binary under standard name
COPY --from=whisper_builder /out/whisper-cli /usr/local/bin/whisper-cli
RUN chmod +x /usr/local/bin/whisper-cli

# Install default model to /models
RUN mkdir -p /models
COPY --from=whisper_builder /build/whisper.cpp/models/ggml-base.bin /models/ggml-base.bin

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV APP_PORT=8080 \
    APP_HOST=0.0.0.0 \
    ENABLE_GPU=false \
    GPU_BACKEND= \
    GPU_DEVICE_ID=0 \
    MODEL_PATH=/models/ggml-base.bin

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
