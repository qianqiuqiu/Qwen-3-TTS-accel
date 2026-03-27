# ---- Base: CUDA 12.8 + Ubuntu 22.04 ----
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System deps (sox required by qwen-tts for reference audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        sox libsox-dev libsndfile1 \
        git curl \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- Deps: pin versions from validated environment ----
FROM base AS deps

# PyTorch 2.10.0 + CUDA 12.8
RUN pip install --no-cache-dir \
        torch==2.10.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ML / TTS dependencies
RUN pip install --no-cache-dir \
        transformers==4.57.3 \
        accelerate==1.12.0 \
        qwen-tts==0.1.1 \
        numpy==2.2.6

# API dependencies
RUN pip install --no-cache-dir \
        fastapi==0.135.1 \
        uvicorn==0.42.0

# ---- App ----
FROM deps AS app

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e .

# Model path via runtime mount: docker run -v /path/to/model:/model ...
EXPOSE 8000

ENTRYPOINT ["python", "-m", "qwen3tts_accel.api_server"]
CMD ["--model-path", "/model", "--host", "0.0.0.0", "--port", "8000"]
