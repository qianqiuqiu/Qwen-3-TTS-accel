# Qwen-3-TTS-accel

[中文说明](./README_ZH.md)

## Why this exists

The reason this repository exists is honestly pretty personal.

I really like Awa Subaru, a character from *Girls Band Cry*. After getting a bit tired of routine algorithm work, I wanted to build something more fun in the AIGC space, so I decided to make a desktop companion based on her voice and personality. The idea was simple and a little impulsive: connect a multimodal API, run TTS locally, add a Live2D model or something similar, and make a tiny Subaru companion that could chat with me on the desktop and occasionally look at what was on my screen.

For the voice part, Qwen3-TTS voice cloning worked surprisingly well. The cloned voice sounded very close to the Subaru feeling I wanted. The problem was latency. It was way too high for a desktop companion. Even after adding streaming with sentence-level chunking, the real-time factor was still far above 1. A line that should take around 3 seconds to say could take 7 to 8 seconds to generate. That completely broke the feeling of presence. My Subaru should not be getting stuck mid-sentence for that long.

So I started tracing the bottleneck and learned more about how TTS differs from GPT-style generation. The architecture has two stages: a `main talker` (28-layer transformer generating coarse codec tokens) and a `subtalker` (5-layer code_predictor generating fine codec tokens for each coarse token). Profiling showed the `subtalker` was the dominant bottleneck — it gets called once per frame and HF's `generate()` overhead adds up fast.

The key insight was that the `subtalker` always decodes exactly 15 tokens (one per remaining codebook). Fixed decode length is the perfect use case for CUDA Graphs — capture the entire decode loop once, then replay it with near-zero CPU dispatch overhead. This alone gave a ~7.5x speedup on the sub-talker path.

For the `main talker`, HF's `generate()` also carries unnecessary overhead (DynamicCache allocation, prepare_inputs, _update_model_kwargs per step). Since we already need to interleave sub-talker calls between main talker steps anyway, I wrote a direct SDPA runner with a pre-allocated static KV cache that drives the 28 layers manually. This eliminates the framework overhead while reusing the HF model's weights in-place.

At that point, it felt like this inference optimization work deserved to live on its own instead of staying buried inside the desktop companion project. So I cleaned up the logic, separated the reusable parts, and published them as this repository.

After I finished all of this, I discovered that the open-source community had already shipped more polished implementations following nearly the same approach. But I had already done the work, so I figured I might as well put it up as a repo anyway.

## What this is

`qwen3tts-accel` is a reusable **high-performance inference core** for **Qwen3-TTS**.

This repository is not a full product service. Its purpose is to isolate and share the optimization techniques that make Qwen3-TTS fast enough for real-time use:

| Component | Location | What it does |
|-----------|----------|-------------|
| **CUDA Graph sub-talker** | `subtalker/cuda_graph.py` | Captures the 15-step codec prediction loop as CUDA graphs. ~7.5x faster than HF generate for the 5-layer code_predictor. |
| **Direct SDPA main talker** | `direct/main_talker_runner.py` | Drives the 28-layer main talker with manual KV cache + SDPA, bypassing HF generate() overhead entirely. |
| **vLLM plugin** (framework) | `vllm/plugin.py` | Re-implements the main talker for vLLM's PagedAttention. Framework code for future vLLM integration. |
| **Preprocessor** | `preprocess/preprocessor.py` | Builds the exact `inputs_embeds`, `trailing_text_hidden`, and `tts_pad_embed` needed by the optimized paths. |

The default inference path uses **Direct SDPA + CUDA Graph sub-talker** (no vLLM dependency at runtime).

The repository includes a thin HTTP API only to verify that the optimized inference path works and to provide a minimal integration surface.

## Validated environment

The following combination has been tested end-to-end:

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| transformers | 4.57.3 |
| qwen-tts | 0.1.1 |
| accelerate | 1.12.0 |
| GPU | NVIDIA A100 80GB |

## Install

```bash
pip install -e ".[dev]"
```

## Docker

```bash
docker build -t qwen3tts-accel .

docker run --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-TTS-12Hz-1.7B-Base:/model \
  qwen3tts-accel
```

## Quick start (Python)

```python
from qwen3tts_accel.pipeline import Qwen3TTSAccelPipeline

pipe = Qwen3TTSAccelPipeline.from_pretrained(
    model_path="/path/to/Qwen3-TTS-12Hz-1.7B-Base",
)

wav_bytes = pipe.synthesize(
    text="Hello, this is a test.",
    language="English",
    ref_audio_path="/path/to/ref.wav",
    ref_text="Transcript of the reference audio.",
)

with open("out.wav", "wb") as f:
    f.write(wav_bytes)
```

## Start the minimal API

```bash
python -m qwen3tts_accel.api_server \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000
```

Options:

- `--max-seq-len` (default 4096): max KV cache length
- `--no-cuda-graph-subtalker`: disable CUDA graph acceleration for sub-talker
- `--api-key`: optional Bearer token authentication

## API endpoints

- `GET /health` — returns `{"status": "ok"}`
- `GET /meta` — returns model metadata
- `POST /v1/audio/speech` — full synthesis, returns `audio/wav`
- `POST /v1/audio/speech/stream` — streaming synthesis, returns PCM chunks

## Minimal request example

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    json={
        "text": "Hello, this is a minimal request example.",
        "language": "English",
        "ref_audio_path": "/path/to/ref.wav",
        "ref_text": "Hello, this is the transcript of the reference audio.",
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
    },
)

with open("out.wav", "wb") as f:
    f.write(response.content)
```

## Project layout

```text
qwen3tts_accel/
├── api_server.py          # Thin FastAPI server for validation
├── pipeline.py            # Main pipeline (Direct SDPA + CUDA Graph)
├── schemas.py             # Request/response models
├── auth.py                # Optional Bearer token auth
├── audio.py               # WAV/PCM encoding utilities
├── preprocess/            # Qwen3-TTS input construction
├── direct/                # Direct SDPA main talker runner
├── subtalker/             # CUDA Graph sub-talker acceleration
├── vllm/                  # vLLM plugin (framework, for future use)
├── decode/                # Codec decode helpers
├── state/                 # Per-sequence state management
└── benchmarks/            # Timing utilities
```

## How to read this project

If you want to understand or reuse the optimizations:

1. Start with `subtalker/cuda_graph.py` — the CUDA Graph capture/replay logic for the 5-layer code_predictor
2. Then `direct/main_talker_runner.py` — manual KV cache + SDPA for the 28-layer main talker
3. Then `preprocess/preprocessor.py` — how Qwen3-TTS inputs are constructed
4. `vllm/plugin.py` is there as a reference for vLLM integration work

## License

[MIT](LICENSE)
