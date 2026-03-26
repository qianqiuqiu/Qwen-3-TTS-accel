# Qwen-3-TTS-accel

[中文说明](./README_ZH.md)

## Why this exists

The reason this repository exists is honestly pretty personal.

I really like Awa Subaru, a character from *Girls Band Cry*. After getting a bit tired of routine algorithm work, I wanted to build something more fun in the AIGC space, so I decided to make a desktop companion based on her voice and personality. The idea was simple and a little impulsive: connect a multimodal API, run TTS locally, add a Live2D model or something similar, and make a tiny Subaru companion that could chat with me on the desktop and occasionally look at what was on my screen.

For the voice part, Qwen3-TTS voice cloning worked surprisingly well. The cloned voice sounded very close to the Subaru feeling I wanted. The problem was latency. It was way too high for a desktop companion. Even after adding streaming with sentence-level chunking, the real-time factor was still far above 1. A line that should take around 3 seconds to say could take 7 to 8 seconds to generate. That completely broke the feeling of presence. My Subaru should not be getting stuck mid-sentence for that long.

So I started tracing the bottleneck and learned more about how TTS differs from GPT-style generation. The key issue turned out to be the `subtalker`. Before this, I had barely understood this vertical autoregressive codec prediction structure in TTS. My initial idea was to go big and rewrite the vLLM execution path so both `main talker` and `subtalker` would run through vLLM. After digging deeper, I realized the `subtalker` has a fixed decode length, which makes it a much better fit for CUDA Graphs. That ended up being much simpler than the original plan and avoided a much more invasive rewrite.

At that point, it felt like this inference optimization work deserved to live on its own instead of staying buried inside the desktop companion project. So I cleaned up the logic, separated the reusable parts, and published them as this repository.

`qwen3tts-accel` is a reusable **high-performance inference core** for **Qwen3-TTS**.

This repository is not primarily a full product service. Its purpose is to isolate and preserve the most valuable optimization path:

- `main talker`: custom **vLLM** plugin path
- `subtalker`: **CUDA Graph** accelerated decode path
- `prefill`: extracted Qwen3-TTS preprocessing logic

The repository includes a very thin HTTP API only to:

- verify that the optimized inference path works on its own
- provide a minimal integration surface for downstream systems
- make it easy to wrap this core with your own service layer

The API is intentionally not the main point of the project.

## What this repository keeps

- Qwen3-TTS input construction and prefill logic
- vLLM-based `main talker` integration
- CUDA Graph-based `subtalker` acceleration
- codec frame collection and decode helpers
- synchronous and streaming audio output paths

If you need uploads, speaker asset storage, access control, or orchestration, the intended approach is to build those around this inference core.

## Minimal API

The current repository exposes only a minimal verification API:

- `GET /health`
- `GET /meta`
- `POST /v1/audio/speech`
- `POST /v1/audio/speech/stream`

Behavior:

- `/v1/audio/speech` returns full `audio/wav`
- `/v1/audio/speech/stream` returns binary streaming audio chunks

## Install

```bash
pip install -e ".[vllm,dev]"
```

## Environment notes

This project depends on a GPU-oriented runtime stack. In practice, the exact working combination of the following items need be validated on the target machine:

- NVIDIA driver
- CUDA version
- PyTorch version
- vLLM version
- `qwen_tts` version
- `transformers` version


## Start the minimal API

```bash
python -m qwen3tts_accel.api_server \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000
```

Optional authentication:

```bash
python -m qwen3tts_accel.api_server \
  --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000 \
  --api-key your-secret
```

If `--api-key` is set, clients must send:

```http
Authorization: Bearer your-secret
```

## Minimal request example

The current API accepts a reference audio path and reference text per request, which is enough for basic voice-clone validation.

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
├── api_server.py
├── pipeline.py
├── schemas.py
├── auth.py
├── audio.py
├── preprocess/
├── subtalker/
├── vllm/
├── decode/
├── state/
└── benchmarks/
```

## Recommended way to read this project

If your goal is to reuse the high-performance Qwen3-TTS inference path, this repository is already enough:

- the optimized core has been isolated
- the minimal API is sufficient for validation and integration testing
- a richer service layer can be built on top as needed

## License

[MIT](LICENSE)
