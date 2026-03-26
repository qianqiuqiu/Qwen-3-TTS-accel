# qwen3tts-accel Architecture

## Goal

Expose the most useful reusable optimization path for Qwen3-TTS without the surrounding serving stack.

## Retained Components

- `qwen3tts_accel.preprocess.preprocessor.Qwen3TTSPreprocessor`
  Builds the exact `inputs_embeds`, `trailing_text_hidden`, and `tts_pad_embed` values expected by the optimized decode path.
- `qwen3tts_accel.subtalker.cuda_graph.CUDAGraphSubTalkerRunner`
  Captures the 15-step `sub-talker` decode loop as CUDA Graph replays to minimize CPU dispatch overhead.
- `qwen3tts_accel.vllm.plugin.Qwen3TTSForVLLM`
  Re-implements the `main talker` on top of vLLM primitives so `codec_0` generation runs through PagedAttention and vLLM scheduling.
- `qwen3tts_accel.vllm.runner.AsyncEngineRunner`
  Thin wrapper around `AsyncLLMEngine` for collecting codec frames from the custom plugin path.
- `qwen3tts_accel.benchmarks`
  Small utilities for timing benchmark cases and benchmark suites.

## Removed Components

- WebSocket server and protocol
- Profile scanning and hot reload
- Fake model
- HF fast-subtalker path
- Direct SDPA `main talker` runner

## Data Flow

```text
text + voice clone prompt
    |
    v
Qwen3TTSPreprocessor
    |
    v
inputs_embeds + trailing_text_hidden + tts_pad_embed
    |
    v
vLLM AsyncLLMEngine
    |
    v
Qwen3TTSForVLLM
    | \
    |  -> codec_0 logits
    v
CUDAGraphSubTalkerRunner
    |
    v
codec_1..15
    |
    v
codec frame accumulation / optional decode
```

## Key Constraint

This repository intentionally optimizes for one clearly shareable path rather than preserving every experimental branch from the original project.
