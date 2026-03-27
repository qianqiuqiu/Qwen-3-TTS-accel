# qwen3tts-accel Architecture

## Goal

Share the optimization techniques that make Qwen3-TTS fast enough for real-time use, as a reusable inference core.

## Optimization Components

- `qwen3tts_accel.subtalker.cuda_graph.CUDAGraphSubTalkerRunner`
  Captures the 15-step `sub-talker` (5-layer code_predictor) decode loop as CUDA Graph replays. ~7.5x faster than HF generate. Fixed decode length makes CUDA Graphs a natural fit.

- `qwen3tts_accel.direct.main_talker_runner.MainTalkerRunner`
  Drives the 28-layer main talker directly with manual KV cache + SDPA, bypassing HF generate() overhead (prepare_inputs, _update_model_kwargs, DynamicCache allocation). Reuses HF model weights in-place without copying.

- `qwen3tts_accel.preprocess.preprocessor.Qwen3TTSPreprocessor`
  Builds the exact `inputs_embeds`, `trailing_text_hidden`, and `tts_pad_embed` values expected by the optimized decode paths.

- `qwen3tts_accel.vllm.plugin.Qwen3TTSForVLLM`
  Re-implements the main talker on top of vLLM primitives (PagedAttention, parallel linear layers). Framework code for future vLLM integration — not used in the default inference path.

## Default Data Flow

```text
text + ref_audio + ref_text
    |
    v
Qwen3TTSModel.create_voice_clone_prompt()
    |
    v
Qwen3TTSPreprocessor.prepare_inputs()
    |
    v
inputs_embeds + trailing_text_hidden + tts_pad_embed
    |
    v
MainTalkerRunner.generate()
    |  prefill: full sequence through 28 layers (SDPA + manual KV cache)
    |  decode loop:
    |    1. sample codec_0 token from main talker logits
    |    2. run sub-talker (CUDA Graph) -> codec_1..15
    |    3. sum all 16 codec embeddings + text conditioning -> next input
    |    4. one decode step through 28 layers
    |
    v
codec frames (list of 16-token frames)
    |
    v
speech_tokenizer.decode()
    |
    v
waveform -> WAV bytes
```

## Key Design Decisions

- **No vLLM at runtime by default.** vLLM's v1 engine (0.17+) runs models in subprocesses, which makes custom plugin integration complex (cross-process state, sub-talker sharing). The direct SDPA path avoids this entirely while achieving comparable performance for single-request use.

- **In-place weight reuse.** MainTalkerRunner references the HF model's layers, embeddings, and head directly — no weight copying, no extra memory.

- **CUDA Graphs for fixed-length decode.** The sub-talker always generates exactly 15 tokens (codebooks 1-15). This fixed structure makes CUDA Graph capture/replay ideal — eliminate all CPU dispatch overhead.

- **vLLM plugin kept as framework reference.** The plugin code demonstrates how to register a custom TTS model with vLLM's ModelRegistry, implement weight loading with QKV/gate-up merging, and integrate M-RoPE. Useful for anyone adapting Qwen3-TTS to future vLLM versions with better custom model support.
