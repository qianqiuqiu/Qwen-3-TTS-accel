"""Streaming TTS decode: wraps HF model.generate() to yield audio frames.

Two modes:
1. ``generate_streaming``: Uses a GenerationHook to intercept codec tokens
   during HF generate(), decoding and yielding audio in batches as generation
   progresses.
2. ``generate_all``: Runs full generation then returns all audio at once.
"""

from __future__ import annotations

import threading
import queue
from typing import Any, Iterator

import numpy as np
import torch

from ..preprocess.preprocessor import PrefillPayload


DEFAULT_BUFFER_FRAMES = 8
_SENTINEL = object()


class StreamingDecodeLoop:
    """Yields PCM audio chunks from Qwen3-TTS generation."""

    def __init__(
        self,
        model: Any,
        codec_decoder: Any,
        *,
        sample_rate: int = 24000,
        buffer_frames: int = DEFAULT_BUFFER_FRAMES,
        max_new_tokens: int = 4096,
    ) -> None:
        # Handle both Qwen3TTSModel (wrapper) and Qwen3TTSForConditionalGeneration
        if hasattr(model, "model") and hasattr(model, "processor"):
            self._wrapper = model
            self._hf_model = model.model
            self._speech_tokenizer = self._hf_model.speech_tokenizer
        else:
            self._wrapper = None
            self._hf_model = model
            self._speech_tokenizer = getattr(model, "speech_tokenizer", None)

        self._codec_decoder = codec_decoder
        self._sample_rate = sample_rate
        self._buffer_frames = max(buffer_frames, 1)
        self._max_new_tokens = max_new_tokens
        self._eos_token_id = self._hf_model.config.talker_config.codec_eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_streaming(
        self,
        payload: PrefillPayload,
        sampling_params: dict[str, Any],
        original_prompt: Any = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield ``(pcm_float32, sample_rate)`` tuples as audio is generated.

        Args:
            original_prompt: The original voice_clone_prompt (list of
                VoiceClonePromptItem) before conversion to dict.
        """
        frame_queue: queue.Queue = queue.Queue()
        ref_code = self._get_ref_code(payload.voice_clone_prompt)

        def _run_generate():
            """Run the full HF generate in a background thread."""
            try:
                vcp = payload.voice_clone_prompt
                if self._wrapper is not None:
                    # Use the high-level inference API
                    # Pass original prompt (list of VoiceClonePromptItem)
                    # so generate_voice_clone can extract ref_text for ICL
                    prompt_to_pass = original_prompt if original_prompt is not None else vcp
                    wavs, sr = self._wrapper.generate_voice_clone(
                        text=payload.text,
                        language=payload.language,
                        voice_clone_prompt=prompt_to_pass,
                        temperature=sampling_params.get("temperature", 0.9),
                        top_p=sampling_params.get("top_p", 1.0),
                        top_k=sampling_params.get("top_k", 50),
                        repetition_penalty=sampling_params.get("repetition_penalty", 1.05),
                        max_new_tokens=self._max_new_tokens,
                    )
                    # Push the full result
                    frame_queue.put(("full", wavs, sr))
                else:
                    # Direct HF model path — build inputs_embeds and generate
                    talker_result = self._hf_model.talker.generate(
                        inputs_embeds=payload.inputs_embeds,
                        attention_mask=payload.attention_mask,
                        trailing_text_hidden=payload.trailing_text_hidden,
                        tts_pad_embed=payload.tts_pad_embed,
                        max_new_tokens=self._max_new_tokens,
                        min_new_tokens=2,
                        do_sample=True,
                        temperature=sampling_params.get("temperature", 0.9),
                        top_p=sampling_params.get("top_p", 1.0),
                        top_k=sampling_params.get("top_k", 50),
                        repetition_penalty=sampling_params.get("repetition_penalty", 1.05),
                        eos_token_id=self._eos_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )
                    # Extract codec codes from hidden_states
                    codec_frames = []
                    for hid in talker_result.hidden_states:
                        codec_ids = hid[-1]  # [B, 16] tensor
                        if codec_ids is not None:
                            codec_frames.append(codec_ids)

                    if codec_frames:
                        # Stack all frames: [T, 16]
                        all_codes = torch.stack([c[0] for c in codec_frames], dim=0)
                        # Trim at EOS
                        first_cb = all_codes[:, 0]
                        eos_mask = first_cb == self._eos_token_id
                        if eos_mask.any():
                            eos_idx = eos_mask.nonzero(as_tuple=True)[0][0].item()
                            all_codes = all_codes[:eos_idx]

                        frame_queue.put(("codes", all_codes))
                    else:
                        frame_queue.put(("codes", torch.empty(0, 16, dtype=torch.long)))
            except Exception as exc:
                frame_queue.put(("error", exc))
            finally:
                frame_queue.put(_SENTINEL)

        # Start generation in background
        gen_thread = threading.Thread(target=_run_generate, daemon=True)
        gen_thread.start()

        while True:
            item = frame_queue.get()
            if item is _SENTINEL:
                break

            msg_type = item[0]

            if msg_type == "error":
                raise item[1]

            if msg_type == "full":
                # Got full waveforms from the wrapper API
                wavs, sr = item[1], item[2]
                for wav in wavs:
                    if isinstance(wav, torch.Tensor):
                        wav = wav.cpu().numpy()
                    yield wav, sr
                continue

            if msg_type == "codes":
                all_codes = item[1]
                if all_codes.shape[0] == 0:
                    continue

                # Decode in chunks
                for start in range(0, all_codes.shape[0], self._buffer_frames):
                    end = min(start + self._buffer_frames, all_codes.shape[0])
                    chunk_codes = all_codes[start:end]
                    audio = self._decode_frames(chunk_codes, ref_code)
                    if audio is not None and len(audio) > 0:
                        yield audio, self._sample_rate

        gen_thread.join(timeout=5)

    def generate_all(
        self,
        payload: PrefillPayload,
        sampling_params: dict[str, Any],
        original_prompt: Any = None,
    ) -> tuple[list[np.ndarray], int]:
        """Non-streaming: collect all chunks into a single list."""
        chunks = []
        sr = self._sample_rate
        for audio, sr in self.generate_streaming(payload, sampling_params, original_prompt=original_prompt):
            chunks.append(audio)
        return chunks, sr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_frames(
        self,
        codec_block: torch.Tensor,
        ref_code: torch.Tensor | None,
    ) -> np.ndarray | None:
        """Decode codec frames [T, 16] to PCM float32 audio."""
        if codec_block.shape[0] == 0:
            return None

        if self._speech_tokenizer is None:
            return None

        # Prepend ref_code for voice clone
        if ref_code is not None:
            full_codes = torch.cat([ref_code.to(codec_block.device), codec_block], dim=0)
        else:
            full_codes = codec_block

        wavs, sr = self._speech_tokenizer.decode([{"audio_codes": full_codes}])
        wav = wavs[0]

        # Trim ref portion
        if ref_code is not None and full_codes.shape[0] > 0:
            ref_len = ref_code.shape[0]
            total_len = full_codes.shape[0]
            cut = int(ref_len / total_len * wav.shape[0])
            wav = wav[cut:]

        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        return wav

    def _get_ref_code(self, vcp: Any) -> torch.Tensor | None:
        if not isinstance(vcp, dict):
            return None
        ref_codes = vcp.get("ref_code")
        if isinstance(ref_codes, list) and ref_codes and ref_codes[0] is not None:
            return ref_codes[0]
        return None
