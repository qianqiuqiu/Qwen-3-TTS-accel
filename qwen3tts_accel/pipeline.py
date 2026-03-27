from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

from .audio import waveform_to_pcm_s16le_bytes, waveform_to_wav_bytes
from .preprocess import Qwen3TTSPreprocessor

logger = logging.getLogger(__name__)


def ensure_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "qwen3tts-accel currently requires a CUDA-capable environment. "
            "CUDA was not detected on this machine."
        )


class Qwen3TTSAccelPipeline:
    def __init__(
        self,
        *,
        model_path: str,
        model: Any,
        preprocessor: Qwen3TTSPreprocessor,
        runner: Any,
        sample_rate: int,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.model_name = Path(model_path).name
        self.model = model
        self.preprocessor = preprocessor
        self.runner = runner
        self.sample_rate = sample_rate
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str,
        sample_rate: int = 24000,
        max_seq_len: int = 4096,
        apply_cuda_graph_subtalker: bool = True,
    ) -> "Qwen3TTSAccelPipeline":
        import torch
        from qwen_tts import Qwen3TTSModel

        ensure_cuda_available()

        device = "cuda:0"
        dtype = torch.bfloat16
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )

        preprocessor = Qwen3TTSPreprocessor(model=model)

        from .direct import create_main_talker_runner

        runner = create_main_talker_runner(
            model,
            max_seq_len=max_seq_len,
            max_batch_size=1,
            apply_cuda_graph_subtalker=apply_cuda_graph_subtalker,
        )

        return cls(
            model_path=model_path,
            model=model,
            preprocessor=preprocessor,
            runner=runner,
            sample_rate=sample_rate,
            device=device,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "main_talker": "direct_sdpa",
            "subtalker": "cuda_graph",
        }

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> bytes:
        voice_clone_prompt = self._build_voice_clone_prompt(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )
        request_id = f"speech-{uuid4()}"
        payload = self.preprocessor.prepare_inputs(
            request_id=request_id,
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )

        codec_frames, _ = self.runner.generate(
            inputs_embeds=payload.inputs_embeds,
            attention_mask=payload.attention_mask,
            trailing_text_hidden=payload.trailing_text_hidden,
            tts_pad_embed=payload.tts_pad_embed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        if not codec_frames:
            raise RuntimeError("No codec frames were generated.")

        ref_code = self._get_ref_code(payload.voice_clone_prompt)
        wavs, sr = self._decode_frames(codec_frames, ref_code)
        waveform = _coerce_waveform(wavs[0])
        return waveform_to_wav_bytes(waveform, sr)

    def synthesize_stream(
        self,
        *,
        text: str,
        language: str,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        chunk_size: int = 8,
    ) -> Iterator[bytes]:
        voice_clone_prompt = self._build_voice_clone_prompt(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )
        request_id = f"speech-stream-{uuid4()}"
        payload = self.preprocessor.prepare_inputs(
            request_id=request_id,
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )

        ref_code = self._get_ref_code(payload.voice_clone_prompt)

        # Accumulate all frames then decode once (neural codec uses overlapping
        # windows, so independent chunk decoding causes boundary artifacts).
        all_frames: list[list[int]] = []
        for frame in self.runner.generate_streaming(
            inputs_embeds=payload.inputs_embeds,
            attention_mask=payload.attention_mask,
            trailing_text_hidden=payload.trailing_text_hidden,
            tts_pad_embed=payload.tts_pad_embed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ):
            all_frames.append(frame)

        if all_frames:
            wavs, _sr = self._decode_frames(all_frames, ref_code)
            for wav in wavs:
                yield waveform_to_pcm_s16le_bytes(_coerce_waveform(wav))

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Voice clone prompt
    # ------------------------------------------------------------------

    def _build_voice_clone_prompt(
        self,
        *,
        ref_audio_path: str | None,
        ref_text: str | None,
    ) -> Any:
        if bool(ref_audio_path) ^ bool(ref_text):
            raise ValueError("ref_audio_path and ref_text must be provided together.")
        if not ref_audio_path and not ref_text:
            raise ValueError("ref_audio_path and ref_text are required.")

        if not Path(ref_audio_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        builder = getattr(self.model, "create_voice_clone_prompt", None)
        if not callable(builder):
            raise RuntimeError("Loaded model does not expose create_voice_clone_prompt().")

        return builder(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )

    # ------------------------------------------------------------------
    # Codec decode (same pattern as DirectTtsBackend._decode_frames)
    # ------------------------------------------------------------------

    def _get_speech_tokenizer(self) -> Any:
        if hasattr(self.model, "model"):
            return self.model.model.speech_tokenizer
        return getattr(self.model, "speech_tokenizer", None)

    def _get_ref_code(self, vcp: Any) -> Any:
        if not isinstance(vcp, dict):
            return None
        ref_codes = vcp.get("ref_code")
        if isinstance(ref_codes, list) and ref_codes and ref_codes[0] is not None:
            return ref_codes[0]
        return None

    def _decode_frames(
        self,
        codec_frames: list[list[int]],
        ref_code: Any | None,
    ) -> tuple[list[Any], int]:
        import torch

        speech_tok = self._get_speech_tokenizer()
        if speech_tok is None:
            raise RuntimeError("No speech tokenizer available for codec decoding")

        device = next(self.runner._talker.parameters()).device
        codes_tensor = torch.tensor(codec_frames, device=device, dtype=torch.long)

        if ref_code is not None:
            ref_code_t = ref_code.to(device) if hasattr(ref_code, "to") else torch.tensor(ref_code, device=device)
            if ref_code_t.dim() == 1:
                ref_code_t = ref_code_t.unsqueeze(-1)
            full_codes = torch.cat([ref_code_t, codes_tensor], dim=0)
        else:
            full_codes = codes_tensor

        wavs, sr = speech_tok.decode([{"audio_codes": full_codes}])
        wav = wavs[0]

        # Trim reference portion
        if ref_code is not None and full_codes.shape[0] > 0:
            ref_len = ref_code_t.shape[0]
            total_len = full_codes.shape[0]
            cut = int(ref_len / total_len * wav.shape[0])
            wav = wav[cut:]

        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()

        return [wav], sr


def _coerce_waveform(waveform: Any) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    return np.asarray(waveform, dtype=np.float32)
