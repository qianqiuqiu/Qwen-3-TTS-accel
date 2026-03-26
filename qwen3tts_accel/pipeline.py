from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

from .audio import waveform_to_pcm_s16le_bytes, waveform_to_wav_bytes
from .preprocess import Qwen3TTSPreprocessor
from .subtalker import patch_code_predictor_cuda_graph
from .vllm import VllmEngineConfig, create_vllm_runner, register_qwen3_tts_model


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
        tokenizer: Any,
        preprocessor: Qwen3TTSPreprocessor,
        subtalker_runner: Any,
        vllm_runner: Any,
        sample_rate: int,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.model_name = Path(model_path).name
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.subtalker_runner = subtalker_runner
        self.vllm_runner = vllm_runner
        self.sample_rate = sample_rate
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 8,
        sample_rate: int = 24000,
    ) -> "Qwen3TTSAccelPipeline":
        import torch
        from qwen_tts import Qwen3TTSModel
        from transformers import AutoTokenizer

        ensure_cuda_available()
        register_qwen3_tts_model()

        device = "cuda:0"
        dtype = torch.bfloat16
        attn_impl = "sdpa"
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        tokenizer = getattr(model, "processor", None) or AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        preprocessor = Qwen3TTSPreprocessor(model=model, tokenizer=tokenizer)
        subtalker_runner = patch_code_predictor_cuda_graph(model.model.talker, max_batch_size=1, capture=True)

        codec_decoder = _build_codec_decoder_from_model(model.model if hasattr(model, "model") else model)
        vllm_config = VllmEngineConfig(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            sample_rate=sample_rate,
            codec_decoder=codec_decoder,
            helper_model=model,
        )
        vllm_runner = create_vllm_runner(vllm_config)
        if vllm_runner is None:
            raise RuntimeError("vLLM is not available in the current environment.")

        return cls(
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            subtalker_runner=subtalker_runner,
            vllm_runner=vllm_runner,
            sample_rate=sample_rate,
            device=device,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "main_talker": "vllm_plugin",
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
        waveforms, sample_rate = self.vllm_runner.synthesize(
            payload=payload,
            affect_style={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            },
        )
        if not waveforms:
            raise RuntimeError("No audio was generated.")
        waveform = _coerce_waveform(waveforms[0])
        return waveform_to_wav_bytes(waveform, sample_rate)

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
        for waveforms, _sample_rate in self.vllm_runner.synthesize_stream(
            payload=payload,
            affect_style={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            },
            buffer_frames=chunk_size,
        ):
            for waveform in waveforms:
                yield waveform_to_pcm_s16le_bytes(_coerce_waveform(waveform))

    def close(self) -> None:
        return None

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
            reference_audio=ref_audio_path,
            reference_text=ref_text,
        )


def _build_codec_decoder_from_model(model: Any):
    candidate_paths = [
        ("speech_tokenizer", "decode"),
        ("speech_tokenizer", "decode_codec"),
        ("audio_tokenizer", "decode"),
        ("codec_decoder", "__call__"),
    ]

    for attr_name, method_name in candidate_paths:
        owner = getattr(model, attr_name, None)
        if owner is None:
            continue
        if method_name == "__call__" and callable(owner):
            return lambda *, codec_tokens, sample_rate: owner(codec_tokens)
        method = getattr(owner, method_name, None)
        if callable(method):
            if attr_name in {"speech_tokenizer", "audio_tokenizer"}:
                return lambda *, codec_tokens, sample_rate, _method=method: _decode_with_tokenizer_method(_method, codec_tokens)
            return lambda *, codec_tokens, sample_rate, _method=method: _method(codec_tokens)
    return None


def _decode_with_tokenizer_method(method, codec_tokens):
    result = method([{"audio_codes": codes} for codes in codec_tokens])
    if isinstance(result, tuple) and len(result) == 2:
        wavs, _sample_rate = result
        return wavs
    return result


def _coerce_waveform(waveform: Any) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    return np.asarray(waveform, dtype=np.float32)
