from __future__ import annotations

import asyncio
import importlib.util
import queue
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from ..preprocess.preprocessor import PrefillPayload
from ..state.sequence_state import get_default_sequence_store
from .config import VllmEngineConfig


class VllmRunner(Protocol):
    def synthesize(self, *, payload: PrefillPayload, affect_style: dict[str, Any]):
        """Run the vLLM-backed synthesis path and return waveform data."""

    def synthesize_stream(
        self,
        *,
        payload: PrefillPayload,
        affect_style: dict[str, Any],
        buffer_frames: int = 8,
    ):
        """Yield waveform chunks as soon as codec frames are available."""


@dataclass
class VllmSynthesisRequest:
    request_id: str
    prompt: str
    sampling_params: dict[str, Any]
    payload: PrefillPayload


def collect_sampling_params(affect_style: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": affect_style.get("temperature", 0.9),
        "top_p": affect_style.get("top_p", 1.0),
        "top_k": affect_style.get("top_k", -1),
        "max_tokens": int(affect_style.get("max_tokens", 1)),
    }


def decode_codec_tokens(
    *,
    codec_tokens: list[list[int]],
    decoder: Any,
    sample_rate: int,
    voice_clone_prompt: Any | None = None,
):
    if decoder is None:
        raise NotImplementedError("No codec decoder configured for vLLM TTS synthesis.")
    codes_for_decode = prepare_codes_for_decode(
        codec_tokens=codec_tokens,
        voice_clone_prompt=voice_clone_prompt,
    )
    waveform_batch = decoder(codec_tokens=codes_for_decode, sample_rate=sample_rate)
    return postprocess_waveforms(
        waveform_batch=waveform_batch,
        generated_codec_tokens=codec_tokens,
        voice_clone_prompt=voice_clone_prompt,
    ), sample_rate


def prepare_codes_for_decode(
    *,
    codec_tokens: list[list[int]],
    voice_clone_prompt: Any | None = None,
) -> list[list[int]]:
    ref_code_list = None
    if isinstance(voice_clone_prompt, dict):
        ref_code_list = voice_clone_prompt.get("ref_code")

    prepared: list[list[int]] = []
    for index, codes in enumerate(codec_tokens):
        merged = list(codes)
        ref_code = None
        if isinstance(ref_code_list, list) and index < len(ref_code_list):
            ref_code = ref_code_list[index]

        if ref_code is not None:
            ref_values = _tensor_like_to_list(ref_code)
            merged = ref_values + merged
        prepared.append(merged)
    return prepared


def postprocess_waveforms(
    *,
    waveform_batch: list[Any],
    generated_codec_tokens: list[list[int]],
    voice_clone_prompt: Any | None = None,
) -> list[Any]:
    ref_code_list = None
    if isinstance(voice_clone_prompt, dict):
        ref_code_list = voice_clone_prompt.get("ref_code")

    processed = []
    for index, waveform in enumerate(waveform_batch):
        ref_code = None
        if isinstance(ref_code_list, list) and index < len(ref_code_list):
            ref_code = ref_code_list[index]

        if ref_code is None:
            processed.append(waveform)
            continue

        ref_len = len(_tensor_like_to_list(ref_code))
        total_len = ref_len + len(generated_codec_tokens[index])
        if total_len <= 0 or not hasattr(waveform, "__len__"):
            processed.append(waveform)
            continue

        cut = int(ref_len / total_len * len(waveform))
        processed.append(waveform[cut:])
    return processed


def _tensor_like_to_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        result = value.tolist()
        if result and isinstance(result[0], list):
            return [int(item) for row in result for item in row]
        return [int(item) for item in result]
    return [int(item) for item in value]


@dataclass
class AsyncEngineRunner:
    """
    Thin placeholder around a future `vllm.AsyncLLMEngine`.

    Keeps engine construction and codec-token collection isolated so the
    optimization framework can benchmark the vLLM path directly.
    """

    engine: Any
    engine_args: Any
    decoder: Any | None = None
    sample_rate: int = 24000

    def synthesize(self, *, payload: PrefillPayload, affect_style: dict[str, Any]):
        request = VllmSynthesisRequest(
            request_id=payload.request_id,
            prompt=payload.engine_prompt,
            sampling_params=collect_sampling_params(affect_style),
            payload=payload,
        )
        try:
            codec_tokens = asyncio.run(self._collect_codec_tokens(request))
            return decode_codec_tokens(
                codec_tokens=codec_tokens,
                decoder=self.decoder,
                sample_rate=self.sample_rate,
                voice_clone_prompt=payload.voice_clone_prompt,
            )
        finally:
            get_default_sequence_store().pop(request.request_id)

    def synthesize_stream(
        self,
        *,
        payload: PrefillPayload,
        affect_style: dict[str, Any],
        buffer_frames: int = 8,
    ):
        request = VllmSynthesisRequest(
            request_id=payload.request_id,
            prompt=payload.engine_prompt,
            sampling_params=collect_sampling_params(affect_style),
            payload=payload,
        )
        event_queue: queue.Queue = queue.Queue()
        sentinel = object()

        def _run() -> None:
            try:
                asyncio.run(
                    self._produce_streaming_waveforms(
                        request,
                        buffer_frames=max(buffer_frames, 1),
                        event_queue=event_queue,
                    )
                )
            except Exception as exc:
                event_queue.put(exc)
            finally:
                event_queue.put(sentinel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _collect_codec_tokens(self, request: VllmSynthesisRequest) -> list[list[int]]:
        sampling_params = self._build_sampling_params(request.sampling_params)
        outputs = self.engine.generate(
            request.prompt,
            sampling_params,
            request_id=request.request_id,
        )

        final_output = None
        async for item in outputs:
            final_output = item

        if final_output is None:
            raise RuntimeError("vLLM returned no outputs for synthesis request.")

        output_list = getattr(final_output, "outputs", None) or []
        if not output_list:
            raise RuntimeError("vLLM synthesis request produced no candidate outputs.")

        state = get_default_sequence_store().get(request.request_id)
        if state is not None and state.codec_frames:
            return state.codec_frames

        raise NotImplementedError(
            "vLLM request finished without full codec frames from the custom plugin."
        )

    async def _produce_streaming_waveforms(
        self,
        request: VllmSynthesisRequest,
        *,
        buffer_frames: int,
        event_queue: queue.Queue,
    ) -> None:
        sampling_params = self._build_sampling_params(request.sampling_params)
        outputs = self.engine.generate(
            request.prompt,
            sampling_params,
            request_id=request.request_id,
        )

        emitted = 0

        async for _item in outputs:
            state = get_default_sequence_store().get(request.request_id)
            if state is None or not state.codec_frames:
                continue
            emitted, decoded_chunks = self._decode_available_chunks(
                codec_frames=state.codec_frames,
                emitted=emitted,
                buffer_frames=buffer_frames,
                voice_clone_prompt=request.payload.voice_clone_prompt,
            )
            for chunk in decoded_chunks:
                event_queue.put(chunk)

        state = get_default_sequence_store().get(request.request_id)
        if state is not None and state.codec_frames:
            emitted, decoded_chunks = self._decode_available_chunks(
                codec_frames=state.codec_frames,
                emitted=emitted,
                buffer_frames=buffer_frames,
                voice_clone_prompt=request.payload.voice_clone_prompt,
                flush=True,
            )
            for chunk in decoded_chunks:
                event_queue.put(chunk)
        get_default_sequence_store().pop(request.request_id)

    def _decode_available_chunks(
        self,
        *,
        codec_frames: list[list[int]],
        emitted: int,
        buffer_frames: int,
        voice_clone_prompt: Any | None,
        flush: bool = False,
    ) -> tuple[int, list[tuple[list[Any], int]]]:
        chunks: list[tuple[list[Any], int]] = []
        use_prompt = voice_clone_prompt

        while len(codec_frames) - emitted >= buffer_frames:
            frame_block = codec_frames[emitted : emitted + buffer_frames]
            chunks.append(
                decode_codec_tokens(
                    codec_tokens=frame_block,
                    decoder=self.decoder,
                    sample_rate=self.sample_rate,
                    voice_clone_prompt=use_prompt,
                )
            )
            emitted += buffer_frames
            use_prompt = None

        if flush and emitted < len(codec_frames):
            frame_block = codec_frames[emitted:]
            chunks.append(
                decode_codec_tokens(
                    codec_tokens=frame_block,
                    decoder=self.decoder,
                    sample_rate=self.sample_rate,
                    voice_clone_prompt=use_prompt,
                )
            )
            emitted = len(codec_frames)

        return emitted, chunks

    def _build_sampling_params(self, values: dict[str, Any]):
        try:
            from vllm import SamplingParams
        except ImportError:
            from vllm.sampling_params import SamplingParams

        return SamplingParams(**values)


def create_vllm_runner(config: VllmEngineConfig) -> VllmRunner | None:
    if importlib.util.find_spec("vllm") is None:
        return None

    from vllm import AsyncEngineArgs

    try:
        from vllm import AsyncLLMEngine
    except ImportError:
        from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=config.model_path,
        trust_remote_code=True,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        enforce_eager=config.enforce_eager,
        **config.extra_engine_args,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return AsyncEngineRunner(
        engine=engine,
        engine_args=engine_args,
        decoder=config.codec_decoder,
        sample_rate=config.sample_rate,
    )
