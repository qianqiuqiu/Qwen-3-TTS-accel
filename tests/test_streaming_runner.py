from __future__ import annotations

from dataclasses import dataclass

from qwen3tts_accel.state import get_default_sequence_store
from qwen3tts_accel.vllm.runner import AsyncEngineRunner


@dataclass
class FakePayload:
    request_id: str
    engine_prompt: str
    voice_clone_prompt: dict | None = None


class FakeEngine:
    def __init__(self) -> None:
        self.request_ids: list[str] = []

    async def generate(self, prompt, sampling_params, request_id):
        self.request_ids.append(request_id)
        store = get_default_sequence_store()
        store.create(
            seq_id=request_id,
            trailing_text_hidden=None,
            tts_pad_embed=None,
            past_hidden=None,
        )
        store.append_codec_frame(request_id, [1, 2, 3])
        store.append_codec_frame(request_id, [4, 5, 6])
        yield type("Step", (), {"outputs": ["partial"]})()
        store.append_codec_frame(request_id, [7, 8, 9])
        yield type("Step", (), {"outputs": ["final"]})()


def test_async_engine_runner_streams_codec_chunks_before_completion():
    observed = []

    def decoder(*, codec_tokens, sample_rate):
        observed.append(codec_tokens)
        return [codec_tokens]

    runner = AsyncEngineRunner(
        engine=FakeEngine(),
        engine_args=None,
        decoder=decoder,
        sample_rate=24000,
    )
    runner._build_sampling_params = lambda values: values
    payload = FakePayload(request_id="req-stream", engine_prompt="hello")

    chunks = list(runner.synthesize_stream(payload=payload, affect_style={}, buffer_frames=2))

    assert len(chunks) == 2
    assert chunks[0] == ([[[1, 2, 3], [4, 5, 6]]], 24000)
    assert chunks[1] == ([[[7, 8, 9]]], 24000)
    assert observed == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9]],
    ]
