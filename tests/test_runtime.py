from __future__ import annotations

import inspect
from unittest.mock import patch

from qwen3tts_accel import register_qwen3_tts_model
from qwen3tts_accel.api_server import create_app
from qwen3tts_accel.pipeline import ensure_cuda_available
from qwen3tts_accel.benchmarks import BenchmarkCase, collect_sampling_params
from qwen3tts_accel.decode import decode_codec_tokens, prepare_codes_for_decode
from qwen3tts_accel.state import SequenceStateStore
from qwen3tts_accel.vllm.runner import AsyncEngineRunner


def test_sequence_state_store_tracks_lifecycle():
    store = SequenceStateStore()
    state = store.create(
        seq_id="req-1",
        trailing_text_hidden=["hidden"],
        tts_pad_embed="pad",
        past_hidden="past",
    )

    assert state.seq_id == "req-1"
    assert store.get("req-1") is state
    assert state.generation_step == 0

    store.advance("req-1", "new-hidden")
    assert store.get("req-1").generation_step == 1
    assert store.get("req-1").past_hidden == "new-hidden"

    removed = store.pop("req-1")
    assert removed is state
    assert store.get("req-1") is None


def test_sequence_state_store_tracks_codec_frames():
    store = SequenceStateStore()
    store.create(
        seq_id="req-1",
        trailing_text_hidden=["hidden"],
        tts_pad_embed="pad",
        past_hidden="past",
    )

    store.append_codec_frame("req-1", [10, 11, 12])
    store.append_codec_frame("req-1", [20, 21, 22])

    assert store.get("req-1").codec_frames == [
        [10, 11, 12],
        [20, 21, 22],
    ]


def test_register_qwen3_tts_model_returns_false_without_vllm():
    with patch("qwen3tts_accel.vllm.importlib.util.find_spec", return_value=None):
        assert register_qwen3_tts_model() is False


def test_collect_sampling_params_uses_tts_defaults():
    params = collect_sampling_params({"temperature": 0.7})

    assert params["temperature"] == 0.7
    assert params["top_p"] == 1.0
    assert params["max_tokens"] == 1


def test_benchmark_case_describes_optimization_surface():
    case = BenchmarkCase(
        name="cuda-graph-subtalker",
        description="Measure CUDA graph subtalker throughput",
        runner=lambda: None,
    )

    assert case.name == "cuda-graph-subtalker"
    assert "subtalker" in case.description


def test_decode_codec_tokens_uses_decoder_callback():
    observed = {}

    def decoder(*, codec_tokens, sample_rate):
        observed["codec_tokens"] = codec_tokens
        observed["sample_rate"] = sample_rate
        return ["waveform"]

    result = decode_codec_tokens(
        codec_tokens=[[1, 2, 3]],
        decoder=decoder,
        sample_rate=24000,
    )

    assert result == (["waveform"], 24000)
    assert observed == {
        "codec_tokens": [[1, 2, 3]],
        "sample_rate": 24000,
    }


def test_prepare_codes_for_decode_prepends_ref_code():
    class FakeTensor:
        def __init__(self, values):
            self.values = values
            self.device = "cpu"
            self.shape = (len(values),)

        def tolist(self):
            return list(self.values)

        def to(self, _device):
            return self

    codes = prepare_codes_for_decode(
        codec_tokens=[[3, 4]],
        voice_clone_prompt={"ref_code": [FakeTensor([1, 2])]},
    )

    assert len(codes) == 1
    assert codes[0] == [1, 2, 3, 4]


def test_sync_synthesize_cleans_up_sequence_state_after_completion():
    from qwen3tts_accel.state import sequence_state as state_module

    class FakeEngine:
        async def generate(self, prompt, sampling_params, request_id):
            store = state_module.get_default_sequence_store()
            store.create(
                seq_id=request_id,
                trailing_text_hidden=None,
                tts_pad_embed=None,
                past_hidden=None,
            )
            store.append_codec_frame(request_id, [1, 2, 3])
            yield type("Step", (), {"outputs": ["final"]})()

    state_module._DEFAULT_SEQUENCE_STORE = SequenceStateStore()

    def decoder(*, codec_tokens, sample_rate):
        return ["waveform"]

    runner = AsyncEngineRunner(
        engine=FakeEngine(),
        engine_args=None,
        decoder=decoder,
        sample_rate=24000,
    )
    runner._build_sampling_params = lambda values: values
    payload = type("Payload", (), {
        "request_id": "req-sync",
        "engine_prompt": "hello",
        "voice_clone_prompt": None,
    })()

    result = runner.synthesize(payload=payload, affect_style={})

    assert result == (["waveform"], 24000)
    assert state_module.get_default_sequence_store().get("req-sync") is None


def test_create_app_registers_sync_route_handlers():
    class Pipeline:
        sample_rate = 24000

        def metadata(self):
            return {
                "model_name": "model",
                "model_path": "/tmp/model",
                "sample_rate": 24000,
                "device": "cuda:0",
                "main_talker": "vllm_plugin",
                "subtalker": "cuda_graph",
            }

        def close(self):
            return None

        def synthesize(self, **kwargs):
            return b"wav"

        def synthesize_stream(self, **kwargs):
            yield b"pcm"

    app = create_app(Pipeline())
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}

    assert not inspect.iscoroutinefunction(routes["/v1/audio/speech"])
    assert not inspect.iscoroutinefunction(routes["/v1/audio/speech/stream"])


def test_ensure_cuda_available_raises_clear_error_when_cuda_missing():
    with patch("torch.cuda.is_available", return_value=False):
        try:
            ensure_cuda_available()
        except RuntimeError as exc:
            assert "CUDA" in str(exc)
        else:
            raise AssertionError("ensure_cuda_available() should fail when CUDA is unavailable")
