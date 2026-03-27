from __future__ import annotations

from fastapi.testclient import TestClient

from qwen3tts_accel.api_server import create_app


class RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.sample_rate = 24000

    def metadata(self) -> dict:
        return {
            "model_name": "Qwen3-TTS-12Hz-1.7B-Base",
            "model_path": "/models/qwen3",
            "sample_rate": 24000,
            "device": "cuda:0",
            "main_talker": "direct_sdpa",
            "subtalker": "cuda_graph",
        }

    def synthesize(self, **kwargs) -> bytes:
        self.calls.append(kwargs)
        return b"RIFFfakewav"

    def synthesize_stream(self, **kwargs):
        self.calls.append({"stream": True, **kwargs})
        yield b"\x01\x02"
        yield b"\x03\x04"


def test_health_endpoint_reports_ok():
    app = create_app(RecordingPipeline())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_meta_endpoint_uses_pipeline_metadata():
    app = create_app(RecordingPipeline())
    client = TestClient(app)

    response = client.get("/meta")

    assert response.status_code == 200
    assert response.json()["main_talker"] == "direct_sdpa"
    assert response.json()["subtalker"] == "cuda_graph"


def test_speech_endpoint_requires_bearer_token_when_api_key_configured():
    app = create_app(RecordingPipeline(), api_key="secret")
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={
            "text": "hello",
            "language": "English",
            "ref_audio_path": "D:/ref.wav",
            "ref_text": "hello",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_speech_endpoint_returns_wav_bytes_and_calls_pipeline():
    pipeline = RecordingPipeline()
    app = create_app(pipeline, api_key="secret")
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer secret"},
        json={
            "text": "hello",
            "language": "English",
            "ref_audio_path": "D:/ref.wav",
            "ref_text": "hello",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 32,
            "repetition_penalty": 1.1,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFfakewav"
    assert pipeline.calls == [
        {
            "text": "hello",
            "language": "English",
            "ref_audio_path": "D:/ref.wav",
            "ref_text": "hello",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 32,
            "repetition_penalty": 1.1,
        }
    ]


def test_stream_endpoint_returns_pcm_chunks_and_headers():
    pipeline = RecordingPipeline()
    app = create_app(pipeline)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/audio/speech/stream",
        json={
            "text": "hello",
            "language": "English",
            "ref_audio_path": "D:/ref.wav",
            "ref_text": "hello",
            "chunk_size": 4,
        },
    ) as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["x-audio-format"] == "pcm_s16le"
    assert body == b"\x01\x02\x03\x04"
    assert pipeline.calls == [
        {
            "stream": True,
            "text": "hello",
            "language": "English",
            "ref_audio_path": "D:/ref.wav",
            "ref_text": "hello",
            "temperature": 0.9,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "chunk_size": 4,
        }
    ]
