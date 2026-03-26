from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import validate_bearer_token
from .pipeline import Qwen3TTSAccelPipeline
from .schemas import ErrorResponse, HealthResponse, MetaResponse, SpeechRequest


def create_app(pipeline: Qwen3TTSAccelPipeline, api_key: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline
        app.state.api_key = api_key
        yield
        pipeline.close()

    app = FastAPI(title="qwen3tts-accel", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.api_key = api_key

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok")

    @app.get("/meta", response_model=MetaResponse)
    def meta():
        return MetaResponse(**app.state.pipeline.metadata())

    @app.post("/v1/audio/speech")
    def speech(request: SpeechRequest, authorization: str | None = Header(default=None)):
        auth_error = _validate_auth(authorization, app.state.api_key)
        if auth_error is not None:
            return auth_error

        try:
            wav_bytes = app.state.pipeline.synthesize(
                text=request.text,
                language=request.language,
                ref_audio_path=request.ref_audio_path,
                ref_text=request.ref_text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            )
        except FileNotFoundError as exc:
            return _error_response(404, "file_not_found", str(exc))
        except ValueError as exc:
            return _error_response(400, "invalid_request", str(exc))
        except Exception as exc:
            return _error_response(500, "inference_failed", str(exc))

        return Response(content=wav_bytes, media_type="audio/wav")

    @app.post("/v1/audio/speech/stream")
    def speech_stream(request: SpeechRequest, authorization: str | None = Header(default=None)):
        auth_error = _validate_auth(authorization, app.state.api_key)
        if auth_error is not None:
            return auth_error

        try:
            stream = app.state.pipeline.synthesize_stream(
                text=request.text,
                language=request.language,
                ref_audio_path=request.ref_audio_path,
                ref_text=request.ref_text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                chunk_size=request.chunk_size,
            )
        except FileNotFoundError as exc:
            return _error_response(404, "file_not_found", str(exc))
        except ValueError as exc:
            return _error_response(400, "invalid_request", str(exc))
        except Exception as exc:
            return _error_response(500, "inference_failed", str(exc))

        return StreamingResponse(
            stream,
            media_type="application/octet-stream",
            headers={
                "X-Audio-Sample-Rate": str(app.state.pipeline.sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Format": "pcm_s16le",
            },
        )

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS acceleration API server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    pipeline = Qwen3TTSAccelPipeline.from_pretrained(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        sample_rate=args.sample_rate,
    )
    app = create_app(pipeline, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _validate_auth(authorization: str | None, api_key: str | None) -> JSONResponse | None:
    try:
        validate_bearer_token(authorization, api_key)
    except PermissionError as exc:
        return _error_response(401, "unauthorized", str(exc))
    return None


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse.model_validate({"error": {"code": code, "message": message}})
    return JSONResponse(status_code=status_code, content=payload.model_dump())
