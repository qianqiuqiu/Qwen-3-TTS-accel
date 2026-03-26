from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SpeechRequest(BaseModel):
    text: str
    language: str = "Chinese"
    ref_audio_path: str | None = None
    ref_text: str | None = None
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    audio_format: Literal["wav"] = "wav"
    chunk_size: int = Field(default=8, ge=1, le=32)

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text must not be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("temperature must be greater than 0")
        return value

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, value: float) -> float:
        if not 0 < value <= 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        return value


class HealthResponse(BaseModel):
    status: str


class MetaResponse(BaseModel):
    model_name: str
    model_path: str
    sample_rate: int
    device: str
    main_talker: str
    subtalker: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
