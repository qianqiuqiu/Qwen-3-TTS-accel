from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SequenceState:
    seq_id: str
    trailing_text_hidden: Any
    tts_pad_embed: Any
    past_hidden: Any
    generation_step: int = 0
    codec_frames: list[list[int]] | None = None


class SequenceStateStore:
    def __init__(self) -> None:
        self._states: dict[str, SequenceState] = {}

    def create(
        self,
        *,
        seq_id: str,
        trailing_text_hidden: Any,
        tts_pad_embed: Any,
        past_hidden: Any,
    ) -> SequenceState:
        state = SequenceState(
            seq_id=seq_id,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            past_hidden=past_hidden,
            codec_frames=[],
        )
        self._states[seq_id] = state
        return state

    def get(self, seq_id: str) -> SequenceState | None:
        return self._states.get(seq_id)

    def advance(self, seq_id: str, past_hidden: Any) -> SequenceState:
        state = self._states[seq_id]
        state.generation_step += 1
        state.past_hidden = past_hidden
        return state

    def pop(self, seq_id: str) -> SequenceState | None:
        return self._states.pop(seq_id, None)

    def append_codec_frame(self, seq_id: str, codec_frame: list[int]) -> SequenceState:
        state = self._states[seq_id]
        state.codec_frames.append(codec_frame)
        return state


_DEFAULT_SEQUENCE_STORE: SequenceStateStore | None = None


def get_default_sequence_store() -> SequenceStateStore:
    global _DEFAULT_SEQUENCE_STORE
    if _DEFAULT_SEQUENCE_STORE is None:
        _DEFAULT_SEQUENCE_STORE = SequenceStateStore()
    return _DEFAULT_SEQUENCE_STORE
