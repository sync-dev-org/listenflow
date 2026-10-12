from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class EngineState(StrEnum):
    """The engine's preparation and closure state."""

    UNPREPARED = "unprepared"
    READY = "ready"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class BiasApplicationSnapshot:
    session_id: int
    kind: Literal["none", "prompt"]
    prompt: str | None
    applied_value: str | None
    applied: bool
    evidence_method: Literal["verified_apply_path"]
    evidence_detail: str


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    state: EngineState
    busy: bool
    last_progress_monotonic_s: float | None
    processed_pcm_chunks: int = 0
    emitted_events: int = 0
    bias_application: BiasApplicationSnapshot | None = None
    decoding_utterance_id: int | None = None
    pending_decode_utterances: int = 0
