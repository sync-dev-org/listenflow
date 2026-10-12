from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class TranscriptEventBase:
    utterance_id: int
    ts_offset_s: float


@dataclass(frozen=True, slots=True)
class UtteranceStart(TranscriptEventBase):
    kind: ClassVar[Literal["utterance_start"]] = "utterance_start"


@dataclass(frozen=True, slots=True)
class Partial(TranscriptEventBase):
    text: str
    speaker: int | str | None = None
    kind: ClassVar[Literal["partial"]] = "partial"


@dataclass(frozen=True, slots=True)
class Final(TranscriptEventBase):
    text: str
    speaker: int | str | None = None
    kind: ClassVar[Literal["final"]] = "final"


@dataclass(frozen=True, slots=True)
class UtteranceEnd(TranscriptEventBase):
    kind: ClassVar[Literal["utterance_end"]] = "utterance_end"


TranscriptEvent: TypeAlias = UtteranceStart | Partial | Final | UtteranceEnd
