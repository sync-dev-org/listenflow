from __future__ import annotations

import asyncio
import math
import struct
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from listenflow._async import cancel_once, finish
from listenflow.events import Final, Partial, TranscriptEvent, UtteranceEnd, UtteranceStart


SAMPLE_RATE_HZ = 16_000
DEFAULT_VOICE_THRESHOLD = 500 / 32768


class EngineTransport(StrEnum):
    IN_PROCESS = "in_process"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    HTTP_UTTERANCE = "http_utterance"


class EngineInputFormat(StrEnum):
    PCM_S16LE = "pcm_s16le"
    PCM_F32 = "pcm_f32"
    BASE64_PCM16 = "base64_pcm16"
    WAV_PCM_S16LE = "wav_pcm_s16le"


class Segmentation(StrEnum):
    """Choose automatic utterance boundaries or one caller-selected utterance."""

    AUTO = "auto"
    SINGLE = "single"


_RAW_PCM_INPUT_FORMATS = (
    EngineInputFormat.PCM_S16LE,
    EngineInputFormat.PCM_F32,
)
_SAMPLE_WIDTH_BYTES = {
    EngineInputFormat.PCM_S16LE: 2,
    EngineInputFormat.PCM_F32: 4,
}


def validate_input_format(
    input_format: object,
    *,
    target: str,
    accepted: tuple[EngineInputFormat, ...],
) -> EngineInputFormat:
    accepted_values = ", ".join(item.value for item in accepted)
    if not isinstance(input_format, EngineInputFormat) or input_format not in accepted:
        raise ValueError(
            f"{target} input_format {input_format!r} is not supported; "
            f"accepted formats: {accepted_values}"
        )
    return input_format


def validate_voice_threshold(voice_threshold: object) -> float:
    if (
        type(voice_threshold) is not float
        or not math.isfinite(voice_threshold)
        or not 0.0 <= voice_threshold <= 1.0
    ):
        raise ValueError(
            "voice_threshold must be a finite float in normalized full-scale "
            "range [0.0, 1.0]; legacy int16 amplitude 500 is approximately "
            "0.01526 (500 / 32768)"
        )
    return voice_threshold


@dataclass(frozen=True, slots=True)
class UtteranceBoundarySettings:
    """Configure voice detection and complete-utterance decoding.

    ``min_voiced_s`` is the minimum voiced span to decode, in finite, non-negative
    seconds. It defaults to ``0.0``, which decodes every utterance. The span runs
    from the start of the first voiced chunk to the end of the last voiced chunk,
    excluding preroll and trailing silence. Shorter utterances emit only
    ``UtteranceStart`` and ``UtteranceEnd``, including at end of input; their start
    event is emitted as soon as the first voiced chunk arrives.
    """

    silence_threshold_s: float = 0.8
    voice_threshold: float = DEFAULT_VOICE_THRESHOLD
    preroll_s: float = 0.3
    min_voiced_s: float = 0.0

    def __post_init__(self) -> None:
        for name, allow_zero in (
            ("silence_threshold_s", False),
            ("preroll_s", True),
            ("min_voiced_s", True),
        ):
            value = getattr(self, name)
            range_description = "non-negative" if allow_zero else "positive"
            error_message = f"{name} must be a finite {range_description} number of seconds"
            if type(value) not in (int, float):
                raise ValueError(error_message)
            try:
                seconds = float(value)
            except OverflowError as error:
                raise ValueError(error_message) from error
            if not math.isfinite(seconds) or (seconds < 0 if allow_zero else seconds <= 0):
                raise ValueError(error_message)
            if name == "preroll_s" and not math.isfinite(seconds * SAMPLE_RATE_HZ):
                raise ValueError(
                    f"{name} must produce a finite sample count at {SAMPLE_RATE_HZ} Hz"
                )
            object.__setattr__(self, name, seconds)
        validate_voice_threshold(self.voice_threshold)


def validate_pcm_chunk(chunk: bytes, *, input_format: EngineInputFormat) -> None:
    width = _SAMPLE_WIDTH_BYTES[input_format]
    if len(chunk) % width:
        raise ValueError(
            f"{input_format.value} chunk length must be a multiple of {width} bytes "
            "and contain whole samples"
        )


def _normalized_samples(
    chunk: bytes,
    *,
    input_format: EngineInputFormat,
) -> object:
    if input_format is EngineInputFormat.PCM_S16LE:
        return (sample[0] / 32768.0 for sample in struct.iter_unpack("<h", chunk))
    return (sample[0] for sample in struct.iter_unpack("<f", chunk))


def has_voice(
    chunk: bytes,
    *,
    input_format: EngineInputFormat,
    voice_threshold: float,
) -> bool:
    input_format = validate_input_format(
        input_format,
        target="has_voice",
        accepted=_RAW_PCM_INPUT_FORMATS,
    )
    voice_threshold = validate_voice_threshold(voice_threshold)
    validate_pcm_chunk(chunk, input_format=input_format)
    return any(
        abs(sample) >= voice_threshold
        for sample in _normalized_samples(chunk, input_format=input_format)
    )


class PartialMode(StrEnum):
    REPLACE = "replace"
    APPEND = "append"
    COMMIT_ONLY = "commit_only"
    NONE = "none"


class FinalizationMode(StrEnum):
    ENGINE_SEGMENT = "engine_segment"
    ENGINE_ENDPOINT = "engine_endpoint"
    CLIENT_FLUSH = "client_flush"
    LOCAL_VAD = "local_vad"


class TimestampDetail(StrEnum):
    NONE = "none"
    SEGMENT = "segment"
    WORD = "word"
    TOKEN = "token"
    SAMPLE_OFFSET = "sample_offset"


class DiarizationMode(StrEnum):
    NONE = "none"
    SEGMENT_LABEL = "segment_label"
    WORD_LABEL = "word_label"
    MUTABLE_SPANS = "mutable_spans"


@dataclass(frozen=True, slots=True)
class OfflineModelDirectorySource:
    directory_path: str

    def __post_init__(self) -> None:
        if not Path(self.directory_path).is_absolute():
            raise ValueError("directory_path must be an absolute path")


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    transport: EngineTransport
    input_format: EngineInputFormat
    partial_mode: PartialMode
    finalization: frozenset[FinalizationMode]
    timestamp_detail: TimestampDetail
    diarization: DiarizationMode
    languages: frozenset[str]
    auto_language: bool
    single_utterance: bool = False

    def __post_init__(self) -> None:
        finalization = frozenset(self.finalization)
        if not finalization:
            raise ValueError("finalization must contain at least one mode")
        object.__setattr__(self, "finalization", finalization)
        languages = frozenset(self.languages)
        if not all(isinstance(language, str) for language in languages):
            raise TypeError("languages must contain str values")
        object.__setattr__(self, "languages", languages)
        if type(self.auto_language) is not bool:
            raise TypeError("auto_language must be bool")
        if type(self.single_utterance) is not bool:
            raise TypeError("single_utterance must be bool")
        expected_types = {
            "transport": EngineTransport,
            "input_format": EngineInputFormat,
            "partial_mode": PartialMode,
            "timestamp_detail": TimestampDetail,
            "diarization": DiarizationMode,
        }
        for attribute, expected_type in expected_types.items():
            if not isinstance(getattr(self, attribute), expected_type):
                raise TypeError(f"{attribute} must be {expected_type.__name__}")
        if not all(isinstance(mode, FinalizationMode) for mode in finalization):
            raise TypeError("finalization must contain FinalizationMode values")


@dataclass(frozen=True, slots=True)
class ProvisionalObservation:
    text: str
    speaker: int | str | None = None
    source_offset_s: float | None = None


@dataclass(frozen=True, slots=True)
class FinalObservation:
    text: str
    source_offset_s: float
    speaker: int | str | None = None


@dataclass(frozen=True, slots=True)
class ClosedUtterance:
    utterance_id: int
    pcm: bytes
    start_offset_s: float
    end_offset_s: float


@dataclass(slots=True)
class _UtteranceEventState:
    utterance_id: int
    in_utterance: bool
    emitted_start: bool
    emitted_partial: bool
    emitted_final: bool
    last_partial_text: str
    last_partial_ts_offset_s: float | None
    last_partial_speaker: int | str | None
    latest_provisional_finalized: bool
    last_final_text: str
    seen_final_observations: set[FinalObservation]
    start_offset_s: float | None
    end_offset_s: float | None


@dataclass(slots=True)
class UtteranceBoundarySession:
    silence_threshold_s: float
    voice_threshold: float
    input_format: EngineInputFormat = EngineInputFormat.PCM_S16LE
    utterance_id: int = 1
    input_offset_s: float = 0.0
    last_voice_offset_s: float | None = None
    pending_end_ts: float | None = None
    in_utterance: bool = False
    emitted_start: bool = False
    emitted_partial: bool = False
    emitted_final: bool = False
    input_finished: bool = False
    last_partial_text: str = ""
    last_partial_ts_offset_s: float | None = None
    last_partial_speaker: int | str | None = None
    _latest_provisional_finalized: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _last_final_text: str = field(
        default="",
        init=False,
        repr=False,
    )
    _seen_final_observations: set[FinalObservation] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _current_start_offset_s: float | None = field(default=None, init=False, repr=False)
    _closed_utterances: dict[int, _UtteranceEventState] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        validate_input_format(
            self.input_format,
            target="UtteranceBoundarySession",
            accepted=_RAW_PCM_INPUT_FORMATS,
        )
        self.voice_threshold = validate_voice_threshold(self.voice_threshold)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "input_format":
            try:
                object.__getattribute__(self, name)
            except AttributeError:
                pass
            else:
                raise AttributeError("UtteranceBoundarySession input_format is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "input_format":
            raise AttributeError("UtteranceBoundarySession input_format is immutable")
        object.__delattr__(self, name)

    def observe_pcm(self, chunk: bytes) -> list[TranscriptEvent]:
        validate_pcm_chunk(chunk, input_format=self.input_format)
        start_s = self.input_offset_s
        duration_s = (len(chunk) // _SAMPLE_WIDTH_BYTES[self.input_format]) / SAMPLE_RATE_HZ
        end_s = start_s + duration_s
        events: list[TranscriptEvent] = []
        if has_voice(
            chunk,
            input_format=self.input_format,
            voice_threshold=self.voice_threshold,
        ):
            self.last_voice_offset_s = end_s
            self.pending_end_ts = None
            if not self.in_utterance:
                self.in_utterance = True
                self.emitted_start = True
                self._current_start_offset_s = start_s
                events.append(
                    UtteranceStart(
                        utterance_id=self.utterance_id,
                        ts_offset_s=start_s,
                    )
                )
        elif (
            self.in_utterance
            and self.last_voice_offset_s is not None
            and end_s - self.last_voice_offset_s >= self.silence_threshold_s
        ):
            self.pending_end_ts = self.last_voice_offset_s + self.silence_threshold_s
        self.input_offset_s = end_s
        return events

    def observe_provisional(
        self,
        observation: ProvisionalObservation,
        *,
        utterance_id: int | None = None,
    ) -> list[TranscriptEvent]:
        state, is_current = self._event_state(utterance_id)
        if not observation.text:
            return []
        if observation.text == state.last_partial_text:
            if state.emitted_partial and observation.text != state.last_final_text:
                state.latest_provisional_finalized = False
                self._store_event_state(state, is_current=is_current)
            return []
        source_offset_s = (
            observation.source_offset_s
            if observation.source_offset_s is not None
            else (self.input_offset_s if is_current else self._closed_end_offset(state))
        )
        events = self._ensure_state_started(state, source_offset_s, is_current=is_current)
        state.last_partial_text = observation.text
        state.last_partial_ts_offset_s = source_offset_s
        state.last_partial_speaker = observation.speaker
        state.emitted_partial = True
        state.latest_provisional_finalized = False
        events.append(
            Partial(
                utterance_id=state.utterance_id,
                ts_offset_s=source_offset_s,
                text=observation.text,
                speaker=observation.speaker,
            )
        )
        self._store_event_state(state, is_current=is_current)
        return events

    def observe_final(
        self,
        observation: FinalObservation,
        *,
        utterance_id: int | None = None,
    ) -> list[TranscriptEvent]:
        state, is_current = self._event_state(utterance_id)
        if not observation.text or observation in state.seen_final_observations:
            return []
        state.seen_final_observations.add(observation)
        events = self._ensure_state_started(
            state,
            observation.source_offset_s,
            is_current=is_current,
        )
        if not state.emitted_partial:
            events.append(
                Partial(
                    utterance_id=state.utterance_id,
                    ts_offset_s=observation.source_offset_s,
                    text=observation.text,
                    speaker=observation.speaker,
                )
            )
            state.last_partial_text = observation.text
            state.last_partial_ts_offset_s = observation.source_offset_s
            state.last_partial_speaker = observation.speaker
            state.emitted_partial = True
        events.append(
            Final(
                utterance_id=state.utterance_id,
                ts_offset_s=observation.source_offset_s,
                text=observation.text,
                speaker=observation.speaker,
            )
        )
        state.emitted_final = True
        state.latest_provisional_finalized = True
        state.last_final_text = observation.text
        self._store_event_state(state, is_current=is_current)
        return events

    def finish_input(self) -> None:
        self.input_finished = True
        if self.in_utterance and self.pending_end_ts is None:
            fallback_end = self.input_offset_s
            if self.last_voice_offset_s is not None:
                fallback_end = max(
                    fallback_end,
                    self.last_voice_offset_s + self.silence_threshold_s,
                )
            if self.last_partial_ts_offset_s is not None:
                fallback_end = max(fallback_end, self.last_partial_ts_offset_s)
            self.pending_end_ts = fallback_end

    @property
    def has_unfinalized_provisional(self) -> bool:
        return self.emitted_partial and not self._latest_provisional_finalized

    def finalize_events(
        self,
        *,
        utterance_id: int | None = None,
    ) -> list[TranscriptEvent]:
        state, is_current = self._event_state(utterance_id)
        if is_current and not self.input_finished:
            return []
        if not state.in_utterance or state.end_offset_s is None:
            return []

        events: list[TranscriptEvent] = []
        if self._state_has_unfinalized_provisional(state):
            events.append(
                Final(
                    utterance_id=state.utterance_id,
                    ts_offset_s=(
                        state.last_partial_ts_offset_s
                        if state.last_partial_ts_offset_s is not None
                        else state.end_offset_s
                    ),
                    text=state.last_partial_text,
                    speaker=state.last_partial_speaker,
                )
            )
            state.emitted_final = True
            state.latest_provisional_finalized = True
        end_event = self._maybe_end_state(state)
        if end_event is not None:
            events.append(end_event)
            self._expire_event_state(state, is_current=is_current)
        else:
            self._store_event_state(state, is_current=is_current)
        return events

    def maybe_end_event(
        self,
        *,
        utterance_id: int | None = None,
    ) -> UtteranceEnd | None:
        state, is_current = self._event_state(utterance_id)
        event = self._maybe_end_state(state)
        if event is None:
            self._store_event_state(state, is_current=is_current)
            return None
        self._expire_event_state(state, is_current=is_current)
        return event

    def _maybe_end_state(self, state: _UtteranceEventState) -> UtteranceEnd | None:
        if not (
            state.in_utterance
            and state.end_offset_s is not None
            and (state.emitted_final or not state.emitted_partial)
            and not self._state_has_unfinalized_provisional(state)
        ):
            return None
        return UtteranceEnd(
            utterance_id=state.utterance_id,
            ts_offset_s=state.end_offset_s,
        )

    def _close_current_utterance(self, pcm: bytes) -> ClosedUtterance:
        if not self.in_utterance or self.pending_end_ts is None:
            raise RuntimeError("current utterance has no confirmed end boundary")
        if self._current_start_offset_s is None:
            raise RuntimeError("current utterance has no start offset")
        state = self._current_event_state()
        self._closed_utterances[state.utterance_id] = state
        closed = ClosedUtterance(
            utterance_id=state.utterance_id,
            pcm=pcm,
            start_offset_s=self._current_start_offset_s,
            end_offset_s=self.pending_end_ts,
        )
        self._reset_current_event_state(advance_id=True)
        return closed

    def _finalize_closed_events_for_pipeline(
        self,
        utterance_id: int,
    ) -> list[TranscriptEvent]:
        try:
            state = self._closed_utterances[utterance_id]
        except KeyError:
            raise ValueError(f"unknown or expired utterance_id {utterance_id}") from None
        events: list[TranscriptEvent] = []
        if self._state_has_unfinalized_provisional(state):
            events.append(
                Final(
                    utterance_id=state.utterance_id,
                    ts_offset_s=(
                        state.last_partial_ts_offset_s
                        if state.last_partial_ts_offset_s is not None
                        else self._closed_end_offset(state)
                    ),
                    text=state.last_partial_text,
                    speaker=state.last_partial_speaker,
                )
            )
            state.emitted_final = True
            state.latest_provisional_finalized = True
        end_event = self._maybe_end_state(state)
        if end_event is not None:
            events.append(end_event)
        return events

    def _release_closed_utterance(self, utterance_id: int) -> None:
        try:
            del self._closed_utterances[utterance_id]
        except KeyError:
            raise ValueError(f"unknown or expired utterance_id {utterance_id}") from None

    def _discard_closed_utterances(self) -> None:
        self._closed_utterances.clear()

    def _event_state(
        self,
        utterance_id: int | None,
    ) -> tuple[_UtteranceEventState, bool]:
        if utterance_id is None or utterance_id == self.utterance_id:
            return self._current_event_state(), True
        try:
            return self._closed_utterances[utterance_id], False
        except KeyError:
            raise ValueError(f"unknown or expired utterance_id {utterance_id}") from None

    def _current_event_state(self) -> _UtteranceEventState:
        return _UtteranceEventState(
            utterance_id=self.utterance_id,
            in_utterance=self.in_utterance,
            emitted_start=self.emitted_start,
            emitted_partial=self.emitted_partial,
            emitted_final=self.emitted_final,
            last_partial_text=self.last_partial_text,
            last_partial_ts_offset_s=self.last_partial_ts_offset_s,
            last_partial_speaker=self.last_partial_speaker,
            latest_provisional_finalized=self._latest_provisional_finalized,
            last_final_text=self._last_final_text,
            seen_final_observations=set(self._seen_final_observations),
            start_offset_s=self._current_start_offset_s,
            end_offset_s=self.pending_end_ts,
        )

    def _store_event_state(
        self,
        state: _UtteranceEventState,
        *,
        is_current: bool,
    ) -> None:
        if not is_current:
            return
        self.in_utterance = state.in_utterance
        self.emitted_start = state.emitted_start
        self.emitted_partial = state.emitted_partial
        self.emitted_final = state.emitted_final
        self.last_partial_text = state.last_partial_text
        self.last_partial_ts_offset_s = state.last_partial_ts_offset_s
        self.last_partial_speaker = state.last_partial_speaker
        self._latest_provisional_finalized = state.latest_provisional_finalized
        self._last_final_text = state.last_final_text
        self._seen_final_observations = state.seen_final_observations
        self._current_start_offset_s = state.start_offset_s
        self.pending_end_ts = state.end_offset_s

    def _expire_event_state(
        self,
        state: _UtteranceEventState,
        *,
        is_current: bool,
    ) -> None:
        if is_current:
            self._reset_current_event_state(advance_id=True)
        else:
            del self._closed_utterances[state.utterance_id]

    def _reset_current_event_state(self, *, advance_id: bool) -> None:
        if advance_id:
            self.utterance_id += 1
        self.pending_end_ts = None
        self.in_utterance = False
        self.emitted_start = False
        self.emitted_partial = False
        self.emitted_final = False
        self._latest_provisional_finalized = False
        self._last_final_text = ""
        self._seen_final_observations = set()
        self.last_partial_text = ""
        self.last_partial_ts_offset_s = None
        self.last_partial_speaker = None
        self._current_start_offset_s = None

    def _ensure_state_started(
        self,
        state: _UtteranceEventState,
        source_offset_s: float,
        *,
        is_current: bool,
    ) -> list[TranscriptEvent]:
        state.in_utterance = True
        if is_current and self.input_finished and state.end_offset_s is None:
            state.end_offset_s = max(self.input_offset_s, source_offset_s)
        if state.start_offset_s is None:
            state.start_offset_s = source_offset_s
        if state.emitted_start:
            return []
        state.emitted_start = True
        return [
            UtteranceStart(
                utterance_id=state.utterance_id,
                ts_offset_s=source_offset_s,
            )
        ]

    @staticmethod
    def _state_has_unfinalized_provisional(state: _UtteranceEventState) -> bool:
        return state.emitted_partial and not state.latest_provisional_finalized

    @staticmethod
    def _closed_end_offset(state: _UtteranceEventState) -> float:
        if state.end_offset_s is None:
            raise RuntimeError("closed utterance has no end offset")
        return state.end_offset_s


class UtteranceDecodePipeline:
    def __init__(
        self,
        *,
        boundary_settings: UtteranceBoundarySettings = UtteranceBoundarySettings(),
        decode_utterance: Callable[
            [ClosedUtterance],
            Awaitable[Iterable[ProvisionalObservation | FinalObservation]],
        ],
        input_format: EngineInputFormat = EngineInputFormat.PCM_S16LE,
        on_progress: Callable[[], None] | None = None,
    ) -> None:
        validate_input_format(
            input_format,
            target="UtteranceDecodePipeline",
            accepted=_RAW_PCM_INPUT_FORMATS,
        )
        if not isinstance(boundary_settings, UtteranceBoundarySettings):
            raise TypeError("boundary_settings must be UtteranceBoundarySettings")
        self._boundary_settings = boundary_settings
        self._decode_utterance = decode_utterance
        self._input_format = input_format
        self._on_progress = on_progress
        self._decoding_utterance_id: int | None = None
        self._pending_decode_utterances = 0
        self._processed_pcm_chunks = 0
        self._emitted_events = 0
        self._last_progress_monotonic_s: float | None = None
        self._active_stream = False

    def stream(
        self,
        pcm_chunks: AsyncIterator[bytes],
        *,
        segmentation: Segmentation = Segmentation.AUTO,
    ) -> AsyncIterator[TranscriptEvent]:
        """Decode automatic boundaries, or the whole input once with SINGLE.

        SINGLE retains all PCM until end of input, bypassing voice detection and
        min_voiced_s. Non-empty input emits one start/end pair even without text;
        empty input emits nothing and is not decoded. Offsets start at zero.
        """
        if not isinstance(segmentation, Segmentation):
            raise ValueError("segmentation must be Segmentation.AUTO or Segmentation.SINGLE")
        return self._stream(pcm_chunks, segmentation=segmentation)

    async def _stream(
        self, pcm_chunks: AsyncIterator[bytes], *, segmentation: Segmentation
    ) -> AsyncIterator[TranscriptEvent]:
        if self._active_stream:
            raise RuntimeError("UtteranceDecodePipeline already has an active stream")
        self._active_stream = True

        decode_done = object()
        decode_stop = object()
        output_wake = object()
        decode_queue: asyncio.Queue[ClosedUtterance | object] = asyncio.Queue()
        output_queue: asyncio.Queue[TranscriptEvent | object] = asyncio.Queue()
        event_buffers: dict[int, deque[TranscriptEvent]] = {}
        session = UtteranceBoundarySession(
            silence_threshold_s=self._boundary_settings.silence_threshold_s,
            voice_threshold=self._boundary_settings.voice_threshold,
            input_format=self._input_format,
        )
        next_emit_utterance_id = 1
        terminal_cause: BaseException | None = None
        stop_requested = False
        normal_complete = False
        producer_task: asyncio.Task[None] | None = None

        def wake_output() -> None:
            output_queue.put_nowait(output_wake)

        def record_terminal_cause(error: BaseException) -> None:
            nonlocal terminal_cause, stop_requested
            if terminal_cause is None:
                terminal_cause = error
            stop_requested = True
            wake_output()

        def queue_events(events: Iterable[TranscriptEvent]) -> None:
            nonlocal next_emit_utterance_id
            for event in events:
                event_buffers.setdefault(event.utterance_id, deque()).append(event)

            while True:
                buffered = event_buffers.get(next_emit_utterance_id)
                if not buffered:
                    return
                reached_end = False
                while buffered:
                    event = buffered.popleft()
                    output_queue.put_nowait(event)
                    if isinstance(event, UtteranceEnd):
                        reached_end = True
                        break
                if not buffered:
                    del event_buffers[next_emit_utterance_id]
                if not reached_end:
                    return
                next_emit_utterance_id += 1

        def close_current_utterance(pcm: bytes, *, voice_start_s: float) -> None:
            if segmentation is Segmentation.AUTO:
                if session.last_voice_offset_s is None:
                    raise RuntimeError("active utterance has no voiced audio")
                # Recover whole samples so rounding cannot shorten an exact threshold.
                voiced_samples = round(
                    (session.last_voice_offset_s - voice_start_s) * SAMPLE_RATE_HZ
                )
                if voiced_samples / SAMPLE_RATE_HZ < self._boundary_settings.min_voiced_s:
                    utterance = session._close_current_utterance(pcm)
                    queue_events(
                        session._finalize_closed_events_for_pipeline(utterance.utterance_id)
                    )
                    return
            utterance = session._close_current_utterance(pcm)
            self._pending_decode_utterances += 1
            decode_queue.put_nowait(utterance)

        async def receive_pcm() -> None:
            nonlocal stop_requested
            current_pcm: bytearray | None = None
            voice_start_s = 0.0
            preroll_pcm = bytearray()
            sample_width = _SAMPLE_WIDTH_BYTES[self._input_format]
            preroll_bytes = round(self._boundary_settings.preroll_s * SAMPLE_RATE_HZ) * sample_width
            try:
                async for chunk in pcm_chunks:
                    if stop_requested:
                        break
                    if segmentation is Segmentation.SINGLE:
                        validate_pcm_chunk(chunk, input_format=self._input_format)
                        if chunk:
                            if current_pcm is None:
                                current_pcm = bytearray()
                                session.in_utterance = True
                                session.emitted_start = True
                                session._current_start_offset_s = 0.0
                                queue_events([UtteranceStart(session.utterance_id, 0.0)])
                            current_pcm.extend(chunk)
                            session.input_offset_s = (
                                len(current_pcm) / sample_width / SAMPLE_RATE_HZ
                            )
                        self._record_progress()
                        self._processed_pcm_chunks += 1
                        continue
                    was_in_utterance = session.in_utterance
                    chunk_start_s = session.input_offset_s
                    events = session.observe_pcm(chunk)
                    self._record_progress()
                    self._processed_pcm_chunks += 1
                    if not was_in_utterance and session.in_utterance:
                        voice_start_s = chunk_start_s
                        start_offset_s = max(
                            0.0,
                            chunk_start_s - len(preroll_pcm) / sample_width / SAMPLE_RATE_HZ,
                        )
                        session._current_start_offset_s = start_offset_s
                        events[0] = UtteranceStart(session.utterance_id, start_offset_s)
                        current_pcm = preroll_pcm
                        preroll_pcm = bytearray()
                    if session.in_utterance:
                        if current_pcm is None:
                            current_pcm = bytearray()
                        current_pcm.extend(chunk)
                    elif preroll_bytes != 0:
                        preroll_pcm.extend(chunk)
                        del preroll_pcm[:-preroll_bytes]
                    queue_events(events)
                    if session.in_utterance and session.pending_end_ts is not None:
                        if current_pcm is None:
                            raise RuntimeError("active utterance has no PCM buffer")
                        close_current_utterance(bytes(current_pcm), voice_start_s=voice_start_s)
                        current_pcm = None

                if stop_requested:
                    return
                session.finish_input()
                if session.in_utterance:
                    if current_pcm is None:
                        raise RuntimeError("active utterance has no PCM buffer")
                    session.pending_end_ts = session.input_offset_s
                    close_current_utterance(bytes(current_pcm), voice_start_s=voice_start_s)
                decode_queue.put_nowait(decode_done)
            except BaseException as error:
                if not stop_requested:
                    record_terminal_cause(error)
                decode_queue.put_nowait(decode_stop)
            finally:
                wake_output()

        async def decode_closed() -> None:
            nonlocal stop_requested
            try:
                while True:
                    item = await decode_queue.get()
                    if item is decode_done or item is decode_stop or stop_requested:
                        break
                    if not isinstance(item, ClosedUtterance):
                        raise RuntimeError("decode queue contained an invalid item")
                    self._pending_decode_utterances -= 1
                    self._decoding_utterance_id = item.utterance_id
                    try:
                        observations = await self._decode_utterance(item)
                        self._record_progress()
                        if stop_requested:
                            break
                        for observation in observations:
                            if isinstance(observation, ProvisionalObservation):
                                events = session.observe_provisional(
                                    observation,
                                    utterance_id=item.utterance_id,
                                )
                            elif isinstance(observation, FinalObservation):
                                events = session.observe_final(
                                    observation,
                                    utterance_id=item.utterance_id,
                                )
                            else:
                                raise TypeError(
                                    "decode_utterance must return provisional or final observations"
                                )
                            queue_events(events)
                        queue_events(
                            session._finalize_closed_events_for_pipeline(item.utterance_id)
                        )
                    except BaseException as error:
                        if not stop_requested:
                            record_terminal_cause(error)
                            if producer_task is not None and not producer_task.done():
                                producer_task.cancel()
                        break
                    finally:
                        self._decoding_utterance_id = None
            except BaseException as error:
                if not stop_requested:
                    record_terminal_cause(error)
                    if producer_task is not None and not producer_task.done():
                        producer_task.cancel()
            finally:
                wake_output()

        producer_task = asyncio.create_task(
            receive_pcm(),
            name="listenflow-utterance-pipeline-receiver",
        )
        decode_task = asyncio.create_task(
            decode_closed(),
            name="listenflow-utterance-pipeline-decoder",
        )

        try:
            while True:
                if terminal_cause is not None:
                    break
                if producer_task.done() and decode_task.done() and output_queue.empty():
                    normal_complete = True
                    break
                item = await output_queue.get()
                if terminal_cause is not None or item is output_wake:
                    continue
                if not isinstance(item, (UtteranceStart, Partial, Final, UtteranceEnd)):
                    raise RuntimeError("output queue contained an invalid item")
                self._emitted_events += 1
                if isinstance(item, UtteranceEnd):
                    session._release_closed_utterance(item.utterance_id)
                yield item
        finally:
            if not normal_complete:
                stop_requested = True
                cancel_once(producer_task)
                cancel_once(decode_task)
                decode_queue.put_nowait(decode_stop)
            await finish(asyncio.gather(producer_task, decode_task, return_exceptions=True))
            self._decoding_utterance_id = None
            self._pending_decode_utterances = 0
            event_buffers.clear()
            session._discard_closed_utterances()
            self._active_stream = False

        if terminal_cause is not None:
            raise terminal_cause

    def _record_progress(self) -> None:
        if self._on_progress is None:
            self._last_progress_monotonic_s = time.monotonic()
        else:
            self._on_progress()

    @property
    def decoding_utterance_id(self) -> int | None:
        return self._decoding_utterance_id

    @property
    def pending_decode_utterances(self) -> int:
        return self._pending_decode_utterances

    @property
    def processed_pcm_chunks(self) -> int:
        return self._processed_pcm_chunks

    @property
    def emitted_events(self) -> int:
        return self._emitted_events

    @property
    def last_progress_monotonic_s(self) -> float | None:
        return self._last_progress_monotonic_s
