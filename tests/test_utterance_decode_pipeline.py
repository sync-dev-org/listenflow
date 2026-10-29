from __future__ import annotations

import asyncio
import inspect
import struct
from collections.abc import AsyncIterator, Iterable
from dataclasses import FrozenInstanceError

import pytest

import listenflow
from listenflow import (
    ClosedUtterance,
    EngineInputFormat,
    Final,
    FinalObservation,
    HealthSnapshot,
    Partial,
    ProvisionalObservation,
    STTEngineBase,
    UtteranceBoundarySession,
    UtteranceDecodePipeline,
    UtteranceEnd,
    UtteranceStart,
)


def _pcm_chunk(sample: int, *, duration_s: float) -> bytes:
    sample_count = round(16_000 * duration_s)
    return sample.to_bytes(2, byteorder="little", signed=True) * sample_count


_END_OF_INPUT = object()


class _ControlledPCM(AsyncIterator[bytes]):
    def __init__(self, *, cancellation_error: BaseException | None = None) -> None:
        self._items: asyncio.Queue[tuple[bytes | BaseException | object, asyncio.Event]] = (
            asyncio.Queue()
        )
        self._cancellation_error = cancellation_error
        self.cancelled = asyncio.Event()

    def __aiter__(self) -> _ControlledPCM:
        return self

    async def __anext__(self) -> bytes:
        try:
            item, received = await self._items.get()
        except asyncio.CancelledError:
            self.cancelled.set()
            if self._cancellation_error is not None:
                raise self._cancellation_error
            raise
        received.set()
        if item is _END_OF_INPUT:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, bytes)
        return item

    async def send(self, chunk: bytes) -> None:
        await self._send(chunk)

    async def fail(self, error: BaseException) -> None:
        await self._send(error)

    async def finish(self) -> None:
        await self._send(_END_OF_INPUT)

    async def _send(self, item: bytes | BaseException | object) -> None:
        received = asyncio.Event()
        await self._items.put((item, received))
        await received.wait()


async def _collect(stream: AsyncIterator[object]) -> list[object]:
    return [item async for item in stream]


def _pipeline_tasks() -> list[asyncio.Task[object]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name().startswith("listenflow-utterance-pipeline-")
        and not task.done()
    ]


async def _empty_decode(
    utterance: ClosedUtterance,
) -> Iterable[ProvisionalObservation | FinalObservation]:
    del utterance
    return ()


@pytest.mark.req("SP-LF-035")
@pytest.mark.req("SP-LF-005")
def test_boundary_settings_are_public_immutable_and_have_shared_defaults() -> None:
    """利用側は発話境界設定を一つの変更できない値として省略して構築できる。"""
    engine_module = __import__("listenflow.engine", fromlist=["engine"])
    settings_type = listenflow.UtteranceBoundarySettings

    assert settings_type is engine_module.UtteranceBoundarySettings
    assert "UtteranceBoundarySettings" in listenflow.__all__
    settings = settings_type()
    assert settings.silence_threshold_s == 0.8
    assert settings.voice_threshold == engine_module.DEFAULT_VOICE_THRESHOLD
    assert settings.preroll_s == 0.3
    with pytest.raises(FrozenInstanceError):
        settings.preroll_s = 0.5


@pytest.mark.req("SP-LF-035")
@pytest.mark.req("SP-LF-005")
@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("silence_threshold_s", 0),
        ("silence_threshold_s", -0.1),
        ("silence_threshold_s", float("inf")),
        ("silence_threshold_s", float("nan")),
        ("silence_threshold_s", True),
        ("silence_threshold_s", "0.1"),
        ("voice_threshold", -0.1),
        ("voice_threshold", 1.1),
        ("voice_threshold", float("inf")),
        ("voice_threshold", float("nan")),
        ("voice_threshold", True),
        ("voice_threshold", 1),
        ("voice_threshold", "0.1"),
        ("preroll_s", -0.1),
        ("preroll_s", float("inf")),
        ("preroll_s", float("nan")),
        ("preroll_s", False),
        ("preroll_s", "0.1"),
    ],
)
def test_boundary_settings_reject_invalid_values_at_construction(
    field_name: str, invalid_value: object
) -> None:
    """発話境界設定は範囲外、有限でない値、bool と非数値を構築時に拒否する。"""
    with pytest.raises(ValueError, match=field_name):
        listenflow.UtteranceBoundarySettings(**{field_name: invalid_value})


@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-031")
def test_boundary_settings_reject_silence_duration_too_large_for_float() -> None:
    """無音の長さが時刻へ加算できない大きさなら、stream を始める前に拒否する。"""
    with pytest.raises(ValueError, match="silence_threshold_s"):
        listenflow.UtteranceBoundarySettings(silence_threshold_s=10**400)


@pytest.mark.req("SP-LF-035")
def test_boundary_settings_reject_preroll_duration_that_overflows_sample_count() -> None:
    """発話の始まりの前に取る音の長さを 16 kHz の標本数へ換算できなければ構築時に拒否する。"""
    with pytest.raises(ValueError, match="preroll_s"):
        listenflow.UtteranceBoundarySettings(preroll_s=1e308)


@pytest.mark.req("SP-LF-095")
def test_boundary_settings_default_min_voiced_to_zero() -> None:
    """声の部分の最短の秒数は、省略すると 0.0 になる。"""
    assert listenflow.UtteranceBoundarySettings().min_voiced_s == 0.0


@pytest.mark.req("SP-LF-095")
@pytest.mark.parametrize("seconds", [0, 0.0, 0.2, 1, 1e308])
def test_boundary_settings_accept_finite_nonnegative_min_voiced(seconds: int | float) -> None:
    """声の部分の最短の秒数は 0 以上の有限な数を受け取る。"""
    settings = listenflow.UtteranceBoundarySettings(min_voiced_s=seconds)
    assert settings.min_voiced_s == float(seconds)
    assert type(settings.min_voiced_s) is float


@pytest.mark.req("SP-LF-095")
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0.2",
        None,
        0.2j,
        pytest.param(10**400, id="integer-overflows-float"),
    ],
)
def test_boundary_settings_reject_invalid_min_voiced(invalid_value: object) -> None:
    """声の部分の最短の秒数は負の値、有限でない値、非数値を ValueError で拒否する。"""
    with pytest.raises(ValueError, match="min_voiced_s"):
        listenflow.UtteranceBoundarySettings(min_voiced_s=invalid_value)


@pytest.mark.req("SP-LF-005")
def test_pipeline_rejects_old_separate_boundary_arguments() -> None:
    """一括 decode pipeline は発話境界設定を一つの値として受け取る。"""
    with pytest.raises(TypeError):
        UtteranceDecodePipeline(
            silence_threshold_s=0.1,
            voice_threshold=0.2,
            decode_utterance=_empty_decode,
        )


@pytest.mark.req("SP-LF-031")
@pytest.mark.req("SP-LF-001")
@pytest.mark.req("SP-LF-014")
def test_ac_24_1_public_pipeline_surface_is_typed_and_read_only() -> None:
    """公開 API は変更できない閉じた発話の型、操作の口、pipeline の待機中の値を持つ。"""
    engine_module = __import__("listenflow.engine", fromlist=["engine"])

    assert listenflow.ClosedUtterance is engine_module.ClosedUtterance
    assert listenflow.UtteranceDecodePipeline is engine_module.UtteranceDecodePipeline
    assert {"ClosedUtterance", "UtteranceDecodePipeline"} <= set(listenflow.__all__)

    utterance = ClosedUtterance(
        utterance_id=7,
        pcm=b"pcm",
        start_offset_s=1.25,
        end_offset_s=2.5,
    )
    assert utterance == ClosedUtterance(7, b"pcm", 1.25, 2.5)
    with pytest.raises(FrozenInstanceError):
        utterance.pcm = b"replacement"

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.1,
            voice_threshold=0.2,
        ),
        decode_utterance=_empty_decode,
    )
    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0
    assert pipeline.processed_pcm_chunks == 0
    assert pipeline.emitted_events == 0
    assert pipeline.last_progress_monotonic_s is None
    for property_name in (
        "decoding_utterance_id",
        "pending_decode_utterances",
        "processed_pcm_chunks",
        "emitted_events",
        "last_progress_monotonic_s",
    ):
        with pytest.raises(AttributeError):
            setattr(pipeline, property_name, 99)

    assert str(inspect.signature(STTEngineBase.prepare)) == "(self) -> 'None'"
    assert str(inspect.signature(STTEngineBase.stream)) == (
        "(self, pcm_chunks: 'AsyncIterator[bytes]') -> 'AsyncIterator[TranscriptEvent]'"
    )
    assert str(inspect.signature(STTEngineBase.health)) == "(self) -> 'HealthSnapshot'"
    expected_boundary_signatures = {
        "observe_provisional": (
            "(self, observation: 'ProvisionalObservation', *, "
            "utterance_id: 'int | None' = None) -> 'list[TranscriptEvent]'"
        ),
        "observe_final": (
            "(self, observation: 'FinalObservation', *, "
            "utterance_id: 'int | None' = None) -> 'list[TranscriptEvent]'"
        ),
        "finalize_events": (
            "(self, *, utterance_id: 'int | None' = None) -> 'list[TranscriptEvent]'"
        ),
        "maybe_end_event": (
            "(self, *, utterance_id: 'int | None' = None) -> 'UtteranceEnd | None'"
        ),
    }
    assert {
        method_name: str(inspect.signature(getattr(UtteranceBoundarySession, method_name)))
        for method_name in expected_boundary_signatures
    } == expected_boundary_signatures


@pytest.mark.req("SP-LF-014")
def test_health_snapshot_reports_shared_state_and_recognition_idle_defaults() -> None:
    """健全性の観測は処理量と稼働中の状態を示す。"""
    snapshot = HealthSnapshot(
        state=listenflow.EngineState.READY,
        busy=False,
        processed_pcm_chunks=12,
        emitted_events=34,
        last_progress_monotonic_s=56.0,
    )

    assert snapshot.state == listenflow.EngineState.READY
    assert not snapshot.busy
    assert snapshot.processed_pcm_chunks == 12
    assert snapshot.emitted_events == 34
    assert snapshot.last_progress_monotonic_s == 56.0
    assert snapshot.bias_application is None
    assert snapshot.decoding_utterance_id is None
    assert snapshot.pending_decode_utterances == 0


@pytest.mark.req("SP-LF-031")
@pytest.mark.req("SP-LF-005")
def test_ac_24_13_15_closed_state_targets_only_its_id_until_end() -> None:
    """発話の通知は ID ごとに相関し、終了後の発話には通知を追加しない。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.05,
        voice_threshold=500 / 32768,
    )
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.1)

    assert session.observe_pcm(voice) == [UtteranceStart(utterance_id=1, ts_offset_s=0.0)]
    session.observe_pcm(silence)
    closed = session._close_current_utterance(voice + silence)

    assert closed == ClosedUtterance(
        utterance_id=1,
        pcm=voice + silence,
        start_offset_s=0.0,
        end_offset_s=pytest.approx(0.06),
    )
    assert session.utterance_id == 2
    assert not session.in_utterance

    closed_events = session.observe_provisional(
        ProvisionalObservation(text="closed", speaker="a"),
        utterance_id=1,
    )
    current_events = session.observe_provisional(
        ProvisionalObservation(text="current", speaker="b", source_offset_s=0.2)
    )

    assert closed_events == [
        Partial(
            utterance_id=1,
            ts_offset_s=pytest.approx(0.06),
            text="closed",
            speaker="a",
        )
    ]
    assert current_events == [
        UtteranceStart(utterance_id=2, ts_offset_s=0.2),
        Partial(utterance_id=2, ts_offset_s=0.2, text="current", speaker="b"),
    ]
    assert session.finalize_events(utterance_id=1) == [
        Final(
            utterance_id=1,
            ts_offset_s=pytest.approx(0.06),
            text="closed",
            speaker="a",
        ),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.06)),
    ]
    assert session.utterance_id == 2
    assert session.in_utterance
    assert session.last_partial_text == "current"

    for operation in (
        lambda: session.observe_provisional(ProvisionalObservation(text="expired"), utterance_id=1),
        lambda: session.observe_final(
            FinalObservation(text="expired", source_offset_s=0.3), utterance_id=1
        ),
        lambda: session.finalize_events(utterance_id=1),
        lambda: session.maybe_end_event(utterance_id=1),
    ):
        with pytest.raises(ValueError, match=r"unknown or expired utterance_id 1"):
            operation()


@pytest.mark.req("SP-LF-031")
@pytest.mark.req("SP-LF-005")
def test_ac_24_13_maybe_end_event_targets_only_the_live_closed_id() -> None:
    """発話の終了は対象の ID だけに通知する。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.05,
        voice_threshold=500 / 32768,
    )
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.1)

    session.observe_pcm(voice)
    session.observe_pcm(silence)
    first = session._close_current_utterance(voice + silence)
    assert session.observe_pcm(voice) == [
        UtteranceStart(utterance_id=2, ts_offset_s=pytest.approx(0.11))
    ]
    session.observe_pcm(silence)
    second = session._close_current_utterance(voice + silence)

    assert (first.utterance_id, second.utterance_id) == (1, 2)
    assert second.end_offset_s == pytest.approx(0.17)
    assert session.utterance_id == 3
    assert not session.in_utterance
    session.observe_provisional(
        ProvisionalObservation(text="pending", speaker="a"),
        utterance_id=1,
    )

    assert session.maybe_end_event(utterance_id=1) is None
    assert session.maybe_end_event(utterance_id=2) == UtteranceEnd(
        utterance_id=2,
        ts_offset_s=pytest.approx(0.17),
    )
    assert session.maybe_end_event() is None
    assert session.utterance_id == 3
    assert not session.in_utterance
    assert session.last_partial_text == ""
    with pytest.raises(ValueError, match=r"unknown or expired utterance_id 2"):
        session.maybe_end_event(utterance_id=2)
    assert session.finalize_events(utterance_id=1) == [
        Final(
            utterance_id=1,
            ts_offset_s=pytest.approx(0.06),
            text="pending",
            speaker="a",
        ),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.06)),
    ]
    with pytest.raises(ValueError, match=r"unknown or expired utterance_id 1"):
        session.maybe_end_event(utterance_id=1)


@pytest.mark.req("SP-LF-031")
@pytest.mark.req("SP-LF-005")
def test_ac_24_13_observation_deduplication_is_per_utterance_id() -> None:
    """同じ確定の観測を再び適用しても通知を増やさない判定は、発話の ID ごとに行う。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.05,
        voice_threshold=500 / 32768,
    )
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.1)
    observation = FinalObservation(text="same", source_offset_s=0.01)

    session.observe_pcm(voice)
    session.observe_pcm(silence)
    session._close_current_utterance(voice + silence)
    session.observe_pcm(voice)
    session.observe_pcm(silence)
    session._close_current_utterance(voice + silence)

    first = session.observe_final(observation, utterance_id=1)
    second = session.observe_final(observation, utterance_id=2)

    assert [event.utterance_id for event in first] == [1, 1]
    assert [event.utterance_id for event in second] == [2, 2]
    assert session.observe_final(observation, utterance_id=1) == []
    assert session.observe_final(observation, utterance_id=2) == []


@pytest.mark.req("SP-LF-031")
@pytest.mark.req("SP-LF-005")
def test_ac_24_16_closed_empty_decode_finalizes_as_start_then_end() -> None:
    """文字起こしで text にならなかった閉じた発話も開始と終了の通知で閉じる。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.05,
        voice_threshold=500 / 32768,
    )
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.1)

    start = session.observe_pcm(voice)
    session.observe_pcm(silence)
    session._close_current_utterance(voice + silence)

    assert start + session.finalize_events(utterance_id=1) == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.06)),
    ]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-031")
@pytest.mark.req("SP-LF-001")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-006")
@pytest.mark.req("SP-LF-008")
@pytest.mark.req("SP-LF-014")
async def test_ac_24_3_to_8_16_18_pipeline_receives_while_serial_decode_drains_fifo() -> None:
    """非同期の音声を受け続け、元の音声の時刻を保つ text を発話順に返して全発話を閉じる。"""
    source = _ControlledPCM()
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.01)
    started = {utterance_id: asyncio.Event() for utterance_id in range(1, 4)}
    release = {utterance_id: asyncio.Event() for utterance_id in range(1, 4)}
    received: list[ClosedUtterance] = []
    active_callbacks = 0
    maximum_active_callbacks = 0

    async def decode(
        utterance: ClosedUtterance,
    ) -> Iterable[ProvisionalObservation | FinalObservation]:
        nonlocal active_callbacks, maximum_active_callbacks
        received.append(utterance)
        active_callbacks += 1
        maximum_active_callbacks = max(maximum_active_callbacks, active_callbacks)
        started[utterance.utterance_id].set()
        await release[utterance.utterance_id].wait()
        active_callbacks -= 1
        if utterance.utterance_id == 1:
            return (
                ProvisionalObservation(text="alpha", source_offset_s=0.015),
                FinalObservation(text="alpha", source_offset_s=0.016),
                ProvisionalObservation(text="alpha beta", source_offset_s=0.017),
                FinalObservation(text="beta", source_offset_s=0.018),
            )
        if utterance.utterance_id == 2:
            return ()
        return (FinalObservation(text="gamma", source_offset_s=0.05),)

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.01,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    collector = asyncio.create_task(_collect(pipeline.stream(source)))

    await source.send(voice)
    first_progress_time = pipeline.last_progress_monotonic_s
    await source.send(silence)
    await started[1].wait()
    assert pipeline.decoding_utterance_id == 1
    assert pipeline.pending_decode_utterances == 0

    await source.send(voice)
    await source.send(silence)
    await source.send(voice)
    await source.finish()

    assert pipeline.processed_pcm_chunks == 5
    assert first_progress_time is not None
    assert pipeline.last_progress_monotonic_s is not None
    assert pipeline.last_progress_monotonic_s >= first_progress_time
    assert pipeline.decoding_utterance_id == 1
    assert pipeline.pending_decode_utterances == 2
    assert [utterance.utterance_id for utterance in received] == [1]

    release[1].set()
    await started[2].wait()
    assert pipeline.decoding_utterance_id == 2
    assert pipeline.pending_decode_utterances == 1
    release[2].set()
    await started[3].wait()
    assert pipeline.decoding_utterance_id == 3
    assert pipeline.pending_decode_utterances == 0
    release[3].set()

    events = await collector

    assert maximum_active_callbacks == 1
    assert [utterance.utterance_id for utterance in received] == [1, 2, 3]
    assert [utterance.pcm for utterance in received] == [
        voice + silence,
        voice + silence,
        voice,
    ]
    assert [utterance.start_offset_s for utterance in received] == pytest.approx([0.0, 0.02, 0.04])
    assert [utterance.end_offset_s for utterance in received] == pytest.approx([0.02, 0.04, 0.05])
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        Partial(utterance_id=1, ts_offset_s=0.015, text="alpha"),
        Final(utterance_id=1, ts_offset_s=0.016, text="alpha"),
        Partial(utterance_id=1, ts_offset_s=0.017, text="alpha beta"),
        Final(utterance_id=1, ts_offset_s=0.018, text="beta"),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.02)),
        UtteranceStart(utterance_id=2, ts_offset_s=pytest.approx(0.02)),
        UtteranceEnd(utterance_id=2, ts_offset_s=pytest.approx(0.04)),
        UtteranceStart(utterance_id=3, ts_offset_s=pytest.approx(0.04)),
        Partial(utterance_id=3, ts_offset_s=0.05, text="gamma"),
        Final(utterance_id=3, ts_offset_s=0.05, text="gamma"),
        UtteranceEnd(utterance_id=3, ts_offset_s=pytest.approx(0.05)),
    ]
    assert pipeline.emitted_events == len(events)
    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-094")
@pytest.mark.parametrize("input_format", [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32])
@pytest.mark.parametrize("ending", ["silence", "input"])
@pytest.mark.parametrize("voice_samples", [3_199, 3_200, 3_201])
async def test_pipeline_filters_by_voiced_span_without_preroll_or_trailing_silence(
    input_format: EngineInputFormat, ending: str, voice_samples: int
) -> None:
    """前後の無音を除いた声の長さだけで判定し、閾値以上の発話だけを文字起こしに渡す。"""
    pack_format = "<h" if input_format is EngineInputFormat.PCM_S16LE else "<f"
    width = struct.calcsize(pack_format)
    prefix = struct.pack(pack_format, 0) * 12_800
    voice = struct.pack(pack_format, 600 if width == 2 else 0.05) * voice_samples
    trailing_s = 0.6 if ending == "silence" else 0.05
    trailing = struct.pack(pack_format, 0) * round(16_000 * trailing_s)
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return (FinalObservation(text="recognized", source_offset_s=0.8),)

    async def source() -> AsyncIterator[bytes]:
        for chunk in (prefix, voice, trailing):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.4, preroll_s=0.3, min_voiced_s=0.2
        ),
        decode_utterance=decode,
        input_format=input_format,
    )
    events = await _collect(pipeline.stream(source()))

    expected_end_s = 0.8 + voice_samples / 16_000 + (0.4 if ending == "silence" else 0.05)
    start = UtteranceStart(1, pytest.approx(0.5))
    end = UtteranceEnd(1, pytest.approx(expected_end_s))
    if voice_samples < 3_200:
        assert received == []
        assert events == [start, end]
    else:
        assert received == [
            ClosedUtterance(
                1,
                prefix[-4_800 * width :] + voice + trailing,
                pytest.approx(0.5),
                pytest.approx(expected_end_s),
            )
        ]
        assert events == [
            start,
            Partial(1, 0.8, "recognized"),
            Final(1, 0.8, "recognized"),
            end,
        ]
    assert pipeline.pending_decode_utterances == 0
    assert pipeline.decoding_utterance_id is None
    assert pipeline.emitted_events == len(events)
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-094")
async def test_pipeline_min_voiced_span_includes_whole_chunks_and_internal_silence() -> None:
    """声の部分は声と判定した塊の全長と、その間に挟まる無音を含めて数える。"""
    first_voice = _pcm_chunk(600, duration_s=1 / 16_000) + _pcm_chunk(0, duration_s=0.0499375)
    gap = _pcm_chunk(0, duration_s=0.1)
    last_voice = _pcm_chunk(0, duration_s=0.0499375) + _pcm_chunk(600, duration_s=1 / 16_000)
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return (FinalObservation(text="recognized", source_offset_s=0.2),)

    async def source() -> AsyncIterator[bytes]:
        for chunk in (first_voice, gap, last_voice):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.4, min_voiced_s=0.2
        ),
        decode_utterance=decode,
    )
    events = await _collect(pipeline.stream(source()))

    assert received == [ClosedUtterance(1, first_voice + gap + last_voice, 0.0, pytest.approx(0.2))]
    assert events == [
        UtteranceStart(1, 0.0),
        Partial(1, 0.2, "recognized"),
        Final(1, 0.2, "recognized"),
        UtteranceEnd(1, pytest.approx(0.2)),
    ]


@pytest.mark.req("SP-LF-095")
async def test_pipeline_starts_short_voice_immediately_before_input_finishes() -> None:
    """短い声でも最初の塊で始まりを伝え、入力が終わる時は文字起こしせずに閉じる。"""
    source = _ControlledPCM()
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return (FinalObservation(text="unexpected", source_offset_s=0.0),)

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(min_voiced_s=0.2),
        decode_utterance=decode,
    )
    stream = pipeline.stream(source)
    reading = asyncio.create_task(anext(stream))
    await source.send(_pcm_chunk(600, duration_s=0.05))
    try:
        assert await asyncio.wait_for(reading, timeout=1) == UtteranceStart(1, 0.0)
        assert received == []
        await source.finish()
        assert await _collect(stream) == [UtteranceEnd(1, pytest.approx(0.05))]
        assert received == []
    finally:
        await stream.aclose()
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-094")
async def test_pipeline_skips_short_voice_while_decode_waits_and_keeps_event_order() -> None:
    """先行する文字起こしを待つ間の短い発話を捨て、後続する発話も順番を保って処理する。"""
    source = _ControlledPCM()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    received: list[ClosedUtterance] = []
    long_voice = _pcm_chunk(600, duration_s=0.25)
    short_voice = _pcm_chunk(600, duration_s=0.05)
    silence = _pcm_chunk(0, duration_s=0.5)

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        if utterance.utterance_id == 1:
            first_started.set()
            await release_first.wait()
        return (FinalObservation(str(utterance.utterance_id), utterance.start_offset_s),)

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.4, min_voiced_s=0.2
        ),
        decode_utterance=decode,
    )
    collector = asyncio.create_task(_collect(pipeline.stream(source)))
    try:
        await source.send(long_voice)
        await source.send(silence)
        await first_started.wait()
        await source.send(short_voice)
        await source.send(silence)
        await source.send(long_voice)
        await source.finish()
        assert [utterance.utterance_id for utterance in received] == [1]
    finally:
        release_first.set()
        events = await collector

    assert [utterance.utterance_id for utterance in received] == [1, 3]
    assert [utterance.pcm for utterance in received] == [long_voice + silence, long_voice]
    assert events == [
        UtteranceStart(1, 0.0),
        Partial(1, 0.0, "1"),
        Final(1, 0.0, "1"),
        UtteranceEnd(1, pytest.approx(0.65)),
        UtteranceStart(2, pytest.approx(0.75)),
        UtteranceEnd(2, pytest.approx(1.2)),
        UtteranceStart(3, pytest.approx(1.3)),
        Partial(3, pytest.approx(1.3), "3"),
        Final(3, pytest.approx(1.3), "3"),
        UtteranceEnd(3, pytest.approx(1.55)),
    ]
    assert pipeline.pending_decode_utterances == 0
    assert pipeline.decoding_utterance_id is None
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-035")
@pytest.mark.req("SP-LF-034")
@pytest.mark.req("SP-LF-029")
@pytest.mark.parametrize("input_format", [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32])
async def test_pipeline_keeps_exactly_300_ms_before_voice_for_each_pcm_format(
    input_format: EngineInputFormat,
) -> None:
    """発話の判定前の音を標本単位で 300 ms 含め、その最初の標本の時刻を返す。"""
    pack_format = "<h" if input_format is EngineInputFormat.PCM_S16LE else "<f"
    width = struct.calcsize(pack_format)
    low_chunks = [
        struct.pack(pack_format, value if width == 2 else value / 10_000) * 701
        for value in range(100, 108)
    ]
    prefix = b"".join(low_chunks)
    voice = struct.pack(pack_format, 600 if width == 2 else 0.05) * 400
    silence = struct.pack(pack_format, 0) * 1_600
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return ()

    async def source() -> AsyncIterator[bytes]:
        for chunk in (*low_chunks, voice, silence):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        decode_utterance=decode,
        input_format=input_format,
    )
    events = await _collect(pipeline.stream(source()))

    assert len(received) == 1
    assert received[0].pcm == prefix[-4_800 * width :] + voice + silence
    assert received[0].start_offset_s == pytest.approx(808 / 16_000)
    assert events[0] == UtteranceStart(utterance_id=1, ts_offset_s=pytest.approx(808 / 16_000))


@pytest.mark.req("SP-LF-035")
@pytest.mark.req("SP-LF-034")
@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-004")
@pytest.mark.parametrize("input_format", [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32])
@pytest.mark.parametrize("preroll_s, expected_samples", [(0.125, 2_000), (0, 0)])
async def test_pipeline_uses_configured_preroll_on_sample_boundaries(
    input_format: EngineInputFormat, preroll_s: float, expected_samples: int
) -> None:
    """指定した長さだけ発話の前の音を取り、PCM と開始時刻を同じ標本境界に揃える。"""
    pack_format = "<h" if input_format is EngineInputFormat.PCM_S16LE else "<f"
    width = struct.calcsize(pack_format)
    low = struct.pack(pack_format, 100 if width == 2 else 0.01) * 4_800
    voice = struct.pack(pack_format, 600 if width == 2 else 0.05) * 400
    silence = struct.pack(pack_format, 0) * 1_600
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return ()

    async def source() -> AsyncIterator[bytes]:
        for chunk in (low, voice, silence):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.05,
            preroll_s=preroll_s,
        ),
        decode_utterance=decode,
        input_format=input_format,
    )
    events = await _collect(pipeline.stream(source()))

    expected_start_s = (4_800 - expected_samples) / 16_000
    expected_pcm = (
        low[-expected_samples * width :] + voice + silence if expected_samples else voice + silence
    )
    assert len(received) == 1
    assert received[0].pcm == expected_pcm
    assert received[0].start_offset_s == pytest.approx(expected_start_s)
    assert events[0] == UtteranceStart(utterance_id=1, ts_offset_s=pytest.approx(expected_start_s))


@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-031")
async def test_pipeline_uses_boundary_settings_for_voice_and_silence() -> None:
    """一括 decode pipeline は渡した閾値で発話を始め、渡した無音の長さで閉じる。"""
    low = _pcm_chunk(6_000, duration_s=0.02)
    voice = _pcm_chunk(9_000, duration_s=0.02)
    silence = _pcm_chunk(0, duration_s=0.12)
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return ()

    async def source() -> AsyncIterator[bytes]:
        for chunk in (low, voice, silence):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.1,
            voice_threshold=0.25,
            preroll_s=0,
        ),
        decode_utterance=decode,
    )
    events = await _collect(pipeline.stream(source()))

    assert [utterance.pcm for utterance in received] == [voice + silence]
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=pytest.approx(0.02)),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.14)),
    ]


@pytest.mark.req("SP-LF-034")
@pytest.mark.req("SP-LF-029")
async def test_pipeline_limits_preroll_to_input_start() -> None:
    """入力開始から 300 ms 未満で発話が始まる時、発話の始まりの前に取る音と開始時刻は入力の始まりで止まる。"""
    low = _pcm_chunk(300, duration_s=0.1)
    voice = _pcm_chunk(600, duration_s=0.02)
    silence = _pcm_chunk(0, duration_s=0.1)
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return ()

    async def source() -> AsyncIterator[bytes]:
        for chunk in (low, voice, silence):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.05,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    events = await _collect(pipeline.stream(source()))

    assert len(received) == 1
    assert received[0].pcm == low + voice + silence
    assert received[0].start_offset_s == 0.0
    assert events[0] == UtteranceStart(utterance_id=1, ts_offset_s=0.0)


@pytest.mark.req("SP-LF-034")
@pytest.mark.req("SP-LF-029")
async def test_pipeline_does_not_reuse_pcm_from_a_closed_utterance() -> None:
    """次の発話の始まりの前に取る音は前の発話の音の末尾から始まり、二つの発話の音は重ならない。"""
    first_voice = _pcm_chunk(600, duration_s=0.02)
    long_silence = _pcm_chunk(0, duration_s=0.5)
    low = _pcm_chunk(300, duration_s=0.05)
    second_voice = _pcm_chunk(700, duration_s=0.02)
    short_silence = _pcm_chunk(0, duration_s=0.05)
    received: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received.append(utterance)
        return ()

    async def source() -> AsyncIterator[bytes]:
        for chunk in (first_voice, long_silence, low, second_voice, short_silence):
            yield chunk

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.05,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    events = await _collect(pipeline.stream(source()))

    assert [utterance.pcm for utterance in received] == [
        first_voice + long_silence,
        low + second_voice + short_silence,
    ]
    first_pcm_end_s = received[0].start_offset_s + len(received[0].pcm) / 2 / 16_000
    assert events[1] == UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.07))
    assert events[2] == UtteranceStart(utterance_id=2, ts_offset_s=pytest.approx(0.52))
    assert received[1].start_offset_s == pytest.approx(first_pcm_end_s)
    assert received[1].start_offset_s == pytest.approx(events[2].ts_offset_s)


@pytest.mark.req("SP-LF-004")
@pytest.mark.parametrize(
    ("input_format", "voice", "silence", "invalid"),
    [
        (
            EngineInputFormat.PCM_S16LE,
            struct.pack("<h", 16384),
            struct.pack("<h", 0),
            b"\x00",
        ),
        (
            EngineInputFormat.PCM_F32,
            struct.pack("<f", 0.5),
            struct.pack("<f", 0.0),
            b"\x00\x00\x00",
        ),
    ],
)
async def test_ac_24_11_pipeline_validates_before_advancing_any_state(
    input_format: EngineInputFormat,
    voice: bytes,
    silence: bytes,
    invalid: bytes,
) -> None:
    """不正な PCM chunk は音声の処理前に拒否し、後続の正しい入力を受けられる。"""
    decoded: list[ClosedUtterance] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        decoded.append(utterance)
        return ()

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=1 / 16_000,
            voice_threshold=0.5,
        ),
        decode_utterance=decode,
        input_format=input_format,
    )

    async def invalid_source() -> AsyncIterator[bytes]:
        yield invalid

    error = None
    try:
        await _collect(pipeline.stream(invalid_source()))
    except ValueError as caught:
        error = caught
    assert error is not None
    assert "whole samples" in str(error)
    assert pipeline.processed_pcm_chunks == 0
    assert pipeline.emitted_events == 0
    assert decoded == []

    async def valid_source() -> AsyncIterator[bytes]:
        yield voice
        yield silence

    events = await _collect(pipeline.stream(valid_source()))

    assert [type(event) for event in events] == [UtteranceStart, UtteranceEnd]
    assert len(decoded) == 1
    assert decoded[0].pcm == voice + silence
    assert pipeline.processed_pcm_chunks == 2


@pytest.mark.req("SP-LF-054")
@pytest.mark.req("SP-LF-001")
@pytest.mark.req("SP-LF-014")
async def test_ac_24_12_18_overlap_is_rejected_and_sequential_progress_accumulates() -> None:
    """pipeline 単体は重なる stream を先行 stream を壊さず拒否し、順次利用では処理量を累積する。"""
    first_source = _ControlledPCM()
    first_started = asyncio.Event()
    first_release = asyncio.Event()

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        if utterance.utterance_id == 1 and not first_started.is_set():
            first_started.set()
            await first_release.wait()
        return ()

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.01,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    first_collector = asyncio.create_task(_collect(pipeline.stream(first_source)))
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.01)
    await first_source.send(voice)
    await first_source.send(silence)
    await first_started.wait()

    async def empty_source() -> AsyncIterator[bytes]:
        if False:
            yield b""

    with pytest.raises(RuntimeError, match="already has an active stream"):
        await anext(pipeline.stream(empty_source()))
    assert pipeline.decoding_utterance_id == 1

    await first_source.finish()
    first_release.set()
    first_events = await first_collector
    first_time = pipeline.last_progress_monotonic_s

    assert [type(event) for event in first_events] == [UtteranceStart, UtteranceEnd]
    assert pipeline.processed_pcm_chunks == 2
    assert pipeline.emitted_events == 2
    assert first_time is not None

    async def second_source() -> AsyncIterator[bytes]:
        yield voice
        yield silence

    second_events = await _collect(pipeline.stream(second_source()))
    second_time = pipeline.last_progress_monotonic_s

    assert [type(event) for event in second_events] == [UtteranceStart, UtteranceEnd]
    assert pipeline.processed_pcm_chunks == 4
    assert pipeline.emitted_events == 4
    assert second_time is not None
    assert second_time > first_time

    third_events = await _collect(pipeline.stream(empty_source()))

    assert third_events == []
    assert pipeline.processed_pcm_chunks == 4
    assert pipeline.emitted_events == 4
    assert pipeline.last_progress_monotonic_s == second_time


class _RaisingIterable:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __iter__(self) -> _RaisingIterable:
        return self

    def __next__(self) -> ProvisionalObservation | FinalObservation:
        raise self._error


@pytest.mark.req("SP-LF-032")
@pytest.mark.parametrize("failure_location", ["callback", "iterable"])
async def test_ac_24_9_decode_failure_preserves_identity_and_stops_pending_work(
    failure_location: str,
) -> None:
    """decode が失敗した時はその例外を返し、後続の発話を処理せず状態を解放する。"""
    cleanup_error = RuntimeError("producer cleanup failed later")
    source = _ControlledPCM(cancellation_error=cleanup_error)
    failure = LookupError(f"decode failed in {failure_location}")
    started = asyncio.Event()
    release = asyncio.Event()
    received_ids: list[int] = []

    async def decode(
        utterance: ClosedUtterance,
    ) -> Iterable[ProvisionalObservation | FinalObservation]:
        received_ids.append(utterance.utterance_id)
        started.set()
        await release.wait()
        if failure_location == "callback":
            raise failure
        return _RaisingIterable(failure)

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.01,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    collector = asyncio.create_task(_collect(pipeline.stream(source)))
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.01)

    await source.send(voice)
    await source.send(silence)
    await started.wait()
    await source.send(voice)
    await source.send(silence)
    assert pipeline.pending_decode_utterances == 1
    release.set()

    caught = None
    try:
        await collector
    except LookupError as error:
        caught = error
    assert caught is failure
    assert received_ids == [1]
    assert source.cancelled.is_set()
    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-054")
@pytest.mark.req("SP-LF-032")
@pytest.mark.req("SP-LF-055")
async def test_ac_24_10_17_input_failure_waits_for_callback_and_keeps_first_cause() -> None:
    """入力の失敗時は decode に取り消しを届け、後始末を待って最初の例外を返す。"""
    source = _ControlledPCM()
    input_failure = OSError("input failed first")
    callback_failure = RuntimeError("callback failed during cleanup")
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    callback_cancelled = asyncio.Event()
    received_ids: list[int] = []

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        received_ids.append(utterance.utterance_id)
        callback_started.set()
        try:
            await callback_release.wait()
        except asyncio.CancelledError:
            callback_cancelled.set()
            await callback_release.wait()
        raise callback_failure

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.01,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    collector = asyncio.create_task(_collect(pipeline.stream(source)))
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.01)
    await source.send(voice)
    await source.send(silence)
    await callback_started.wait()
    await source.send(voice)
    await source.send(silence)
    await source.fail(input_failure)
    try:
        await asyncio.wait_for(callback_cancelled.wait(), timeout=1)
    except TimeoutError:
        callback_release.set()
        await asyncio.gather(collector, return_exceptions=True)
        raise

    assert not collector.done()
    assert callback_cancelled.is_set()
    assert pipeline.decoding_utterance_id == 1
    assert pipeline.pending_decode_utterances == 1

    async def empty_source() -> AsyncIterator[bytes]:
        if False:
            yield b""

    with pytest.raises(RuntimeError, match="already has an active stream"):
        await anext(pipeline.stream(empty_source()))

    callback_release.set()
    caught = None
    try:
        await collector
    except OSError as error:
        caught = error
    assert caught is input_failure
    assert received_ids == [1]
    assert callback_cancelled.is_set()
    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-054")
@pytest.mark.req("SP-LF-032")
@pytest.mark.req("SP-LF-055")
async def test_ac_24_14_17_consumer_close_waits_for_callback_before_reuse() -> None:
    """利用側が stream を閉じる時は decode に取り消しを届け、後始末を待って再利用できる。"""
    source = _ControlledPCM()
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        del utterance
        callback_started.set()
        try:
            await callback_release.wait()
        except asyncio.CancelledError:
            callback_cancelled.set()
            await callback_release.wait()
        return ()

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(
            silence_threshold_s=0.01,
            voice_threshold=500 / 32768,
        ),
        decode_utterance=decode,
    )
    stream = pipeline.stream(source)
    first_event_task = asyncio.create_task(anext(stream))
    voice = _pcm_chunk(600, duration_s=0.01)
    silence = _pcm_chunk(0, duration_s=0.01)
    await source.send(voice)
    assert await first_event_task == UtteranceStart(utterance_id=1, ts_offset_s=0.0)
    await source.send(silence)
    await callback_started.wait()

    close_task = asyncio.create_task(stream.aclose())
    await source.cancelled.wait()
    try:
        await asyncio.wait_for(callback_cancelled.wait(), timeout=1)
    except TimeoutError:
        callback_release.set()
        await close_task
        raise
    assert not close_task.done()
    assert callback_cancelled.is_set()
    assert pipeline.decoding_utterance_id == 1

    async def empty_source() -> AsyncIterator[bytes]:
        if False:
            yield b""

    with pytest.raises(RuntimeError, match="already has an active stream"):
        await anext(pipeline.stream(empty_source()))

    callback_release.set()
    await close_task

    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0
    assert callback_cancelled.is_set()
    assert await _collect(pipeline.stream(empty_source())) == []
    assert _pipeline_tasks() == []


@pytest.mark.req("SP-LF-056")
@pytest.mark.parametrize("external_owner", [False, True])
async def test_progress_tracks_silent_pcm_and_empty_decode_without_event_reading(
    external_owner: bool,
) -> None:
    """処理の時刻は無音の PCM と空の decode で進み、event の読み取りには依存しない。"""
    source = _ControlledPCM()
    progress: list[int] = []
    decoded = asyncio.Event()

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        del utterance
        decoded.set()
        return ()

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(silence_threshold_s=0.01),
        decode_utterance=decode,
        on_progress=(lambda: progress.append(1)) if external_owner else None,
    )
    stream = pipeline.stream(source)
    reading = asyncio.create_task(anext(stream))
    await source.send(_pcm_chunk(0, duration_s=0.01))
    assert not reading.done()
    silent_time = pipeline.last_progress_monotonic_s
    assert (silent_time is None) is external_owner
    if external_owner:
        assert progress == [1]
    await source.send(_pcm_chunk(600, duration_s=0.01))
    assert isinstance(await reading, UtteranceStart)
    await source.finish()
    await decoded.wait()
    await asyncio.sleep(0)
    finished_time = pipeline.last_progress_monotonic_s
    if external_owner:
        assert finished_time is None
        assert len(progress) == 3
    else:
        assert finished_time > silent_time
    assert await _collect(stream) == [UtteranceEnd(1, 0.02)]
    assert pipeline.last_progress_monotonic_s == finished_time
    if external_owner:
        assert len(progress) == 3


@pytest.mark.req("SP-LF-055")
async def test_input_failure_cancels_a_callback_that_stops_immediately() -> None:
    """入力の失敗はすぐ止まる decode にも取り消しを届け、最初の入力例外を返す。"""
    source = _ControlledPCM()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    failure = OSError("input failed")

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        del utterance
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(silence_threshold_s=0.01),
        decode_utterance=decode,
    )
    collecting = asyncio.create_task(_collect(pipeline.stream(source)))
    await source.send(_pcm_chunk(600, duration_s=0.01))
    await source.send(_pcm_chunk(0, duration_s=0.01))
    await started.wait()
    await source.fail(failure)
    with pytest.raises(OSError) as caught:
        await collecting
    assert caught.value is failure
    assert cancelled.is_set()
    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0


@pytest.mark.req("SP-LF-055")
async def test_repeated_consumer_cancellation_waits_for_decode_cleanup() -> None:
    """利用側の重なる取り消しは decode の後始末を中断せず、終わった後に再利用できる。"""
    source = _ControlledPCM()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finish = asyncio.Event()
    cleaned = asyncio.Event()

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        del utterance
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
            await finish.wait()
            cleaned.set()
        return ()

    pipeline = UtteranceDecodePipeline(
        boundary_settings=listenflow.UtteranceBoundarySettings(silence_threshold_s=0.01),
        decode_utterance=decode,
    )
    collecting = asyncio.create_task(_collect(pipeline.stream(source)))
    await source.send(_pcm_chunk(600, duration_s=0.01))
    await source.send(_pcm_chunk(0, duration_s=0.01))
    await started.wait()
    collecting.cancel()
    await cancelled.wait()
    collecting.cancel()
    await asyncio.sleep(0)
    try:
        assert not collecting.done()
    finally:
        finish.set()
        await asyncio.gather(collecting, return_exceptions=True)
    assert cleaned.is_set()
    assert pipeline.decoding_utterance_id is None
    assert pipeline.pending_decode_utterances == 0
    assert await _collect(pipeline.stream(_empty_pcm())) == []


async def _empty_pcm() -> AsyncIterator[bytes]:
    if False:
        yield b""


@pytest.mark.req("SP-LF-056")
@pytest.mark.req("SP-LF-058")
async def test_pipeline_engine_keeps_progress_in_the_parent_while_consumer_pauses() -> None:
    """pipeline を持つ engine は event の読み取りによらず health() の処理の時刻を記録する。

    text にならなかった発話も UtteranceStart と UtteranceEnd の対で閉じ、error にしない。
    """
    decoded = asyncio.Event()

    async def decode(utterance: ClosedUtterance) -> Iterable[FinalObservation]:
        del utterance
        decoded.set()
        return ()

    class PipelineEngine(STTEngineBase):
        def __init__(self) -> None:
            super().__init__()
            self.pipeline = UtteranceDecodePipeline(
                decode_utterance=decode,
                on_progress=self._mark_progress,
            )

        @property
        def capabilities(self) -> listenflow.EngineCapabilities:
            return listenflow.EngineCapabilities(
                transport=listenflow.EngineTransport.IN_PROCESS,
                input_format=EngineInputFormat.PCM_S16LE,
                partial_mode=listenflow.PartialMode.NONE,
                finalization=frozenset({listenflow.FinalizationMode.LOCAL_VAD}),
                timestamp_detail=listenflow.TimestampDetail.SEGMENT,
                diarization=listenflow.DiarizationMode.NONE,
                languages=frozenset(),
                auto_language=False,
            )

        async def _load(self) -> None:
            pass

        async def _release(self) -> None:
            pass

        def _recognize(
            self, pcm_chunks: AsyncIterator[bytes]
        ) -> AsyncIterator[listenflow.TranscriptEvent]:
            return self.pipeline.stream(pcm_chunks)

        def _health_details(self) -> dict[str, object]:
            return {
                "processed_pcm_chunks": self.pipeline.processed_pcm_chunks,
                "emitted_events": self.pipeline.emitted_events,
                "decoding_utterance_id": self.pipeline.decoding_utterance_id,
                "pending_decode_utterances": self.pipeline.pending_decode_utterances,
            }

    engine = PipelineEngine()
    await engine.prepare()
    source = _ControlledPCM()
    stream = engine.stream(source)
    first = asyncio.create_task(anext(stream))
    await source.send(_pcm_chunk(0, duration_s=0.01))
    silence_time = engine.health().last_progress_monotonic_s
    assert silence_time is not None
    await source.send(_pcm_chunk(600, duration_s=0.01))
    assert isinstance(await first, UtteranceStart)
    await source.finish()
    await decoded.wait()
    await asyncio.sleep(0)
    decode_time = engine.health().last_progress_monotonic_s
    assert decode_time > silence_time
    assert engine.pipeline.last_progress_monotonic_s is None
    assert engine.health().processed_pcm_chunks == 2
    assert await _collect(stream) == [UtteranceEnd(1, 0.02)]
    assert engine.health().last_progress_monotonic_s == decode_time
    await engine.aclose()
