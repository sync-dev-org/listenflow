import inspect
import struct
from dataclasses import FrozenInstanceError

import pytest

from listenflow import (
    DiarizationMode,
    EngineCapabilities,
    EngineInputFormat,
    EngineTransport,
    Final,
    FinalizationMode,
    FinalObservation,
    Partial,
    PartialMode,
    ProvisionalObservation,
    STTEngineBase,
    TimestampDetail,
    UtteranceBoundarySession,
    UtteranceEnd,
    UtteranceStart,
    has_voice,
)


def _pcm_chunk(sample: int, *, duration_s: float) -> bytes:
    sample_count = round(16_000 * duration_s)
    return sample.to_bytes(2, byteorder="little", signed=True) * sample_count


@pytest.mark.req("SP-LF-001")
@pytest.mark.req("SP-LF-010")
@pytest.mark.req("SP-LF-014")
@pytest.mark.req("SP-LF-050")
@pytest.mark.req("SP-LF-098")
def test_engine_base_keeps_public_method_signatures_and_requires_capabilities() -> None:
    """engine は prepare()、stream()、health() を公開し、capabilities の実装を求める。"""
    assert str(inspect.signature(STTEngineBase.prepare)) == "(self) -> 'None'"
    assert str(inspect.signature(STTEngineBase.stream)) == (
        "(self, pcm_chunks: 'AsyncIterator[bytes]', *, "
        "segmentation: 'Segmentation' = <Segmentation.AUTO: 'auto'>) -> 'AsyncIterator[TranscriptEvent]'"
    )
    assert str(inspect.signature(STTEngineBase.health)) == "(self) -> 'HealthSnapshot'"
    assert STTEngineBase.capabilities.__isabstractmethod__


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
def test_voice_then_silence_emits_source_timed_utterance_boundaries() -> None:
    """音声の発話は元の音声の時刻を持つ開始と終了の通知で閉じる。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )

    events = session.observe_pcm(_pcm_chunk(600, duration_s=0.01))
    events += session.observe_pcm(_pcm_chunk(0, duration_s=0.2))
    end_event = session.maybe_end_event()
    assert end_event is not None
    events.append(end_event)

    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.11),
    ]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-007")
def test_provisional_observations_are_replace_semantic_and_deduplicated_by_text() -> None:
    """途中の仮説は前の仮説を置き換え、話者と元の音声の時刻を保つ。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )

    events = session.observe_provisional(
        ProvisionalObservation(text="hel", speaker="speaker-a", source_offset_s=0.1)
    )
    duplicate = session.observe_provisional(
        ProvisionalObservation(text="hel", speaker="speaker-b", source_offset_s=0.2)
    )
    events += session.observe_provisional(
        ProvisionalObservation(text="hello", speaker="speaker-a", source_offset_s=0.3)
    )

    assert duplicate == []
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.1),
        Partial(
            utterance_id=1,
            ts_offset_s=0.1,
            text="hel",
            speaker="speaker-a",
        ),
        Partial(
            utterance_id=1,
            ts_offset_s=0.3,
            text="hello",
            speaker="speaker-a",
        ),
    ]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-007")
def test_final_observation_supplies_start_and_partial_and_preserves_metadata() -> None:
    """確定した text は発話の開始、途中の仮説、話者、元の音声の時刻とともに通知される。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )

    events = session.observe_final(FinalObservation(text="hello", speaker=7, source_offset_s=0.0))

    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        Partial(utterance_id=1, ts_offset_s=0.0, text="hello", speaker=7),
        Final(utterance_id=1, ts_offset_s=0.0, text="hello", speaker=7),
    ]


@pytest.mark.req("SP-LF-005")
def test_duplicate_final_observation_is_not_emitted_twice() -> None:
    """同じ確定の観測を同じ発話へ再び適用しても通知を増やさない。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )
    observation = FinalObservation(text="hello", speaker=7, source_offset_s=0.0)

    first_events = session.observe_final(observation)
    duplicate_events = session.observe_final(observation)

    assert [event for event in first_events if isinstance(event, Final)] == [
        Final(utterance_id=1, ts_offset_s=0.0, text="hello", speaker=7)
    ]
    assert duplicate_events == []


@pytest.mark.req("SP-LF-006")
def test_received_final_remains_unchanged_during_later_transcription() -> None:
    """利用側が保持した Final の text、発話の ID、時刻と話者は、後の仮説と確定でも変わらない。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )
    session.observe_pcm(_pcm_chunk(600, duration_s=0.1))
    first_events = session.observe_final(
        FinalObservation(text="committed", speaker=7, source_offset_s=0.05)
    )
    retained_final = next(event for event in first_events if isinstance(event, Final))
    assert (
        retained_final.text,
        retained_final.utterance_id,
        retained_final.ts_offset_s,
        retained_final.speaker,
    ) == ("committed", 1, 0.05, 7)

    for utterance_id, partial_offset_s, final_offset_s in ((1, 0.06, 0.08), (2, 0.35, 0.38)):
        if utterance_id == 2:
            session.observe_pcm(_pcm_chunk(0, duration_s=0.2))
            assert session.maybe_end_event() == UtteranceEnd(1, 0.2)
            assert session.observe_pcm(_pcm_chunk(600, duration_s=0.1)) == [
                UtteranceStart(2, pytest.approx(0.3))
            ]

        partial_events = session.observe_provisional(
            ProvisionalObservation(text="pending", speaker=8, source_offset_s=partial_offset_s)
        )
        assert (
            retained_final.text,
            retained_final.utterance_id,
            retained_final.ts_offset_s,
            retained_final.speaker,
        ) == ("committed", 1, 0.05, 7)
        assert partial_events == [Partial(utterance_id, partial_offset_s, "pending", 8)]

        final_events = session.observe_final(
            FinalObservation(text="later", speaker=9, source_offset_s=final_offset_s)
        )
        assert (
            retained_final.text,
            retained_final.utterance_id,
            retained_final.ts_offset_s,
            retained_final.speaker,
        ) == ("committed", 1, 0.05, 7)
        assert final_events == [Final(utterance_id, final_offset_s, "later", 9)]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-008")
def test_input_end_finalizes_pending_provisional_before_utterance_end() -> None:
    """入力の終端では元の音声の時刻を保つ確定した text を返してから発話を閉じる。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )
    session.observe_pcm(_pcm_chunk(600, duration_s=0.01))
    session.observe_provisional(
        ProvisionalObservation(text="pending", speaker=3, source_offset_s=0.005)
    )

    session.finish_input()
    events = session.finalize_events()

    assert events == [
        Final(utterance_id=1, ts_offset_s=0.005, text="pending", speaker=3),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.11),
    ]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-008")
def test_new_provisional_after_final_is_finalized_before_input_end() -> None:
    """確定した text の後の仮説も元の音声の時刻で確定し、入力の終端で発話を閉じる。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )
    session.observe_final(FinalObservation(text="committed", speaker=1, source_offset_s=0.0))
    session.observe_provisional(
        ProvisionalObservation(text="pending", speaker=2, source_offset_s=0.2)
    )

    session.finish_input()

    assert session.maybe_end_event() is None
    assert session.finalize_events() == [
        Final(utterance_id=1, ts_offset_s=0.2, text="pending", speaker=2),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.2),
    ]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-008")
def test_reasserted_provisional_after_final_is_finalized_at_input_end() -> None:
    """再び届いた仮説は重複する途中通知を出さず、元の音声の時刻で確定して発話を閉じる。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )
    session.observe_provisional(
        ProvisionalObservation(text="pending", speaker=2, source_offset_s=0.2)
    )
    session.observe_final(FinalObservation(text="committed", speaker=1, source_offset_s=0.0))
    assert (
        session.observe_provisional(
            ProvisionalObservation(text="pending", speaker=2, source_offset_s=0.4)
        )
        == []
    )

    session.finish_input()

    assert session.maybe_end_event() is None
    assert session.finalize_events() == [
        Final(utterance_id=1, ts_offset_s=0.2, text="pending", speaker=2),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.2),
    ]


@pytest.mark.req("SP-LF-010")
def test_capability_profile_is_typed_and_deeply_immutable() -> None:
    """engine は能力を型付きで変更できない値として公開する。"""
    profile = EngineCapabilities(
        transport=EngineTransport.WEBSOCKET,
        input_format=EngineInputFormat.PCM_S16LE,
        partial_mode=PartialMode.REPLACE,
        finalization=frozenset({FinalizationMode.ENGINE_SEGMENT, FinalizationMode.LOCAL_VAD}),
        timestamp_detail=TimestampDetail.SEGMENT,
        diarization=DiarizationMode.NONE,
        languages=frozenset({"en"}),
        auto_language=False,
    )

    assert profile.finalization == frozenset(
        {FinalizationMode.ENGINE_SEGMENT, FinalizationMode.LOCAL_VAD}
    )
    with pytest.raises(FrozenInstanceError):
        profile.transport = EngineTransport.GRPC
    with pytest.raises(AttributeError):
        profile.finalization.add(FinalizationMode.CLIENT_FLUSH)


@pytest.mark.req("SP-LF-029")
def test_backlog_events_preserve_their_source_timestamps() -> None:
    """遅れて届いた確定した text の時刻は元の音声の位置を保つ。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.8,
        voice_threshold=500 / 32768,
    )
    session.input_offset_s = 100.0

    events = session.observe_provisional(ProvisionalObservation(text="pending"))
    events += session.observe_final(FinalObservation(text="alpha", speaker=1, source_offset_s=10.0))
    session.pending_end_ts = 11.0
    end_event = session.maybe_end_event()
    assert end_event is not None
    events.append(end_event)

    assert [event.ts_offset_s for event in events] == [100.0, 100.0, 10.0, 11.0]


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
def test_voice_after_empty_utterance_uses_a_new_utterance_id() -> None:
    """text のない発話を閉じた後の発話には別の ID と元の音声の時刻を付ける。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )

    session.observe_pcm(_pcm_chunk(600, duration_s=0.01))
    session.observe_pcm(_pcm_chunk(0, duration_s=0.2))
    assert session.maybe_end_event() is not None

    next_start_events = session.observe_pcm(_pcm_chunk(600, duration_s=0.01))

    assert len(next_start_events) == 1
    next_start = next_start_events[0]
    assert isinstance(next_start, UtteranceStart)
    assert next_start.utterance_id == 2
    assert next_start.ts_offset_s == pytest.approx(0.21)


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
def test_partial_waits_for_final_after_silence_threshold() -> None:
    """途中の仮説が残る間は元の音声の時刻を保って発話の終了を待つ。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )

    session.observe_pcm(_pcm_chunk(600, duration_s=0.01))
    partial_events = session.observe_provisional(ProvisionalObservation(text="pending"))
    session.observe_pcm(_pcm_chunk(0, duration_s=0.2))

    assert partial_events == [
        Partial(
            utterance_id=1,
            ts_offset_s=0.01,
            text="pending",
            speaker=None,
        )
    ]
    assert session.maybe_end_event() is None
    assert session.in_utterance
    assert session.utterance_id == 1


@pytest.mark.req("SP-LF-029")
@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-008")
def test_delayed_transcript_after_input_end_closes_its_new_utterance() -> None:
    """入力の終端後に届いた text も元の音声の時刻を保つ独立した発話として閉じる。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=500 / 32768,
    )

    session.observe_pcm(_pcm_chunk(600, duration_s=0.01))
    session.finish_input()
    assert session.maybe_end_event() == UtteranceEnd(
        utterance_id=1,
        ts_offset_s=0.11,
    )

    events = session.observe_provisional(ProvisionalObservation(text="late"))
    events += session.observe_final(
        FinalObservation(text="late", speaker=None, source_offset_s=0.01)
    )
    end_event = session.maybe_end_event()

    assert [type(event) for event in events] == [UtteranceStart, Partial, Final]
    assert {event.utterance_id for event in events} == {2}
    assert end_event == UtteranceEnd(utterance_id=2, ts_offset_s=0.01)


@pytest.mark.req("SP-LF-004")
@pytest.mark.parametrize(
    ("input_format", "whole_chunk", "incomplete_chunk"),
    [
        (
            EngineInputFormat.PCM_S16LE,
            struct.pack("<4h", 10_000, 0, -10_000, 0),
            b"\x00",
        ),
        (
            EngineInputFormat.PCM_F32,
            struct.pack("<4f", 0.25, 0.0, -0.25, 0.0),
            b"\x00\x00\x00",
        ),
    ],
)
def test_ac_11_5_boundary_framing_preserves_state_and_uses_format_sample_width(
    input_format: EngineInputFormat,
    whole_chunk: bytes,
    incomplete_chunk: bytes,
) -> None:
    """PCM の chunk は sample の境界で受け取り、不正な chunk は拒否する。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=0.2,
        input_format=input_format,
    )
    before = (
        session.input_offset_s,
        session.in_utterance,
        session.last_voice_offset_s,
        session.pending_end_ts,
    )

    with pytest.raises(ValueError, match=f"{input_format.value}.*whole samples"):
        session.observe_pcm(incomplete_chunk)

    assert (
        session.input_offset_s,
        session.in_utterance,
        session.last_voice_offset_s,
        session.pending_end_ts,
    ) == before
    assert session.observe_pcm(whole_chunk) == [UtteranceStart(utterance_id=1, ts_offset_s=0.0)]
    assert session.input_offset_s == pytest.approx(4 / 16_000)


@pytest.mark.req("SP-LF-005")
@pytest.mark.parametrize(
    "invalid_threshold",
    [True, False, 0, 1, 500, "0.1", None, float("nan"), float("inf"), -float("inf"), -0.1, 1.1],
)
def test_ac_11_6_and_14_voice_threshold_validation_is_shared(
    invalid_threshold: object,
) -> None:
    """発話判定の閾値は有限な float の範囲外を拒否する。"""
    message = r"normalized full-scale.*\[0\.0, 1\.0\].*500.*0\.01526"

    with pytest.raises(ValueError, match=message):
        UtteranceBoundarySession(
            silence_threshold_s=0.1,
            voice_threshold=invalid_threshold,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=message):
        has_voice(
            b"",
            input_format=EngineInputFormat.PCM_S16LE,
            voice_threshold=invalid_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.req("SP-LF-005")
@pytest.mark.parametrize("voice_threshold", [0.0, 500 / 32768, 1.0])
def test_ac_11_6_voice_threshold_accepts_finite_float_endpoints(
    voice_threshold: float,
) -> None:
    """発話判定の閾値は 0.0 と 1.0 を受理する。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=voice_threshold,
    )

    assert session.voice_threshold == voice_threshold


@pytest.mark.req("SP-LF-004")
def test_ac_11_1_boundary_input_format_is_immutable_during_session() -> None:
    """UtteranceBoundarySession の入力 PCM 形式は構築後に変わらない。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=0.1,
        input_format=EngineInputFormat.PCM_F32,
    )

    with pytest.raises(AttributeError, match="input_format.*immutable"):
        session.input_format = EngineInputFormat.PCM_S16LE

    assert session.input_format is EngineInputFormat.PCM_F32


@pytest.mark.req("SP-LF-004")
def test_ac_11_1_boundary_input_format_cannot_be_deleted_and_reassigned() -> None:
    """UtteranceBoundarySession の入力 PCM 形式は処理中に削除して変更できない。"""
    session = UtteranceBoundarySession(
        silence_threshold_s=0.1,
        voice_threshold=0.1,
        input_format=EngineInputFormat.PCM_S16LE,
    )
    session.observe_pcm(struct.pack("<h", 0))

    with pytest.raises(AttributeError, match="input_format.*immutable"):
        del session.input_format

    assert session.input_format is EngineInputFormat.PCM_S16LE
    offset_before = session.input_offset_s
    session.observe_pcm(struct.pack("<f", 0.5))
    assert session.input_offset_s == pytest.approx(offset_before + 2 / 16_000)


@pytest.mark.req("SP-LF-004")
@pytest.mark.parametrize("invalid_format", [EngineInputFormat.WAV_PCM_S16LE, "pcm_s16le"])
def test_ac_11_2_and_14_input_format_validation_is_shared(invalid_format: object) -> None:
    """入力 PCM 形式には s16le または little-endian f32le を指定する。"""
    message = r"UtteranceBoundarySession.*input_format.*pcm_s16le.*pcm_f32"
    with pytest.raises(ValueError, match=message):
        UtteranceBoundarySession(
            silence_threshold_s=0.1,
            voice_threshold=0.1,
            input_format=invalid_format,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match=r"has_voice.*input_format.*pcm_s16le.*pcm_f32"):
        has_voice(
            b"",
            input_format=invalid_format,  # type: ignore[arg-type]
            voice_threshold=0.1,
        )


@pytest.mark.req("SP-LF-005")
@pytest.mark.parametrize(
    ("input_format", "below", "equal", "above"),
    [
        (
            EngineInputFormat.PCM_S16LE,
            struct.pack("<h", 16383),
            struct.pack("<h", 16384),
            struct.pack("<h", -20000),
        ),
        (
            EngineInputFormat.PCM_F32,
            struct.pack("<f", 0.499),
            struct.pack("<f", 0.5),
            struct.pack("<f", -1.5),
        ),
    ],
)
def test_ac_11_7_and_14_has_voice_is_inclusive_and_matches_boundary(
    input_format: EngineInputFormat,
    below: bytes,
    equal: bytes,
    above: bytes,
) -> None:
    """PCM の振幅が閾値と等しい時も発話ありと判定する。"""
    for chunk, expected in ((b"", False), (below, False), (equal, True), (above, True)):
        first = has_voice(chunk, input_format=input_format, voice_threshold=0.5)
        second = has_voice(chunk, input_format=input_format, voice_threshold=0.5)
        session = UtteranceBoundarySession(
            silence_threshold_s=0.1,
            voice_threshold=0.5,
            input_format=input_format,
        )

        assert first is second is expected
        assert bool(session.observe_pcm(chunk)) is expected


@pytest.mark.req("SP-LF-005")
@pytest.mark.parametrize(
    ("input_format", "zero", "below_one", "at_one", "over_range"),
    [
        (
            EngineInputFormat.PCM_S16LE,
            struct.pack("<h", 0),
            struct.pack("<h", 32767),
            struct.pack("<h", -32768),
            struct.pack("<h", -32768),
        ),
        (
            EngineInputFormat.PCM_F32,
            struct.pack("<f", 0.0),
            struct.pack("<f", 0.999),
            struct.pack("<f", 1.0),
            struct.pack("<f", 1.5),
        ),
    ],
)
def test_ac_11_7_has_voice_endpoint_semantics(
    input_format: EngineInputFormat,
    zero: bytes,
    below_one: bytes,
    at_one: bytes,
    over_range: bytes,
) -> None:
    """空の音声は発話なしとし、端点の振幅は発話ありとして扱う。"""
    assert has_voice(b"", input_format=input_format, voice_threshold=0.0) is False
    assert has_voice(zero, input_format=input_format, voice_threshold=0.0) is True
    assert has_voice(below_one, input_format=input_format, voice_threshold=1.0) is False
    assert has_voice(at_one, input_format=input_format, voice_threshold=1.0) is True
    assert has_voice(over_range, input_format=input_format, voice_threshold=1.0) is True


@pytest.mark.req("SP-LF-004")
@pytest.mark.parametrize(
    ("input_format", "incomplete_chunk", "width"),
    [
        (EngineInputFormat.PCM_S16LE, b"\x00", 2),
        (EngineInputFormat.PCM_F32, b"\x00\x00\x00", 4),
    ],
)
def test_ac_11_14_has_voice_rejects_incomplete_samples(
    input_format: EngineInputFormat,
    incomplete_chunk: bytes,
    width: int,
) -> None:
    """PCM の chunk は選んだ形式の sample 境界で区切られていなければならない。"""
    with pytest.raises(
        ValueError,
        match=rf"{input_format.value}.*multiple of {width} bytes.*whole samples",
    ):
        has_voice(
            incomplete_chunk,
            input_format=input_format,
            voice_threshold=0.1,
        )
