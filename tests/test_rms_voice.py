from __future__ import annotations

import inspect
import struct
from collections.abc import AsyncIterator

import pytest

import listenflow as lf


FORMATS = [lf.EngineInputFormat.PCM_S16LE, lf.EngineInputFormat.PCM_F32]


def _pcm(amplitude: float, samples: int, input_format: lf.EngineInputFormat) -> bytes:
    sample = (
        struct.pack("<h", round(amplitude * 32768))
        if input_format is lf.EngineInputFormat.PCM_S16LE
        else struct.pack("<f", amplitude)
    )
    return sample * samples


def _split(pcm: bytes, width: int, pattern: tuple[int, ...]) -> list[bytes]:
    chunks = [b""]
    position = 0
    index = 0
    while position < len(pcm):
        end = position + pattern[index % len(pattern)] * width
        chunks.extend([pcm[position:end], b""])
        position = end
        index += 1
    return chunks


async def _source(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def _decode(
    chunks: list[bytes],
    input_format: lf.EngineInputFormat,
    **settings: object,
) -> tuple[list[lf.TranscriptEvent], list[lf.ClosedUtterance]]:
    received: list[lf.ClosedUtterance] = []

    async def decode(utterance: lf.ClosedUtterance) -> list[lf.FinalObservation]:
        received.append(utterance)
        return []

    pipeline = lf.UtteranceDecodePipeline(
        boundary_settings=lf.UtteranceBoundarySettings(**settings),
        input_format=input_format,
        decode_utterance=decode,
    )
    events = [event async for event in pipeline.stream(_source(chunks))]
    assert pipeline.processed_pcm_chunks == len(chunks)
    return events, received


@pytest.mark.req("SP-LF-100")
@pytest.mark.req("SP-LF-034")
@pytest.mark.req("SP-LF-035")
@pytest.mark.parametrize("input_format", FORMATS)
async def test_rms_boundaries_and_preroll_pcm_do_not_depend_on_chunks(
    input_format: lf.EngineInputFormat,
) -> None:
    """同じ音声の二発話は chunk の分け方によらず同じ時刻とさかのぼった音を持つ。"""
    width = 2 if input_format is lf.EngineInputFormat.PCM_S16LE else 4
    pcm = b"".join(
        _pcm(amplitude, samples, input_format)
        for amplitude, samples in [(0.001, 768), (0.03125, 3072), (0.0, 13312), (0.0625, 1124)]
    )
    expected = (
        [
            lf.UtteranceStart(1, 288 / 16000),
            lf.UtteranceEnd(1, 16640 / 16000),
            lf.UtteranceStart(2, 16672 / 16000),
            lf.UtteranceEnd(2, 18276 / 16000),
        ],
        [
            lf.ClosedUtterance(1, pcm[288 * width : 16640 * width], 288 / 16000, 16640 / 16000),
            lf.ClosedUtterance(2, pcm[16672 * width :], 16672 / 16000, 18276 / 16000),
        ],
    )
    for chunks in ([pcm], _split(pcm, width, (256,)), _split(pcm, width, (1, 257, 509, 31))):
        assert await _decode(chunks, input_format, preroll_s=0.03) == expected


@pytest.mark.req("SP-LF-099")
@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
def test_rms_threshold_includes_equality_and_rejects_a_short_loud_click(
    input_format: lf.EngineInputFormat,
) -> None:
    """声は block の RMS が閾値以上の時だけ認め、短い大きな click は認めない。"""
    # 64 nonzero samples: RMS is 0.0625, while mean absolute amplitude is 0.03125.
    equal = (
        _pcm(0.125, 32, input_format)
        + _pcm(-0.125, 32, input_format)
        + _pcm(0.0, 192, input_format)
    )
    below = _pcm(2047 / 32768, 256, input_format)
    click = _pcm(0.5, 1, input_format) + _pcm(0.0, 255, input_format)
    for pcm, voiced in [(equal, True), (below, False), (click, False)]:
        session = lf.UtteranceBoundarySession(
            silence_threshold_s=0.8, voice_rms_threshold=0.0625, input_format=input_format
        )
        assert lf.has_voice(pcm, input_format=input_format, voice_rms_threshold=0.0625) is voiced
        assert session.observe_pcm(pcm) == ([lf.UtteranceStart(1, 0.0)] if voiced else [])


@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
def test_has_voice_checks_each_block_instead_of_diluting_voice_with_later_silence(
    input_format: lf.EngineInputFormat,
) -> None:
    """状態を持たない判定でも、一つの声の block を後続する長い無音で薄めない。"""
    pcm = _pcm(0.0625, 256, input_format) + _pcm(0.0, 25600, input_format)
    assert lf.has_voice(pcm, input_format=input_format, voice_rms_threshold=0.0625)


@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
@pytest.mark.parametrize("samples", [1, 17, 255])
async def test_input_end_measures_a_short_block_without_zero_padding(
    input_format: lf.EngineInputFormat, samples: int
) -> None:
    """入力の終わりの声は端数の sample 数で RMS を求め、音と時刻を欠かさず渡す。"""
    prefix = _pcm(0.0, 512, input_format)
    tail = _pcm(0.0625, samples, input_format)
    events, received = await _decode(
        [prefix, tail], input_format, voice_rms_threshold=0.0625, preroll_s=0.0
    )
    assert events == [lf.UtteranceStart(1, 0.032), lf.UtteranceEnd(1, (512 + samples) / 16000)]
    assert received == [lf.ClosedUtterance(1, tail, 0.032, (512 + samples) / 16000)]


@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
def test_rms_session_retains_samples_until_a_complete_block_arrives(
    input_format: lf.EngineInputFormat,
) -> None:
    """session は空の chunk で端数を確定せず、256 sample が揃った時に始まりを伝える。"""
    session = lf.UtteranceBoundarySession(
        silence_threshold_s=0.8, voice_rms_threshold=0.0625, input_format=input_format
    )
    assert session.observe_pcm(_pcm(0.0, 256, input_format)) == []
    assert session.observe_pcm(_pcm(0.0625, 255, input_format)) == []
    assert session.observe_pcm(b"") == []
    assert session.observe_pcm(_pcm(-0.0625, 1, input_format)) == [lf.UtteranceStart(1, 0.016)]
    assert session.last_voice_offset_s == 0.032


@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
@pytest.mark.parametrize("samples", [17, 256])
def test_session_finishes_at_real_input_end_and_drains_a_tail_start_before_end(
    input_format: lf.EngineInputFormat, samples: int
) -> None:
    """session の終端は未入力の無音を足さず、端数で初めて声が来ても始まりの後に閉じる。"""
    session = lf.UtteranceBoundarySession(
        silence_threshold_s=0.8, voice_rms_threshold=0.0625, input_format=input_format
    )
    events = session.observe_pcm(_pcm(0.0625, samples, input_format))
    assert events == ([lf.UtteranceStart(1, 0.0)] if samples == 256 else [])
    assert session.finish_input() is None
    events.extend(session.finalize_events())
    assert events == [lf.UtteranceStart(1, 0.0), lf.UtteranceEnd(1, samples / 16000)]
    assert session.finalize_events() == []


@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-095")
@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
@pytest.mark.parametrize("observation_kind", ["provisional", "final"])
def test_tail_start_precedes_recognition_observations_received_after_input_end(
    input_format: lf.EngineInputFormat, observation_kind: str
) -> None:
    """端数で始まる発話にも遅れた認識結果より先に始まりを伝え、通知を二重に出さない。"""
    session = lf.UtteranceBoundarySession(
        silence_threshold_s=0.8, voice_rms_threshold=0.01, input_format=input_format
    )
    assert session.observe_pcm(_pcm(0.0625, 17, input_format)) == []
    session.finish_input()
    if observation_kind == "provisional":
        observation = lf.ProvisionalObservation("tail", source_offset_s=0.0)
        events = session.observe_provisional(observation)
        assert session.observe_provisional(observation) == []
    else:
        observation = lf.FinalObservation("tail", 0.0)
        events = session.observe_final(observation)
        assert session.observe_final(observation) == []
    events.extend(session.finalize_events())
    assert events == [
        lf.UtteranceStart(1, 0.0),
        lf.Partial(1, 0.0, "tail"),
        lf.Final(1, 0.0, "tail"),
        lf.UtteranceEnd(1, 17 / 16000),
    ]
    assert session.finalize_events() == []
    assert session.maybe_end_event() is None


@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-095")
@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
def test_tail_end_waits_until_its_start_has_been_delivered(
    input_format: lf.EngineInputFormat,
) -> None:
    """端数で始まった発話の Start が未通知なら空の観測で捨てず、End だけを先に返さない。"""
    session = lf.UtteranceBoundarySession(
        silence_threshold_s=0.8, voice_rms_threshold=0.01, input_format=input_format
    )
    assert session.observe_pcm(_pcm(0.0625, 17, input_format)) == []
    session.finish_input()
    assert session.observe_provisional(lf.ProvisionalObservation("")) == []
    assert session.observe_final(lf.FinalObservation("", 0.0)) == []
    assert session.maybe_end_event() is None
    assert session.finalize_events() == [lf.UtteranceStart(1, 0.0), lf.UtteranceEnd(1, 17 / 16000)]
    assert session.finalize_events() == []
    assert session.maybe_end_event() is None


@pytest.mark.req("SP-LF-094")
@pytest.mark.req("SP-LF-100")
@pytest.mark.parametrize("input_format", FORMATS)
@pytest.mark.parametrize("voiced_samples", [3072, 3328])
async def test_rms_minimum_voiced_span_uses_blocks_and_closes_after_exactly_point_eight_seconds(
    input_format: lf.EngineInputFormat, voiced_samples: int
) -> None:
    """0.192 秒の声は捨て、0.208 秒の声は渡し、最後の声から無音 0.8 秒で閉じる。"""
    pcm = _pcm(0.03125, voiced_samples, input_format) + _pcm(0.0, 13056, input_format)
    events, received = await _decode([pcm], input_format, min_voiced_s=0.2, preroll_s=0.0)
    end = (voiced_samples + 12800) / 16000
    assert events == [lf.UtteranceStart(1, 0.0), lf.UtteranceEnd(1, end)]
    assert received == (
        [
            lf.ClosedUtterance(
                1,
                pcm[: (voiced_samples + 12800) * (2 if input_format is FORMATS[0] else 4)],
                0.0,
                end,
            )
        ]
        if voiced_samples >= 3200
        else []
    )


@pytest.mark.req("SP-LF-101")
def test_rms_setting_defaults_to_point_zero_one_and_removes_the_peak_field() -> None:
    """声の閾値は normalized full-scale の RMS 0.01 を既定とし、peak の設定欄を持たない。"""
    settings = lf.UtteranceBoundarySettings()
    assert settings.voice_rms_threshold == 0.01
    assert not hasattr(settings, "voice_threshold")
    assert "voice_threshold" not in inspect.signature(lf.UtteranceBoundarySettings).parameters
    assert "voice_threshold" not in inspect.signature(lf.UtteranceBoundarySession).parameters
    assert "voice_threshold" not in inspect.signature(lf.has_voice).parameters


@pytest.mark.req("SP-LF-101")
@pytest.mark.parametrize("threshold", [0.0, 0.1, 500, None])
def test_peak_threshold_is_rejected_with_rms_migration_guidance(threshold: object) -> None:
    """旧設定は値によらず断り、RMS の新しい名前と単位を案内する。"""
    for constructor, arguments in [
        (lf.UtteranceBoundarySettings, {}),
        (lf.UtteranceBoundarySession, {"silence_threshold_s": 0.8}),
        (lf.has_voice, {"chunk": b"", "input_format": FORMATS[0]}),
    ]:
        with pytest.raises(TypeError, match="voice_rms_threshold.*normalized full-scale.*RMS"):
            constructor(**arguments, voice_threshold=threshold)


@pytest.mark.req("SP-LF-101")
@pytest.mark.parametrize("constructor", [lf.UtteranceBoundarySettings, lf.UtteranceBoundarySession])
def test_positional_peak_threshold_cannot_be_reinterpreted_as_rms(constructor: type) -> None:
    """旧第二位置引数は RMS に読み替えず、新閾値を keyword で明示するよう案内する。"""
    with pytest.raises(TypeError, match="voice_rms_threshold.*keyword.*normalized full-scale.*RMS"):
        constructor(0.8, 0.1)
    assert (
        inspect.signature(constructor).parameters["voice_rms_threshold"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert constructor(0.8, voice_rms_threshold=0.01).voice_rms_threshold == 0.01


@pytest.mark.req("SP-LF-098")
@pytest.mark.req("SP-LF-101")
@pytest.mark.parametrize("input_format", FORMATS)
async def test_single_utterance_ignores_the_rms_threshold(
    input_format: lf.EngineInputFormat,
) -> None:
    """一発話の指定は RMS 閾値にも声の最短時間にも達しない音を一回で全量渡す。"""
    pcm = _pcm(0.001, 17, input_format)
    received: list[lf.ClosedUtterance] = []

    async def decode(utterance: lf.ClosedUtterance) -> list[lf.FinalObservation]:
        received.append(utterance)
        return [lf.FinalObservation("quiet", 0.0)]

    pipeline = lf.UtteranceDecodePipeline(
        boundary_settings=lf.UtteranceBoundarySettings(voice_rms_threshold=1.0, min_voiced_s=0.2),
        input_format=input_format,
        decode_utterance=decode,
    )
    events = [
        event
        async for event in pipeline.stream(_source([pcm]), segmentation=lf.Segmentation.SINGLE)
    ]
    assert received == [lf.ClosedUtterance(1, pcm, 0.0, 17 / 16000)]
    assert events == [
        lf.UtteranceStart(1, 0.0),
        lf.Partial(1, 0.0, "quiet"),
        lf.Final(1, 0.0, "quiet"),
        lf.UtteranceEnd(1, 17 / 16000),
    ]
