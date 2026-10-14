from __future__ import annotations

import inspect
import struct
from collections.abc import AsyncIterator

import pytest

import listenflow as lf


class _Engine(lf.STTEngineBase):
    def __init__(
        self,
        events: list[lf.TranscriptEvent],
        *,
        input_format: lf.EngineInputFormat = lf.EngineInputFormat.PCM_S16LE,
        error: lf.STTError | None = None,
    ) -> None:
        super().__init__()
        self.events = events
        self.input_format = input_format
        self.error = error
        self.received_pcm: list[bytes] = []
        self.preparations = 0
        self.closures = 0

    @property
    def capabilities(self) -> lf.EngineCapabilities:
        return lf.EngineCapabilities(
            transport=lf.EngineTransport.IN_PROCESS,
            input_format=self.input_format,
            partial_mode=lf.PartialMode.REPLACE,
            finalization=frozenset({lf.FinalizationMode.LOCAL_VAD}),
            timestamp_detail=lf.TimestampDetail.SEGMENT,
            diarization=lf.DiarizationMode.SEGMENT_LABEL,
            languages=frozenset(),
            auto_language=False,
        )

    async def prepare(self) -> None:
        self.preparations += 1
        await super().prepare()

    async def aclose(self) -> None:
        self.closures += 1
        await super().aclose()

    async def _load(self) -> None:
        pass

    async def _release(self) -> None:
        pass

    async def _recognize(
        self, pcm_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[lf.TranscriptEvent]:
        async for chunk in pcm_chunks:
            self.received_pcm.append(chunk)
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error


@pytest.mark.req("SP-LF-070")
def test_transcribe_is_a_core_async_function_without_new_engine_operations() -> None:
    """一括の文字起こしは core の非同期関数として公開され、engine の操作は変わらない。"""
    assert "transcribe" in lf.__all__
    assert inspect.iscoroutinefunction(lf.transcribe)
    assert {name for name in vars(lf.STTEngineBase) if not name.startswith("_")} == {
        "capabilities",
        "prepare",
        "stream",
        "aclose",
        "health",
    }


@pytest.mark.req("SP-LF-069")
@pytest.mark.req("SP-LF-070")
@pytest.mark.parametrize(
    ("input_format", "pcm"),
    [
        (lf.EngineInputFormat.PCM_S16LE, struct.pack("<hhhh", 600, 0, 900, 0)),
        (lf.EngineInputFormat.PCM_F32, struct.pack("<ffff", 0.2, 0.0, 0.4, 0.0)),
    ],
)
async def test_transcribe_returns_only_original_finals_in_utterance_order(
    input_format: lf.EngineInputFormat, pcm: bytes
) -> None:
    """PCM 全体の確定した文字起こしは番号・時刻・話者を保ち、発話順に返る。"""
    finals = [
        lf.Final(4, 0.1, "hello", speaker="alice"),
        lf.Final(4, 0.2, "hello", speaker=2),
        lf.Final(9, 0.5, "world"),
    ]
    engine = _Engine(
        [
            lf.UtteranceStart(4, 0.0),
            lf.Partial(4, 0.05, "hel", speaker="alice"),
            finals[0],
            finals[1],
            lf.UtteranceEnd(4, 0.3),
            lf.UtteranceStart(9, 0.4),
            finals[2],
            lf.UtteranceEnd(9, 0.6),
        ],
        input_format=input_format,
    )
    async with engine:
        result = await lf.transcribe(engine, pcm)

        assert isinstance(result, list)
        assert result == finals
        assert all(actual is expected for actual, expected in zip(result, finals, strict=True))
        assert b"".join(engine.received_pcm) == pcm
        assert engine.preparations == 1
        assert engine.closures == 0
        assert engine.health().state == lf.EngineState.READY
        assert not engine.health().busy


@pytest.mark.req("SP-LF-069")
@pytest.mark.parametrize(
    ("pcm", "events"),
    [
        (b"", []),
        (
            b"\x00\x00" * 160,
            [lf.UtteranceStart(1, 0.0), lf.Partial(1, 0.005, "maybe"), lf.UtteranceEnd(1, 0.01)],
        ),
    ],
    ids=["empty_input", "no_final_text"],
)
async def test_transcribe_returns_an_empty_list_without_final_text(
    pcm: bytes, events: list[lf.TranscriptEvent]
) -> None:
    """確定した文字起こしがない入力には空の list が返る。"""
    engine = _Engine(events)
    async with engine:
        assert await lf.transcribe(engine, pcm) == []
        assert b"".join(engine.received_pcm) == pcm


@pytest.mark.req("SP-LF-058")
@pytest.mark.parametrize("error_type", [lf.InferenceError, lf.ConfigurationError])
async def test_transcribe_propagates_the_original_stream_error(
    error_type: type[lf.STTError],
) -> None:
    """認識が途中で失敗すると、一括の呼び出しは同じ error と原因を送出する。"""
    cause = RuntimeError("upstream failure")
    error = error_type("recognition failed")
    error.__cause__ = cause
    engine = _Engine([lf.Final(1, 0.1, "before failure")], error=error)
    async with engine:
        with pytest.raises(error_type) as caught:
            await lf.transcribe(engine, b"\x00\x00")

        assert caught.value is error
        assert caught.value.__cause__ is cause
        assert engine.closures == 0
        assert not engine.health().busy


@pytest.mark.req("SP-LF-070")
async def test_transcribe_leaves_preparation_and_closure_to_the_caller() -> None:
    """用意されていない engine の一括認識は失敗し、engine の用意や終了を行わない。"""
    engine = _Engine([])
    try:
        with pytest.raises(lf.EngineStateError, match="prepare"):
            await lf.transcribe(engine, b"\x00\x00")

        assert engine.preparations == 0
        assert engine.closures == 0
        assert engine.health().state == lf.EngineState.UNPREPARED
    finally:
        await engine.aclose()


@pytest.mark.req("SP-LF-069")
@pytest.mark.req("SP-LF-070")
def test_run_sync_transcribe_returns_the_same_final_events() -> None:
    """同期の code は run_sync() で確定した文字起こしを順番に取得できる。"""
    finals = [lf.Final(3, 0.1, "first", speaker=1), lf.Final(8, 0.4, "second", speaker="bob")]
    engine = _Engine(finals)
    pcm = b"\x00\x00" * 320
    try:
        lf.run_sync(engine.prepare())
        result = lf.run_sync(lf.transcribe(engine, pcm))

        assert result == finals
        assert all(actual is expected for actual, expected in zip(result, finals, strict=True))
        assert b"".join(engine.received_pcm) == pcm
        assert engine.preparations == 1
        assert engine.closures == 0
        assert engine.health().state == lf.EngineState.READY
    finally:
        lf.run_sync(engine.aclose())
