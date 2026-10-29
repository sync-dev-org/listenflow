from __future__ import annotations

import asyncio
import inspect
import struct
from collections.abc import AsyncIterator, Mapping
from dataclasses import FrozenInstanceError, asdict, replace
from enum import StrEnum

import pytest

import listenflow as lf


def _capabilities(**overrides: object) -> lf.EngineCapabilities:
    values = dict(
        transport=lf.EngineTransport.IN_PROCESS,
        input_format=lf.EngineInputFormat.PCM_S16LE,
        partial_mode=lf.PartialMode.NONE,
        finalization=frozenset({lf.FinalizationMode.LOCAL_VAD}),
        timestamp_detail=lf.TimestampDetail.SEGMENT,
        diarization=lf.DiarizationMode.NONE,
        languages=frozenset(),
        auto_language=False,
    )
    values.update(overrides)
    return lf.EngineCapabilities(**values)


def _pcm(value: int, samples: int, input_format: lf.EngineInputFormat) -> bytes:
    sample = (
        struct.pack("<h", value)
        if input_format is lf.EngineInputFormat.PCM_S16LE
        else struct.pack("<f", value / 32768)
    )
    return sample * samples


async def _chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


class _Engine(lf.STTEngineBase):
    def __init__(
        self,
        *,
        input_format: lf.EngineInputFormat = lf.EngineInputFormat.PCM_S16LE,
        min_voiced_s: float = 0.0,
        no_text: bool = False,
    ) -> None:
        super().__init__()
        self.input_format = input_format
        self.no_text = no_text
        self.decoded: list[lf.ClosedUtterance] = []
        self.block_decode = False
        self.decode_started = asyncio.Event()
        self.decode_cancelled = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.decode_finished = asyncio.Event()
        self.pipeline = lf.UtteranceDecodePipeline(
            boundary_settings=lf.UtteranceBoundarySettings(
                silence_threshold_s=0.2,
                preroll_s=0.1,
                min_voiced_s=min_voiced_s,
            ),
            input_format=input_format,
            decode_utterance=self._decode,
            on_progress=self._mark_progress,
        )

    @property
    def capabilities(self) -> lf.EngineCapabilities:
        return _capabilities(input_format=self.input_format, single_utterance=True)

    async def _load(self) -> None:
        pass

    async def _release(self) -> None:
        pass

    async def _decode(self, utterance: lf.ClosedUtterance) -> list[lf.FinalObservation]:
        self.decoded.append(utterance)
        self.decode_started.set()
        if self.block_decode:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.decode_cancelled.set()
                await self.allow_cleanup.wait()
                raise
            finally:
                self.decode_finished.set()
        if self.no_text:
            return []
        return [
            lf.FinalObservation(
                f"request-{len(self.decoded)}", utterance.start_offset_s + 0.025, speaker="a"
            )
        ]

    async def _recognize(
        self, pcm_chunks: AsyncIterator[bytes], **options: object
    ) -> AsyncIterator[lf.TranscriptEvent]:
        iterator = self.pipeline.stream(pcm_chunks, **options)
        try:
            async for event in iterator:
                yield event
        finally:
            await iterator.aclose()

    def _health_details(self) -> Mapping[str, object]:
        return {
            name: getattr(self.pipeline, name)
            for name in (
                "processed_pcm_chunks",
                "emitted_events",
                "decoding_utterance_id",
                "pending_decode_utterances",
            )
        }


@pytest.mark.req("SP-LF-098")
@pytest.mark.req("SP-LF-070")
def test_single_utterance_function_and_segmentation_are_public() -> None:
    """利用側は core の非同期関数と明示的な区切りの指定を公開名で使える。"""
    assert {"transcribe_utterance", "Segmentation"} <= set(lf.__all__)
    assert inspect.iscoroutinefunction(lf.transcribe_utterance)
    assert issubclass(lf.Segmentation, StrEnum)
    assert {item.name: item.value for item in lf.Segmentation} == {
        "AUTO": "auto",
        "SINGLE": "single",
    }
    for method in (lf.STTEngineBase.stream, lf.UtteranceDecodePipeline.stream):
        option = inspect.signature(method).parameters["segmentation"]
        assert option.kind is inspect.Parameter.KEYWORD_ONLY
        assert option.default is lf.Segmentation.AUTO


@pytest.mark.req("SP-LF-056")
def test_single_utterance_capability_is_optional_boolean_and_immutable() -> None:
    """engine は一発話への対応を bool で申告し、省略した engine は非対応になる。"""
    profile = _capabilities()
    assert profile.single_utterance is False
    assert replace(profile, single_utterance=True).single_utterance is True
    with pytest.raises(FrozenInstanceError):
        profile.single_utterance = True
    for invalid in (None, 0, 1, "true"):
        with pytest.raises(TypeError, match="single_utterance"):
            replace(profile, single_utterance=invalid)


@pytest.mark.req("SP-LF-097")
@pytest.mark.req("SP-LF-098")
@pytest.mark.parametrize(
    "input_format", [lf.EngineInputFormat.PCM_S16LE, lf.EngineInputFormat.PCM_F32]
)
@pytest.mark.parametrize("entry", ["function", "stream", "pipeline"])
async def test_single_utterance_decodes_all_samples_once_regardless_of_chunks(
    input_format: lf.EngineInputFormat, entry: str
) -> None:
    """一発話の指定は長い無音と前後の音を含め、chunk の分け方によらず全体を一回渡す。"""
    chunks = [
        _pcm(value, 1600, input_format)
        for value in (0, 1000, 0, 0, 0, 2000, 0, 0, 0, 3000, 0, 0, 0)
    ]
    pcm = b"".join(chunks)
    width = 2 if input_format is lf.EngineInputFormat.PCM_S16LE else 4
    irregular = [pcm[:width], b"", pcm[width : 3521 * width], pcm[3521 * width :]]
    for layout in ([pcm], chunks, irregular):
        async with _Engine(input_format=input_format) as engine:
            if entry == "function":
                events = await lf.transcribe_utterance(engine, b"".join(layout))
            else:
                target = engine if entry == "stream" else engine.pipeline
                events = [
                    event
                    async for event in target.stream(
                        _chunks(layout), segmentation=lf.Segmentation.SINGLE
                    )
                ]
                assert events == [
                    lf.UtteranceStart(1, 0.0),
                    lf.Partial(1, 0.025, "request-1", "a"),
                    lf.Final(1, 0.025, "request-1", "a"),
                    lf.UtteranceEnd(1, 1.3),
                ]
            assert [event for event in events if isinstance(event, lf.Final)] == [
                lf.Final(1, 0.025, "request-1", "a")
            ]
            assert engine.decoded == [lf.ClosedUtterance(1, pcm, 0.0, 1.3)]


@pytest.mark.req("SP-LF-097")
async def test_single_utterance_waits_for_end_of_input_before_decode() -> None:
    """一発話の指定は途中の無音で decode せず、入力の終わりまで全体を受け取る。"""
    voice = _pcm(1000, 1600, lf.EngineInputFormat.PCM_S16LE)
    silence = _pcm(0, 6400, lf.EngineInputFormat.PCM_S16LE)
    input_paused = asyncio.Event()
    finish_input = asyncio.Event()

    async def source():
        yield voice
        yield silence
        input_paused.set()
        await finish_input.wait()
        yield voice

    async with _Engine() as engine:
        stream = engine.stream(source(), segmentation=lf.Segmentation.SINGLE)
        try:
            assert await asyncio.wait_for(anext(stream), 1) == lf.UtteranceStart(1, 0.0)
            await asyncio.wait_for(input_paused.wait(), 1)
            assert engine.decoded == []
            assert engine.pipeline.processed_pcm_chunks == 2
            finish_input.set()
            events = [event async for event in stream]
            assert [event for event in events if isinstance(event, lf.Final)] == [
                lf.Final(1, 0.025, "request-1", "a")
            ]
            assert events[-1] == lf.UtteranceEnd(1, 0.6)
            assert engine.decoded == [lf.ClosedUtterance(1, voice + silence + voice, 0.0, 0.6)]
        finally:
            finish_input.set()
            await stream.aclose()


@pytest.mark.req("SP-LF-058")
async def test_single_utterance_propagates_decode_failure_and_allows_reuse() -> None:
    """一発話の文字起こしの失敗は元の error と原因を返し、後の呼び出しを妨げない。"""
    error = lf.InferenceError("decode failed")
    error.__cause__ = RuntimeError("native failure")

    class FailingEngine(_Engine):
        async def _decode(self, utterance):
            if not self.decoded:
                self.decoded.append(utterance)
                raise error
            return await super()._decode(utterance)

    async with FailingEngine() as engine:
        with pytest.raises(lf.InferenceError) as caught:
            await lf.transcribe_utterance(engine, b"\0\0" * 1600)
        assert caught.value is error
        assert caught.value.__cause__ is error.__cause__
        assert not engine.health().busy
        assert engine.health().pending_decode_utterances == 0
        assert await lf.transcribe_utterance(engine, b"\0\0" * 1600) == [
            lf.Final(1, 0.025, "request-2", "a")
        ]


@pytest.mark.req("SP-LF-096")
@pytest.mark.req("SP-LF-097")
@pytest.mark.req("SP-LF-006")
@pytest.mark.req("SP-LF-029")
@pytest.mark.parametrize("cancellations", [3, 6])
async def test_retained_audio_is_transcribed_without_gaps_or_overlap_and_keeps_prior_finals(
    cancellations: int,
) -> None:
    """応答の準備の取り消しが重なっても保持した音は一回で渡り、先の確定文字と時刻は変わらない。"""
    input_format = lf.EngineInputFormat.PCM_S16LE
    source = _pcm(77, 1600, input_format)
    # The application retains one contiguous range, including every silence gap.
    for number in range(cancellations + 1):
        source += _pcm(1000 + number, 1600, input_format) + _pcm(0, 6400, input_format)
    retained_start = 1600 * 2
    previous: list[tuple[lf.Final, dict]] = []
    async with _Engine() as engine:
        for number in range(cancellations + 1):
            end = retained_start + (number + 1) * 8000 * 2
            retained = source[retained_start:end]
            results = await lf.transcribe_utterance(engine, retained)
            assert results == [lf.Final(1, 0.025, f"request-{number + 1}", "a")]
            assert len(engine.decoded) == number + 1
            assert engine.decoded[-1] == lf.ClosedUtterance(1, retained, 0.0, (number + 1) * 0.5)
            assert retained_start / 2 / 16000 + results[0].ts_offset_s == pytest.approx(0.125)
            for final, snapshot in previous:
                assert asdict(final) == snapshot
                assert results[0] is not final
            previous.append((results[0], asdict(results[0])))


@pytest.mark.req("SP-LF-094")
@pytest.mark.req("SP-LF-098")
@pytest.mark.req("SP-LF-100")
async def test_single_utterance_bypasses_voice_detection_and_minimum_voiced_span() -> None:
    """一発話の指定は短い声と声の閾値未満の音も回し、AUTO は短い声を回さず境界で閉じる。"""
    voice = _pcm(1000, 1600, lf.EngineInputFormat.PCM_S16LE)
    silence = _pcm(0, 4800, lf.EngineInputFormat.PCM_S16LE)
    quiet = _pcm(1, 1600, lf.EngineInputFormat.PCM_S16LE)
    async with _Engine(min_voiced_s=0.2) as engine:
        events = [event async for event in engine.stream(_chunks([voice, silence, voice, silence]))]
        assert events == [
            lf.UtteranceStart(1, 0.0),
            lf.UtteranceEnd(1, 0.32),
            lf.UtteranceStart(2, 0.32),
            lf.UtteranceEnd(2, 0.72),
        ]
        assert engine.decoded == []
        assert await lf.transcribe_utterance(engine, voice) == [
            lf.Final(1, 0.025, "request-1", "a")
        ]
        assert await lf.transcribe_utterance(engine, quiet) == [
            lf.Final(1, 0.025, "request-2", "a")
        ]
        assert [utterance.pcm for utterance in engine.decoded] == [voice, quiet]


@pytest.mark.req("SP-LF-097")
@pytest.mark.req("SP-LF-058")
@pytest.mark.parametrize("entry", ["function", "stream", "pipeline"])
@pytest.mark.parametrize("pcm", [b"", b"\0\0" * 1600], ids=["empty", "no-text"])
async def test_single_utterance_empty_input_and_no_text_are_normal_results(
    entry: str, pcm: bytes
) -> None:
    """長さゼロの音は decode せず、文字にならない音は始まりと終わりの対だけを返す。"""
    async with _Engine(no_text=True) as engine:
        if entry == "function":
            assert await lf.transcribe_utterance(engine, pcm) == []
        else:
            target = engine if entry == "stream" else engine.pipeline
            events = [
                event
                async for event in target.stream(
                    _chunks([b"", pcm, b""]), segmentation=lf.Segmentation.SINGLE
                )
            ]
            assert events == ([lf.UtteranceStart(1, 0.0), lf.UtteranceEnd(1, 0.1)] if pcm else [])
        assert engine.decoded == ([lf.ClosedUtterance(1, pcm, 0.0, 0.1)] if pcm else [])


@pytest.mark.req("SP-LF-057")
@pytest.mark.parametrize(
    "input_format", [lf.EngineInputFormat.PCM_S16LE, lf.EngineInputFormat.PCM_F32]
)
@pytest.mark.parametrize("entry", ["function", "stream", "pipeline"])
async def test_single_utterance_rejects_incomplete_samples_before_decode_and_allows_reuse(
    input_format: lf.EngineInputFormat, entry: str
) -> None:
    """sample の幅を欠く音は文字起こし前に拒否され、同じ engine と pipeline を再利用できる。"""
    async with _Engine(input_format=input_format) as engine:
        error = ValueError if entry == "pipeline" else lf.ConfigurationError
        with pytest.raises(error, match="whole samples"):
            if entry == "function":
                await lf.transcribe_utterance(engine, b"\0")
            else:
                target = engine if entry == "stream" else engine.pipeline
                async for _ in target.stream(
                    _chunks([_pcm(1000, 1600, input_format), b"\0"]),
                    segmentation=lf.Segmentation.SINGLE,
                ):
                    pass
        assert engine.decoded == []
        assert not engine.health().busy
        assert await lf.transcribe_utterance(engine, _pcm(1000, 1600, input_format))


@pytest.mark.req("SP-LF-056")
@pytest.mark.req("SP-LF-057")
@pytest.mark.parametrize("entry", ["function", "stream"])
async def test_single_utterance_requires_declared_support_before_consuming_input(
    entry: str,
) -> None:
    """非対応の engine は入力とモデルに触れず、一発話の指定を設定の誤りとして断る。"""

    class LegacyEngine(_Engine):
        @property
        def capabilities(self) -> lf.EngineCapabilities:
            return _capabilities()

        async def _recognize(self, pcm_chunks):
            raise AssertionError("unsupported recognition hook was called")
            yield

    consumed = False

    async def source():
        nonlocal consumed
        consumed = True
        yield b"\0\0"

    async with LegacyEngine() as engine:
        with pytest.raises(lf.ConfigurationError, match="single_utterance"):
            if entry == "function":
                await lf.transcribe_utterance(engine, b"\0\0")
            else:
                async for _ in engine.stream(source(), segmentation=lf.Segmentation.SINGLE):
                    pass
        assert not consumed
        assert engine.decoded == []
        assert not engine.health().busy


@pytest.mark.req("SP-LF-057")
@pytest.mark.parametrize("invalid", [None, "single", "unknown", 1])
@pytest.mark.parametrize("entry", ["engine", "pipeline"])
async def test_invalid_segmentation_is_rejected_before_decode(invalid: object, entry: str) -> None:
    """区切りの指定に enum 以外の値を渡すと、文字起こしを行う前に拒否する。"""
    async with _Engine() as engine:
        target = engine if entry == "engine" else engine.pipeline
        error = lf.ConfigurationError if entry == "engine" else ValueError
        with pytest.raises(error, match="segmentation"):
            async for _ in target.stream(_chunks([b"\0\0"]), segmentation=invalid):
                pass
        assert engine.decoded == []
        assert not engine.health().busy


@pytest.mark.req("SP-LF-054")
async def test_single_utterance_rejects_overlap_without_disturbing_auto_stream() -> None:
    """通常の stream の動作中の一発話の依頼は、先行する認識を壊さずに断られる。"""
    finish_input = asyncio.Event()
    voice = _pcm(1000, 1600, lf.EngineInputFormat.PCM_S16LE)

    async def source():
        yield voice
        await finish_input.wait()

    async with _Engine() as engine:
        stream = engine.stream(source())
        try:
            assert await asyncio.wait_for(anext(stream), 1) == lf.UtteranceStart(1, 0.0)
            with pytest.raises(lf.EngineStateError, match="active stream"):
                await lf.transcribe_utterance(engine, voice)
            assert engine.health().busy
            finish_input.set()
            events = [event async for event in stream]
            assert events == [
                lf.Partial(1, 0.025, "request-1", "a"),
                lf.Final(1, 0.025, "request-1", "a"),
                lf.UtteranceEnd(1, 0.1),
            ]
            assert engine.decoded == [lf.ClosedUtterance(1, voice, 0.0, 0.1)]
        finally:
            await stream.aclose()


@pytest.mark.req("SP-LF-055")
@pytest.mark.req("SP-LF-054")
@pytest.mark.parametrize("ending", ["close", "cancel"])
async def test_single_utterance_termination_waits_for_decode_cleanup_and_allows_reuse(
    ending: str,
) -> None:
    """一発話の途中終了と取り消しは推論へ届き、後始末の完了までは次の認識を断る。"""
    voice = _pcm(1000, 1600, lf.EngineInputFormat.PCM_S16LE)
    async with _Engine() as engine:
        engine.block_decode = True
        stream = engine.stream(_chunks([voice]), segmentation=lf.Segmentation.SINGLE)
        termination = None
        try:
            assert await asyncio.wait_for(anext(stream), 1) == lf.UtteranceStart(1, 0.0)
            await asyncio.wait_for(engine.decode_started.wait(), 1)
            if ending == "close":
                termination = asyncio.create_task(stream.aclose())
            else:
                termination = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                termination.cancel()
            await asyncio.wait_for(engine.decode_cancelled.wait(), 1)
            assert not termination.done()
            assert engine.health().busy
            assert engine.health().decoding_utterance_id == 1
            with pytest.raises(lf.EngineStateError, match="active stream"):
                await lf.transcribe_utterance(engine, voice)
            engine.allow_cleanup.set()
            if ending == "close":
                await asyncio.wait_for(termination, 1)
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(termination, 1)
            assert engine.decode_finished.is_set()
            assert not engine.health().busy
            assert engine.health().decoding_utterance_id is None
            assert engine.health().pending_decode_utterances == 0
            engine.block_decode = False
            assert await lf.transcribe_utterance(engine, voice) == [
                lf.Final(1, 0.025, "request-2", "a")
            ]
        finally:
            engine.allow_cleanup.set()
            await stream.aclose()
            if termination is not None:
                await asyncio.gather(termination, return_exceptions=True)
