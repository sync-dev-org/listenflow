from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError

import pytest

import listenflow as lf


def _capabilities(**overrides: object) -> lf.EngineCapabilities:
    return lf.EngineCapabilities(
        **dict(
            transport=lf.EngineTransport.IN_PROCESS,
            input_format=lf.EngineInputFormat.PCM_S16LE,
            partial_mode=lf.PartialMode.NONE,
            finalization=frozenset({lf.FinalizationMode.LOCAL_VAD}),
            timestamp_detail=lf.TimestampDetail.SEGMENT,
            diarization=lf.DiarizationMode.NONE,
            languages=frozenset({"en", "ja"}),
            auto_language=True,
        )
        | overrides
    )


async def _pcm(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


class _Engine(lf.STTEngineBase):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = asyncio.Event()
        self.load_gate = asyncio.Event()
        self.load_gate.set()
        self.load_cancelled = asyncio.Event()
        self.load_finishing = asyncio.Event()
        self.load_finishing.set()
        self.load_survives_cancel = False
        self.load_failure: BaseException | None = None
        self.release_failure: BaseException | None = None
        self.release_started = asyncio.Event()
        self.release_gate = asyncio.Event()
        self.release_gate.set()
        self.recognizing = asyncio.Event()
        self.recognition_gate = asyncio.Event()
        self.recognition_cancelled = asyncio.Event()
        self.recognition_finishing = asyncio.Event()
        self.recognition_finishing.set()
        self.loads = 0
        self.releases = 0
        self.recognitions = 0
        self.processed = 0
        self.events = 0
        self.log: list[str] = []

    @property
    def capabilities(self) -> lf.EngineCapabilities:
        return _capabilities()

    async def _load(self) -> None:
        self.loads += 1
        self.loaded.set()
        try:
            await self.load_gate.wait()
        except asyncio.CancelledError:
            self.load_cancelled.set()
            await self.load_finishing.wait()
            if not self.load_survives_cancel:
                raise
        if self.load_failure is not None:
            raise self.load_failure
        self.log.append("loaded")

    async def _release(self) -> None:
        self.releases += 1
        self.release_started.set()
        await self.release_gate.wait()
        self.log.append("released")
        if self.release_failure is not None:
            raise self.release_failure

    async def _recognize(
        self, pcm_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[lf.TranscriptEvent]:
        self.recognitions += 1
        self.recognizing.set()
        try:
            async for chunk in pcm_chunks:
                del chunk
                self.processed += 1
                self._mark_progress()
            self.events += 1
            yield lf.UtteranceStart(1, 0.0)
            await self.recognition_gate.wait()
            self._mark_progress()
            self.events += 1
            yield lf.UtteranceEnd(1, 0.1)
        except asyncio.CancelledError:
            self.recognition_cancelled.set()
            raise
        finally:
            await self.recognition_finishing.wait()
            self.log.append("recognition_finished")

    def _health_details(self) -> dict[str, object]:
        return {"processed_pcm_chunks": self.processed, "emitted_events": self.events}


@pytest.mark.req("SP-LF-057")
@pytest.mark.parametrize(
    ("name", "standard"),
    [
        ("ConfigurationError", ValueError),
        ("EngineStateError", RuntimeError),
        ("EngineClosedError", RuntimeError),
        ("InferenceError", RuntimeError),
        ("EnvironmentUnavailableError", RuntimeError),
        ("ModuleNotInstalledError", ImportError),
    ],
)
def test_errors_support_library_and_standard_exception_handlers(name: str, standard: type) -> None:
    """六種の error は STTError と意味に合う Python の標準の error を兼ねる。

    種類は設定、状態、閉じられたことによる取り消し、実行時の失敗、環境の不足、module の不足。
    """
    error = getattr(lf, name)("failure")
    assert isinstance(error, lf.STTError)
    assert isinstance(error, standard)
    assert name in lf.__all__
    if name == "EngineClosedError":
        assert isinstance(error, lf.EngineStateError)


@pytest.mark.req("SP-LF-056")
def test_capabilities_declare_languages_and_keep_recognition_fields_immutable() -> None:
    """capabilities は言語の指定の仕方と認識の項目を変更できない静的な申告値として持つ。"""
    languages = {"en", "ja"}
    profile = _capabilities(languages=languages)
    languages.add("fr")
    assert profile.languages == frozenset({"en", "ja"})
    assert type(profile.languages) is frozenset
    assert profile.auto_language is True
    assert profile.transport == lf.EngineTransport.IN_PROCESS
    assert profile.input_format == lf.EngineInputFormat.PCM_S16LE
    assert profile.partial_mode == lf.PartialMode.NONE
    assert profile.finalization == frozenset({lf.FinalizationMode.LOCAL_VAD})
    assert profile.timestamp_detail == lf.TimestampDetail.SEGMENT
    assert profile.diarization == lf.DiarizationMode.NONE
    with pytest.raises(FrozenInstanceError):
        profile.auto_language = False
    with pytest.raises(AttributeError):
        profile.languages.add("de")
    assert not hasattr(profile, "language_control")
    assert not hasattr(lf, "LanguageControl")


@pytest.mark.req("SP-LF-056")
@pytest.mark.parametrize("overrides", [{"languages": {1}}, {"auto_language": 1}])
def test_capability_construction_keeps_standard_validation_errors(overrides: dict) -> None:
    """能力の値の構築は言語と自動判定の型が誤っている時に標準の型の誤りを返す。"""
    with pytest.raises(TypeError, match=next(iter(overrides))):
        _capabilities(**overrides)


@pytest.mark.req("SP-LF-052")
async def test_prepare_loads_once_after_success_and_retries_after_failure() -> None:
    """prepare() は環境の不足を失敗で知らせ、成功後の二度目は読み込みを繰り返さない。

    利用側は、失敗の後に用意前の状態から prepare() をやり直せる。
    """
    engine = _Engine()
    failure = lf.EnvironmentUnavailableError("device unavailable")
    engine.load_failure = failure
    with pytest.raises(lf.EnvironmentUnavailableError) as caught:
        await engine.prepare()
    assert caught.value is failure
    assert engine.health().state == lf.EngineState.UNPREPARED
    assert not engine.health().busy
    engine.load_failure = None
    await engine.prepare()
    await engine.prepare()
    assert engine.loads == 2
    assert engine.health().state == lf.EngineState.READY
    await engine.aclose()
    assert engine.releases == 1

    failed_engine = _Engine()
    failed_engine.load_failure = failure
    with pytest.raises(lf.EnvironmentUnavailableError):
        await failed_engine.prepare()
    await failed_engine.aclose()
    assert failed_engine.releases == 0


@pytest.mark.req("SP-LF-052")
async def test_prepared_engine_prepares_again_without_waiting_for_an_active_stream() -> None:
    """用意済みの engine は認識の stream が終わるのを待たずに二度目の用意を終える。"""
    engine = _Engine()
    await engine.prepare()
    stream = engine.stream(_pcm())
    try:
        assert await asyncio.wait_for(anext(stream), timeout=0.5) == lf.UtteranceStart(1, 0.0)
        await asyncio.wait_for(engine.prepare(), timeout=0.5)
        assert engine.loads == 1
        assert engine.recognitions == 1
        assert engine.health().busy
        engine.recognition_gate.set()
        assert await asyncio.wait_for(anext(stream), timeout=0.5) == lf.UtteranceEnd(1, 0.1)
    finally:
        await asyncio.wait_for(stream.aclose(), timeout=0.5)
        await asyncio.wait_for(engine.aclose(), timeout=0.5)


@pytest.mark.req("SP-LF-057")
async def test_stream_requires_successful_preparation_and_starts_on_first_read() -> None:
    """音声認識は最初の読み取りで始まり、用意の前には状態の誤りを返す。"""
    engine = _Engine()
    stream = engine.stream(_pcm())
    assert engine.recognitions == 0
    with pytest.raises(lf.EngineStateError):
        await anext(stream)
    assert engine.recognitions == 0
    await engine.aclose()
    assert engine.releases == 0


@pytest.mark.req("SP-LF-054")
async def test_operations_wait_in_arrival_order_and_cancelled_waiters_do_not_interrupt_loading() -> (
    None
):
    """後続の操作は来た順に待ち、待機の取り消しは先行する読み込みを壊さない。"""
    engine = _Engine()
    engine.load_gate.clear()
    first = asyncio.create_task(engine.prepare())
    await engine.loaded.wait()
    cancelled = asyncio.create_task(engine.prepare())
    await asyncio.sleep(0)
    stream = engine.stream(_pcm())
    read = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    later = asyncio.create_task(engine.prepare())
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not engine.load_cancelled.is_set()
    engine.load_gate.set()
    await first
    assert await read == lf.UtteranceStart(1, 0.0)
    assert not later.done()
    await stream.aclose()
    await later
    assert engine.loads == 1
    await engine.aclose()


@pytest.mark.req("SP-LF-054")
async def test_operations_arriving_together_keep_their_order_behind_loading() -> None:
    """読み込みの後ろへ続けて来た操作でも後続の用意は先行する認識を追い越さない。"""
    engine = _Engine()
    engine.load_gate.clear()
    loading = asyncio.create_task(engine.prepare())
    await engine.loaded.wait()
    stream = engine.stream(_pcm())
    first = asyncio.create_task(anext(stream))
    later = asyncio.create_task(engine.prepare())
    await asyncio.sleep(0)
    engine.load_gate.set()
    await loading
    assert await first == lf.UtteranceStart(1, 0.0)
    try:
        assert not later.done()
    finally:
        await stream.aclose()
        await later
        await engine.aclose()


@pytest.mark.req("SP-LF-054")
async def test_streams_waiting_behind_loading_reject_overlap_when_the_first_starts() -> None:
    """読み込みの後ろで待つ二本の stream は先頭だけを始め、後続を状態の誤りで断る。"""
    engine = _Engine()
    engine.load_gate.clear()
    loading = asyncio.create_task(engine.prepare())
    await engine.loaded.wait()
    first = engine.stream(_pcm())
    second = engine.stream(_pcm())
    read_first = asyncio.create_task(anext(first))
    await asyncio.sleep(0)
    read_second = asyncio.create_task(anext(second))
    await asyncio.sleep(0)
    engine.load_gate.set()
    await loading
    assert await read_first == lf.UtteranceStart(1, 0.0)
    with pytest.raises(lf.EngineStateError):
        await read_second
    assert engine.recognitions == 1
    await first.aclose()
    await engine.aclose()


@pytest.mark.req("SP-LF-053")
async def test_close_cancels_loading_and_waiting_preparation_before_releasing_once() -> None:
    """aclose() は読み込みと待機を取り消し、終了を待って資源を一度だけ返す。

    二度目の aclose() は何もせず正常に戻る。
    """
    engine = _Engine()
    engine.load_gate.clear()
    engine.load_finishing.clear()
    engine.load_survives_cancel = True
    preparing = asyncio.create_task(engine.prepare())
    callers = [preparing]
    try:
        await asyncio.wait_for(engine.loaded.wait(), timeout=0.5)
        waiter = asyncio.create_task(engine.prepare())
        callers.append(waiter)
        await asyncio.sleep(0)
        assert not waiter.done()
        close = asyncio.create_task(engine.aclose())
        await asyncio.wait_for(engine.load_cancelled.wait(), timeout=0.5)
        with pytest.raises(lf.EngineClosedError):
            await asyncio.wait_for(waiter, timeout=0.5)
        assert not close.done()
        assert engine.releases == 0
        engine.load_finishing.set()
        with pytest.raises(lf.EngineClosedError):
            await asyncio.wait_for(preparing, timeout=0.5)
        await asyncio.wait_for(close, timeout=0.5)
        await asyncio.wait_for(engine.aclose(), timeout=0.5)
        assert engine.log == ["loaded", "released"]
        assert engine.loads == 1
        assert engine.releases == 1
    finally:
        engine.load_gate.set()
        engine.load_finishing.set()
        await asyncio.wait_for(engine.aclose(), timeout=0.5)
        await asyncio.wait_for(asyncio.gather(*callers, return_exceptions=True), timeout=0.5)


@pytest.mark.req("SP-LF-053")
async def test_close_cancels_recognition_and_waits_for_cleanup_before_releasing_once() -> None:
    """閉じる操作は推論を取り消し、推論の後始末が終わってから一度だけ資源を返す。"""
    engine = _Engine()
    await engine.prepare()
    engine.recognition_finishing.clear()
    stream = engine.stream(_pcm())
    assert await anext(stream) == lf.UtteranceStart(1, 0.0)
    read = asyncio.create_task(anext(stream))
    try:
        await asyncio.sleep(0)
        close = asyncio.create_task(engine.aclose())
        await asyncio.wait_for(engine.recognition_cancelled.wait(), timeout=0.5)
        assert not close.done()
        assert engine.releases == 0
        engine.recognition_finishing.set()
        with pytest.raises(lf.EngineClosedError):
            await asyncio.wait_for(read, timeout=0.5)
        await asyncio.wait_for(close, timeout=0.5)
        await asyncio.wait_for(engine.aclose(), timeout=0.5)
        assert engine.log == ["loaded", "recognition_finished", "released"]
        assert engine.releases == 1
    finally:
        engine.recognition_finishing.set()
        await asyncio.wait_for(engine.aclose(), timeout=0.5)
        await asyncio.wait_for(asyncio.gather(read, return_exceptions=True), timeout=0.5)


@pytest.mark.req("SP-LF-053")
async def test_close_after_cancelled_prepare_waits_for_successful_native_load() -> None:
    """prepare() の取り消し後も aclose() は読み込みの終了を待ち、読み込みが作った資源を返す。"""
    engine = _Engine()
    engine.load_gate.clear()
    engine.load_finishing.clear()
    engine.load_survives_cancel = True
    preparing = asyncio.create_task(engine.prepare())
    await engine.loaded.wait()
    preparing.cancel()
    await engine.load_cancelled.wait()
    close = asyncio.create_task(engine.aclose())
    await asyncio.sleep(0)
    assert not preparing.done()
    assert not close.done()
    assert engine.releases == 0
    engine.load_finishing.set()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    await close
    assert engine.log == ["loaded", "released"]


@pytest.mark.req("SP-LF-053")
async def test_close_finishes_a_stream_paused_between_event_reads() -> None:
    """aclose() は読み取り待ちの stream を閉じ、閉じた engine の prepare() と stream() を断る。

    閉じた後も capabilities と health() の読み取りは値を返す。
    """
    engine = _Engine()
    await engine.prepare()
    stream = engine.stream(_pcm())
    assert await anext(stream) == lf.UtteranceStart(1, 0.0)
    await engine.aclose()
    assert engine.log == ["loaded", "recognition_finished", "released"]
    with pytest.raises(lf.EngineClosedError):
        await anext(stream)
    with pytest.raises(lf.EngineClosedError):
        await engine.prepare()
    with pytest.raises(lf.EngineClosedError):
        await anext(engine.stream(_pcm()))
    assert engine.capabilities.languages == frozenset({"en", "ja"})
    assert engine.health().state == lf.EngineState.CLOSED
    assert not engine.health().busy


@pytest.mark.req("SP-LF-053")
@pytest.mark.req("SP-LF-057")
async def test_overlapping_close_calls_wait_for_release_and_only_the_first_reports_failure() -> (
    None
):
    """重なる閉じる操作は返却を待ち、返却の失敗を最初の呼び出しだけが原因をつないで返す。"""
    engine = _Engine()
    await engine.prepare()
    original = OSError("release failed")
    engine.release_failure = original
    engine.release_gate.clear()
    first = asyncio.create_task(engine.aclose())
    await engine.release_started.wait()
    second = asyncio.create_task(engine.aclose())
    await asyncio.sleep(0)
    assert not second.done()
    engine.release_gate.set()
    with pytest.raises(lf.InferenceError) as caught:
        await first
    assert caught.value.__cause__ is original
    await second
    await engine.aclose()
    assert engine.releases == 1
    assert engine.health().state == lf.EngineState.CLOSED


@pytest.mark.req("SP-LF-053")
async def test_async_context_prepares_and_releases_after_body_failure() -> None:
    """async with は入る時に prepare()、本文が失敗して抜ける時も aclose() を呼ぶ。"""
    engine = _Engine()
    failure = LookupError("body failed")
    with pytest.raises(LookupError) as caught:
        async with engine as entered:
            assert entered is engine
            assert engine.health().state == lf.EngineState.READY
            raise failure
    assert caught.value is failure
    assert engine.log == ["loaded", "released"]


@pytest.mark.req("SP-LF-054")
@pytest.mark.req("SP-LF-055")
@pytest.mark.parametrize("finish", ["cancel_read", "close_iterator"])
async def test_consumer_termination_waits_for_cleanup_before_allowing_reuse(finish: str) -> None:
    """利用側の取り消しと stream の途中終了は推論の後始末を待ち、同じ engine を再利用できる。

    終了待ちの間も重なる stream を状態の誤りで断り、先行する stream を壊さない。
    """
    engine = _Engine()
    await engine.prepare()
    stream = engine.stream(_pcm())
    assert await anext(stream) == lf.UtteranceStart(1, 0.0)
    with pytest.raises(lf.EngineStateError):
        await anext(engine.stream(_pcm()))
    engine.recognition_finishing.clear()
    if finish == "cancel_read":
        ending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        ending.cancel()
        await engine.recognition_cancelled.wait()
    else:
        ending = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
    assert not ending.done()
    with pytest.raises(lf.EngineStateError):
        await anext(engine.stream(_pcm()))
    engine.recognition_finishing.set()
    if finish == "cancel_read":
        with pytest.raises(asyncio.CancelledError):
            await ending
    else:
        await ending
    next_stream = engine.stream(_pcm())
    assert await anext(next_stream) == lf.UtteranceStart(1, 0.0)
    await next_stream.aclose()
    await engine.aclose()


@pytest.mark.req("SP-LF-050")
@pytest.mark.req("SP-LF-054")
@pytest.mark.req("SP-LF-056")
async def test_health_reports_loading_and_progress_without_waiting_for_model_operations() -> None:
    """health() と capabilities はモデルを使う操作の動作中も待たずに読み取れる。

    health() は用意前の状態、処理中の操作の有無、最後に処理が進んだ時刻を返す。
    """
    engine = _Engine()
    initial = engine.health()
    assert initial.state == lf.EngineState.UNPREPARED
    assert not initial.busy
    assert initial.last_progress_monotonic_s is None
    assert not hasattr(initial, "model_loaded")
    assert not hasattr(initial, "last_processed_monotonic_s")
    engine.load_gate.clear()
    preparing = asyncio.create_task(engine.prepare())
    await engine.loaded.wait()
    assert engine.health().state == lf.EngineState.UNPREPARED
    assert engine.health().busy
    assert engine.capabilities.auto_language
    engine.load_gate.set()
    await preparing
    stream = engine.stream(_pcm(b"\x00\x00"))
    assert await anext(stream) == lf.UtteranceStart(1, 0.0)
    processed = engine.health()
    assert processed.busy
    assert processed.processed_pcm_chunks == 1
    assert processed.last_progress_monotonic_s is not None
    await asyncio.sleep(0)
    assert engine.health().last_progress_monotonic_s == processed.last_progress_monotonic_s
    engine.recognition_gate.set()
    assert await anext(stream) == lf.UtteranceEnd(1, 0.1)
    assert engine.health().last_progress_monotonic_s > processed.last_progress_monotonic_s
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not engine.health().busy
    await engine.aclose()


@pytest.mark.req("SP-LF-057")
async def test_engine_rejects_incomplete_pcm_samples_with_configuration_error() -> None:
    """engine の入力の境界は sample が途中で切れた PCM を設定の誤りとして断る。"""
    engine = _Engine()
    await engine.prepare()
    with pytest.raises(lf.ConfigurationError) as caught:
        await anext(engine.stream(_pcm(b"\x00")))
    assert isinstance(caught.value.__cause__, ValueError)
    assert engine.processed == 0
    await engine.aclose()


@pytest.mark.req("SP-LF-052")
def test_engine_has_no_preflight_api() -> None:
    """モデルの用意は一つの非同期の操作に集まり、事前の可用性を調べる API はない。"""
    for name in ("preflight", "PreflightCheck", "PreflightReport", "PreflightStatus"):
        assert not hasattr(lf, name)
        assert name not in lf.__all__


@pytest.mark.req("SP-LF-053")
@pytest.mark.parametrize("outcome", ["empty", "event", "failure"])
async def test_close_remains_a_closed_error_when_native_cancellation_is_handled(
    outcome: str,
) -> None:
    """閉じる操作による取り消しは native の終了方法にかかわらず閉じた状態の error で伝わる。"""

    class NativeEngine(_Engine):
        async def _recognize(
            self, pcm_chunks: AsyncIterator[bytes]
        ) -> AsyncIterator[lf.TranscriptEvent]:
            del pcm_chunks
            yield lf.UtteranceStart(1, 0.0)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if outcome == "event":
                    yield lf.UtteranceEnd(1, 0.1)
                elif outcome == "failure":
                    raise lf.InferenceError("native cleanup failed") from None

    engine = NativeEngine()
    await engine.prepare()
    stream = engine.stream(_pcm())
    await anext(stream)
    reading = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    await engine.aclose()
    with pytest.raises(lf.EngineClosedError):
        await reading
    assert engine.releases == 1


@pytest.mark.req("SP-LF-055")
async def test_repeated_consumer_cancellation_does_not_interrupt_native_cleanup() -> None:
    """利用側が読み取りを重ねて取り消しても推論の後始末の終了まで新しい認識を始めない。"""
    engine = _Engine()
    await engine.prepare()
    stream = engine.stream(_pcm())
    await anext(stream)
    engine.recognition_finishing.clear()
    reading = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    reading.cancel()
    await engine.recognition_cancelled.wait()
    reading.cancel()
    await asyncio.sleep(0)
    assert not reading.done()
    engine.recognition_finishing.set()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert engine.log == ["loaded", "recognition_finished"]
    await engine.aclose()


@pytest.mark.req("SP-LF-055")
@pytest.mark.parametrize("finish", ["cancel_read", "close_iterator"])
async def test_consumer_termination_finishes_when_native_cancellation_returns_an_event(
    finish: str,
) -> None:
    """取り消し後の最後の event は読み取りを待たせず、推論と列の後始末の後に再利用できる。"""
    original_tasks = asyncio.all_tasks()

    class NativeEngine(_Engine):
        def __init__(self) -> None:
            super().__init__()
            self.waiting = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.finish_native = asyncio.Event()
            self.native_finished = asyncio.Event()
            self.finish_cleanup = asyncio.Event()
            self.cleanup_finished = asyncio.Event()

        async def _recognize(
            self, pcm_chunks: AsyncIterator[bytes]
        ) -> AsyncIterator[lf.TranscriptEvent]:
            del pcm_chunks
            try:
                yield lf.UtteranceStart(1, 0.0)
                self.waiting.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    await self.finish_native.wait()
                    self.native_finished.set()
                    yield lf.UtteranceEnd(1, 0.1)
            finally:
                await self.finish_cleanup.wait()
                self.cleanup_finished.set()

    engine = NativeEngine()
    try:
        await engine.prepare()
        stream = engine.stream(_pcm())
        assert await anext(stream) == lf.UtteranceStart(1, 0.0)
        reading = asyncio.create_task(anext(stream))
        await asyncio.wait_for(engine.waiting.wait(), timeout=1)
        if finish == "cancel_read":
            reading.cancel()
            ending = reading
        else:
            ending = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(engine.cancelled.wait(), timeout=1)
        assert not ending.done()
        assert engine.health().busy
        engine.finish_native.set()
        await asyncio.wait_for(engine.native_finished.wait(), timeout=1)
        assert not ending.done()
        with pytest.raises(lf.EngineStateError):
            await anext(engine.stream(_pcm()))
        engine.finish_cleanup.set()
        finished, _ = await asyncio.wait((ending,), timeout=0.25)
        assert finished, "native inference finished but stream termination did not finish"
        if finish == "cancel_read":
            with pytest.raises(asyncio.CancelledError):
                await ending
        else:
            await ending
        await asyncio.gather(reading, return_exceptions=True)
        assert engine.cleanup_finished.is_set()
        assert not engine.health().busy
        next_stream = engine.stream(_pcm())
        assert await anext(next_stream) == lf.UtteranceStart(1, 0.0)
        await next_stream.aclose()
        await engine.aclose()
        await engine.aclose()
        assert engine.releases == 1
    finally:
        # A regression must fail within this test instead of stalling suite teardown.
        engine.finish_native.set()
        engine.finish_cleanup.set()
        pending = asyncio.all_tasks() - original_tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
