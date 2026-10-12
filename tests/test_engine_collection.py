from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator, Iterator

import pytest

import listenflow as lf


class _Engine(lf.STTEngineBase):
    def __init__(self, *, fail_load: bool = False, fail_recognition: bool = False) -> None:
        super().__init__()
        self.fail_load = fail_load
        self.fail_recognition = fail_recognition

    @property
    def capabilities(self) -> lf.EngineCapabilities:
        return lf.EngineCapabilities(
            transport=lf.EngineTransport.IN_PROCESS,
            input_format=lf.EngineInputFormat.PCM_S16LE,
            partial_mode=lf.PartialMode.NONE,
            finalization=frozenset({lf.FinalizationMode.LOCAL_VAD}),
            timestamp_detail=lf.TimestampDetail.SEGMENT,
            diarization=lf.DiarizationMode.NONE,
            languages=frozenset(),
            auto_language=False,
        )

    async def _load(self) -> None:
        if self.fail_load:
            raise lf.InferenceError("load failed")

    async def _release(self) -> None:
        pass

    async def _recognize(
        self, pcm_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[lf.TranscriptEvent]:
        async for _ in pcm_chunks:
            yield lf.UtteranceStart(1, 0.0)
        if self.fail_recognition:
            raise lf.InferenceError("recognition failed")
        yield lf.UtteranceEnd(1, 0.01)


async def _pcm() -> AsyncIterator[bytes]:
    yield b"\x00\x00" * 160


@pytest.fixture
def without_cyclic_collection() -> Iterator[None]:
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.req("SP-LF-053")
def test_engine_is_collected_after_consuming_its_stream(without_cyclic_collection: None) -> None:
    """利用側が認識の列を最後まで読んで engine を手放すと、循環参照の回収を待たずに回収される。"""

    async def consume() -> weakref.ReferenceType[_Engine]:
        engine = _Engine()
        reference = weakref.ref(engine)
        await engine.prepare()
        assert [event async for event in engine.stream(_pcm())] == [
            lf.UtteranceStart(1, 0.0),
            lf.UtteranceEnd(1, 0.01),
        ]
        return reference

    reference = asyncio.run(consume())
    assert reference() is None


@pytest.mark.req("SP-LF-053")
def test_engine_is_collected_after_receiving_recognition_failure(
    without_cyclic_collection: None,
) -> None:
    """利用側が認識の失敗を受けて engine を手放すと、循環参照の回収を待たずに回収される。"""

    async def consume() -> weakref.ReferenceType[_Engine]:
        engine = _Engine(fail_recognition=True)
        reference = weakref.ref(engine)
        await engine.prepare()
        events = []
        try:
            async for event in engine.stream(_pcm()):
                events.append(event)
        except lf.InferenceError:
            pass
        else:
            pytest.fail("recognition failure was not delivered")
        assert events == [lf.UtteranceStart(1, 0.0)]
        return reference

    reference = asyncio.run(consume())
    assert reference() is None


@pytest.mark.req("SP-LF-055")
@pytest.mark.parametrize("fail_cleanup", [False, True], ids=["cleanup_success", "cleanup_failure"])
def test_engine_is_collected_after_closing_stream_early(
    without_cyclic_collection: None, fail_cleanup: bool
) -> None:
    """利用側が列を途中で閉じて engine を手放すと、後始末の成否によらず循環参照の回収を待たずに回収される。"""
    cleanup_finished = []

    class ClosingEngine(_Engine):
        async def _recognize(
            self, pcm_chunks: AsyncIterator[bytes]
        ) -> AsyncIterator[lf.TranscriptEvent]:
            del pcm_chunks
            try:
                yield lf.UtteranceStart(1, 0.0)
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleanup_finished.append(True)
                if fail_cleanup:
                    raise lf.InferenceError("recognition cleanup failed")

    async def consume() -> weakref.ReferenceType[_Engine]:
        engine = ClosingEngine()
        reference = weakref.ref(engine)
        await engine.prepare()
        stream = engine.stream(_pcm())
        assert await anext(stream) == lf.UtteranceStart(1, 0.0)
        await stream.aclose()
        assert cleanup_finished == [True]
        assert not engine.health().busy
        return reference

    reference = asyncio.run(consume())
    assert reference() is None


@pytest.mark.req("SP-LF-053")
def test_engine_is_collected_after_run_sync_preparation(without_cyclic_collection: None) -> None:
    """利用側が同期の用意を終えて engine を手放すと、循環参照の回収を待たずに回収される。"""
    engine = _Engine()
    reference = weakref.ref(engine)
    lf.run_sync(engine.prepare())
    assert engine.health().state == lf.EngineState.READY
    del engine
    # Let completed background callbacks return before checking ownership.
    lf.run_sync(asyncio.sleep(0))
    assert reference() is None


@pytest.mark.req("SP-LF-053")
def test_engine_is_collected_after_run_sync_failure(without_cyclic_collection: None) -> None:
    """利用側が同期の用意の失敗を受けて engine を手放すと、循環参照の回収を待たずに回収される。"""
    engine = _Engine(fail_load=True)
    reference = weakref.ref(engine)
    try:
        lf.run_sync(engine.prepare())
    except lf.InferenceError:
        pass
    else:
        pytest.fail("preparation failure was not delivered")
    del engine
    lf.run_sync(asyncio.sleep(0))
    assert reference() is None


@pytest.mark.req("SP-LF-053")
def test_engine_is_collected_after_consuming_iterate_sync(without_cyclic_collection: None) -> None:
    """利用側が同期の認識の列を最後まで読んで engine を手放すと、循環参照の回収を待たずに回収される。"""
    engine = _Engine()
    reference = weakref.ref(engine)
    lf.run_sync(engine.prepare())
    assert list(lf.iterate_sync(engine.stream(_pcm()))) == [
        lf.UtteranceStart(1, 0.0),
        lf.UtteranceEnd(1, 0.01),
    ]
    del engine
    lf.run_sync(asyncio.sleep(0))
    assert reference() is None
