from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import Future
from contextlib import aclosing, closing

import pytest

import listenflow as lf


class _Engine(lf.STTEngineBase):
    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []
        self.operation_loops: list[asyncio.AbstractEventLoop] = []
        self.received: list[list[bytes]] = []
        self.cleanup_gate: threading.Event | None = None
        self.cleanup_started = threading.Event()

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
        self.operation_loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        self.log.append("loaded")

    async def _release(self) -> None:
        self.operation_loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        self.log.append("released")

    async def _recognize(
        self, pcm_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[lf.TranscriptEvent]:
        self.operation_loops.append(asyncio.get_running_loop())
        chunks: list[bytes] = []
        self.received.append(chunks)
        utterance_id = len(self.received)
        self.log.append(f"stream_started:{utterance_id}")
        try:
            async with aclosing(pcm_chunks):
                yield lf.UtteranceStart(utterance_id, 0.0)
                async for chunk in pcm_chunks:
                    chunks.append(chunk)
                yield lf.Final(utterance_id, 0.0, text=b"".join(chunks).hex())
                yield lf.UtteranceEnd(utterance_id, sum(map(len, chunks)) / 2 / 16000)
        finally:
            self.cleanup_started.set()
            if self.cleanup_gate is not None:
                await asyncio.to_thread(self.cleanup_gate.wait)
            await asyncio.sleep(0)
            self.log.append(f"stream_finished:{utterance_id}")


@pytest.mark.req("SP-LF-051")
@pytest.mark.req("SP-LF-053")
def test_sync_caller_prepares_recognizes_and_releases_resources_once() -> None:
    """同期の呼び出し元は engine を用意し、PCM の認識の通知を順に受けて資源を一度だけ返せる。"""
    engine = _Engine()
    chunks = [b"\x01\x00\x02\x00", b"\x03\x00"]
    try:
        lf.run_sync(engine.prepare())
        events = list(lf.iterate_sync(engine.stream(lf.iterate_async(chunks))))

        assert events == [
            lf.UtteranceStart(1, 0.0),
            lf.Final(1, 0.0, text="010002000300"),
            lf.UtteranceEnd(1, 3 / 16000),
        ]
        assert engine.received == [chunks]
        assert engine.log == ["loaded", "stream_started:1", "stream_finished:1"]
    finally:
        lf.run_sync(engine.aclose())

    lf.run_sync(engine.aclose())
    assert engine.log == ["loaded", "stream_started:1", "stream_finished:1", "released"]
    assert len(engine.operation_loops) == 3
    assert len(set(engine.operation_loops)) == 1


@pytest.mark.req("SP-LF-051")
@pytest.mark.req("SP-LF-054")
def test_sync_caller_reads_successive_streams_after_each_cleanup() -> None:
    """同期の呼び出し元は先行する認識の終了処理の後に同じ engine で次の認識を始められる。"""
    engine = _Engine()
    try:
        lf.run_sync(engine.prepare())
        first = list(lf.iterate_sync(engine.stream(lf.iterate_async([b"\x01\x00"]))))
        second = list(lf.iterate_sync(engine.stream(lf.iterate_async([b"\x02\x00"]))))

        assert first == [
            lf.UtteranceStart(1, 0.0),
            lf.Final(1, 0.0, text="0100"),
            lf.UtteranceEnd(1, 1 / 16000),
        ]
        assert second == [
            lf.UtteranceStart(2, 0.0),
            lf.Final(2, 0.0, text="0200"),
            lf.UtteranceEnd(2, 1 / 16000),
        ]
        assert engine.received == [[b"\x01\x00"], [b"\x02\x00"]]
        assert engine.log == [
            "loaded",
            "stream_started:1",
            "stream_finished:1",
            "stream_started:2",
            "stream_finished:2",
        ]
    finally:
        lf.run_sync(engine.aclose())


@pytest.mark.req("SP-LF-051")
@pytest.mark.req("SP-LF-055")
def test_sync_iterator_close_waits_for_engine_cleanup_before_reuse() -> None:
    """同期の列を途中で閉じる操作は認識の終了処理を待ち、その後に同じ engine を再利用できる。"""
    cleanup_gate = threading.Event()
    engine = _Engine()
    closed: Future[None] = Future()
    closer: threading.Thread | None = None
    try:
        lf.run_sync(engine.prepare())
        with closing(lf.iterate_sync(engine.stream(lf.iterate_async([b"\x01\x00"])))) as first:
            assert next(first) == lf.UtteranceStart(1, 0.0)

            def close() -> None:
                try:
                    first.close()
                except BaseException as error:
                    closed.set_exception(error)
                else:
                    closed.set_result(None)

            closer = threading.Thread(target=close, daemon=True)
            engine.cleanup_gate = cleanup_gate
            try:
                closer.start()
                assert engine.cleanup_started.wait(timeout=5)
                assert not closed.done()
                assert engine.log == ["loaded", "stream_started:1"]
            finally:
                cleanup_gate.set()
            closed.result(timeout=5)
            closer.join(timeout=5)
            assert not closer.is_alive()
            assert list(first) == []

        second = list(lf.iterate_sync(engine.stream(lf.iterate_async([b"\x02\x00"]))))
        assert second == [
            lf.UtteranceStart(2, 0.0),
            lf.Final(2, 0.0, text="0200"),
            lf.UtteranceEnd(2, 1 / 16000),
        ]
        assert engine.log == [
            "loaded",
            "stream_started:1",
            "stream_finished:1",
            "stream_started:2",
            "stream_finished:2",
        ]
    finally:
        cleanup_gate.set()
        if closer is not None:
            closer.join(timeout=5)
        lf.run_sync(engine.aclose())


@pytest.mark.req("SP-LF-051")
@pytest.mark.req("SP-LF-053")
def test_sync_helpers_propagate_closed_engine_errors() -> None:
    """同期の道具を通しても閉じた engine は用意と認識を閉じた状態の error で断る。"""
    engine = _Engine()
    try:
        lf.run_sync(engine.prepare())
    finally:
        lf.run_sync(engine.aclose())

    with pytest.raises(lf.EngineClosedError):
        lf.run_sync(engine.prepare())
    with pytest.raises(lf.EngineClosedError):
        list(lf.iterate_sync(engine.stream(lf.iterate_async([b"\x01\x00"]))))
    assert engine.received == []
    assert engine.log == ["loaded", "released"]
