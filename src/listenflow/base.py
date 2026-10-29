from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Self

from listenflow._async import cancel_once as _cancel, finish as _finish
from listenflow.engine import (
    EngineCapabilities,
    EngineInputFormat,
    Segmentation,
    validate_pcm_chunk,
)
from listenflow.errors import (
    ConfigurationError,
    EngineClosedError,
    EngineStateError,
    InferenceError,
)
from listenflow.events import TranscriptEvent
from listenflow.health import EngineState, HealthSnapshot


class STTEngineBase(ABC):
    """Shared preparation, recognition, closure, and health behavior.

    Use one engine instance on one event loop. Modules implement the hooks,
    map native errors into STTError subclasses, and finish native work before
    returning from a cancelled hook. Failed loads clean up their own resources.
    """

    def __init__(self) -> None:
        self._model_lock = asyncio.Lock()
        self._operations: set[asyncio.Task[None]] = set()
        self._stream_started = asyncio.Event()
        self._loaded = False
        self._close_task: asyncio.Task[None] | None = None
        self._last_progress_monotonic_s: float | None = None

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return the module's immutable, static capability declaration."""
        raise NotImplementedError

    @abstractmethod
    async def _load(self) -> None:
        """Load resources and run a trial inference; clean up a failed load."""
        raise NotImplementedError

    @abstractmethod
    async def _release(self) -> None:
        """Return resources created by a successful load."""
        raise NotImplementedError

    @abstractmethod
    def _recognize(
        self,
        pcm_chunks: AsyncIterator[bytes],
        *,
        segmentation: Segmentation = Segmentation.AUTO,
    ) -> AsyncIterator[TranscriptEvent]:
        """Recognize PCM and finish native work on cancellation or early closure.

        AUTO calls omit the keyword so existing hooks remain usable. Modules
        declaring single_utterance accept the keyword and honor SINGLE.
        """
        raise NotImplementedError

    def _health_details(self) -> Mapping[str, object]:
        """Return recognition fields only, without the three shared health fields."""
        return {}

    def _mark_progress(self) -> None:
        """Record a processed PCM chunk or completed utterance decode."""
        self._last_progress_monotonic_s = time.monotonic()

    @property
    def _closing(self) -> bool:
        return self._close_task is not None

    def _check_open(self) -> None:
        if self._closing:
            raise EngineClosedError("engine is closed")

    def _start(self, coroutine: Coroutine[object, object, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._operations.add(task)
        task.add_done_callback(self._operations.discard)
        return task

    async def prepare(self) -> None:
        """Load and trial the model once; retry after an unsuccessful load."""
        self._check_open()
        if self._loaded:
            return
        task = self._start(self._prepare())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if self._closing and caller is not None and not caller.cancelling():
                raise EngineClosedError("engine closed while preparing") from None
            _cancel(task)
            await _finish(task)
            raise
        except Exception as error:
            if self._closing:
                raise EngineClosedError("engine closed while preparing") from error
            raise
        finally:
            # A failed task retains this frame through the exception traceback.
            del task

    async def _prepare(self) -> None:
        acquired = False
        try:
            await self._acquire_model_slot()
            acquired = True
            self._check_open()
            if not self._loaded:
                await self._load()
                self._loaded = True
            self._check_open()
        except asyncio.CancelledError:
            if self._closing:
                raise EngineClosedError("engine closed while preparing") from None
            raise
        finally:
            if acquired:
                self._model_lock.release()

    def stream(
        self,
        pcm_chunks: AsyncIterator[bytes],
        *,
        segmentation: Segmentation = Segmentation.AUTO,
    ) -> AsyncIterator[TranscriptEvent]:
        """Start recognition on the first read and keep its slot through cleanup.

        The returned iterator supports aclose(). Close it explicitly when stopping
        before the end, or use contextlib.aclosing around the iterator.
        SINGLE requires capabilities.single_utterance and transcribes the entire
        input once without automatic boundaries. Invalid or unsupported choices
        raise ConfigurationError on the first read, before recognition.
        """
        return _EngineStream(self, pcm_chunks, segmentation=segmentation)

    async def _acquire_model_slot(self, *, recognition: bool = False) -> None:
        self._check_open()
        if recognition and self._stream_started.is_set():
            raise EngineStateError("engine already has an active stream")
        acquire = asyncio.create_task(self._model_lock.acquire())
        overlap = asyncio.create_task(self._stream_started.wait()) if recognition else None
        try:
            if overlap is not None:
                await asyncio.wait((acquire, overlap), return_when=asyncio.FIRST_COMPLETED)
            self._check_open()
            if overlap is not None and overlap.done():
                raise EngineStateError("engine already has an active stream")
            await acquire
        except BaseException:
            _cancel(acquire)
            if overlap is not None:
                _cancel(overlap)
                await _finish(overlap)
            await _finish(acquire)
            if not acquire.cancelled() and acquire.exception() is None:
                self._model_lock.release()
            raise
        finally:
            if overlap is not None:
                _cancel(overlap)

    async def _validated_pcm(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        input_format = self.capabilities.input_format
        async for chunk in pcm_chunks:
            if input_format in (EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32):
                try:
                    validate_pcm_chunk(chunk, input_format=input_format)
                except ValueError as error:
                    raise ConfigurationError(str(error)) from error
            yield chunk

    async def _run_stream(self, stream: _EngineStream) -> None:
        iterator: AsyncIterator[TranscriptEvent] | None = None
        acquired = False
        recognizing = False
        try:
            await self._acquire_model_slot(recognition=True)
            acquired = True
            if not self._loaded:
                raise EngineStateError("prepare the engine before recognition")
            if not isinstance(stream._segmentation, Segmentation):
                raise ConfigurationError(
                    "segmentation must be Segmentation.AUTO or Segmentation.SINGLE"
                )
            if (
                stream._segmentation is Segmentation.SINGLE
                and not self.capabilities.single_utterance
            ):
                raise ConfigurationError("engine does not declare single_utterance support")
            self._stream_started.set()
            recognizing = True
            pcm_chunks = self._validated_pcm(stream._pcm_chunks)
            iterator = aiter(
                self._recognize(pcm_chunks)
                if stream._segmentation is Segmentation.AUTO
                else self._recognize(pcm_chunks, segmentation=stream._segmentation)
            )
            while True:
                request = await stream._requests.get()
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    self._check_open()
                    break
                self._check_open()
                if stream._closed:
                    break
                if not request.done():
                    request.set_result(event)
        except asyncio.CancelledError:
            if self._closing:
                raise EngineClosedError("engine closed during recognition") from None
            raise
        finally:
            try:
                if iterator is not None and hasattr(iterator, "aclose"):
                    await iterator.aclose()
            finally:
                if recognizing:
                    self._stream_started.clear()
                if acquired:
                    self._model_lock.release()
                # Do not let an exception traceback lead back to its task or Future.
                request = None
                del stream

    async def aclose(self) -> None:
        """Cancel operations, wait for their cleanup, and release resources once."""
        first = self._close_task is None
        if first:
            self._close_task = asyncio.create_task(self._close())
        task = self._close_task
        assert task is not None
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await _finish(task)
            raise
        except BaseException:
            if first:
                raise

    async def _close(self) -> None:
        operations = tuple(self._operations)
        for task in operations:
            _cancel(task)
        await asyncio.gather(*operations, return_exceptions=True)
        if self._loaded:
            self._loaded = False
            try:
                await self._release()
            except Exception as error:
                raise InferenceError("engine resource release failed") from error

    async def __aenter__(self) -> Self:
        await self.prepare()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    def health(self) -> HealthSnapshot:
        """Return held values immediately, including after closure."""
        state = (
            EngineState.CLOSED
            if self._closing
            else EngineState.READY
            if self._loaded
            else EngineState.UNPREPARED
        )
        return HealthSnapshot(
            state=state,
            busy=self._model_lock.locked(),
            last_progress_monotonic_s=self._last_progress_monotonic_s,
            **self._health_details(),
        )


class _EngineStream(AsyncIterator[TranscriptEvent]):
    """Keep the model operation alive while the consumer pauses between reads."""

    def __init__(
        self,
        engine: STTEngineBase,
        pcm_chunks: AsyncIterator[bytes],
        *,
        segmentation: Segmentation,
    ) -> None:
        self._engine = engine
        self._pcm_chunks = pcm_chunks
        self._segmentation = segmentation
        self._requests: asyncio.Queue[asyncio.Future[TranscriptEvent]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._pending: asyncio.Future[TranscriptEvent] | None = None
        self._closed = False

    async def __anext__(self) -> TranscriptEvent:
        future = None
        try:
            if self._closed:
                raise StopAsyncIteration
            self._engine._check_open()
            if self._pending is not None:
                raise EngineStateError("stream already has a pending read")
            if self._task is not None and self._task.done():
                self._task.result()
                self._closed = True
                raise StopAsyncIteration
            future = asyncio.get_running_loop().create_future()
            self._pending = future
            self._requests.put_nowait(future)
            if self._task is None:
                self._task = self._engine._start(self._engine._run_stream(self))
                self._task.add_done_callback(self._completed)
            try:
                return await future
            except asyncio.CancelledError:
                await self.aclose()
                raise
            finally:
                self._pending = None
        finally:
            # Both the Future and the stream's task may hold the raised exception.
            del self, future

    def _completed(self, task: asyncio.Task[None]) -> None:
        error = (
            EngineClosedError("engine closed during recognition")
            if task.cancelled() and self._engine._closing
            else asyncio.CancelledError()
            if task.cancelled()
            else task.exception()
        )
        if self._engine._closing and not isinstance(error, EngineClosedError):
            closed = EngineClosedError("engine closed during recognition")
            closed.__cause__ = error
            error = closed
        if self._pending is not None and not self._pending.done():
            self._pending.set_exception(error or StopAsyncIteration())

    async def aclose(self) -> None:
        self._closed = True
        if self._task is not None:
            _cancel(self._task)
            await _finish(self._task)
