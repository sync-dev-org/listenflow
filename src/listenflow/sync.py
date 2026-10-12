"""Bridge asynchronous operations and synchronous callers."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Generator, Iterable
from concurrent.futures import Future
from inspect import CORO_CREATED, getcoroutinestate, iscoroutine
from typing import cast

from .errors import EngineStateError


_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _serve_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _background_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=_serve_loop,
                args=(loop,),
                name="listenflow-sync",
                daemon=True,
            ).start()
            _loop = loop
        return _loop


def _on_background_loop() -> bool:
    try:
        return asyncio.get_running_loop() is _loop
    except RuntimeError:
        return False


class _Operation[T]:
    def __init__(self, awaitable: Awaitable[T]) -> None:
        self.loop = _background_loop()
        self.result: Future[T] = Future()
        self._awaitable = awaitable
        self._task: asyncio.Task[tuple[T | None, BaseException | None]] | None = None

    def start(self) -> None:
        self.loop.call_soon_threadsafe(self._start)

    async def _execute(self) -> tuple[T | None, BaseException | None]:
        try:
            return await self._awaitable, None
        except BaseException as error:
            # Keep control-flow exceptions inside the task so they cannot stop
            # the shared loop or get translated by a cancelled Future.
            return None, error
        finally:
            # The task's result may contain an exception retaining this frame.
            del self

    def _complete(self, finished: asyncio.Task[tuple[T | None, BaseException | None]]) -> None:
        try:
            value, error = finished.result()
        except BaseException as failure:
            self.result.set_exception(failure)
        else:
            if error is not None:
                self.result.set_exception(error)
            else:
                self.result.set_result(cast(T, value))
        finally:
            del self, finished

    def _start(self) -> None:
        self._task = self.loop.create_task(self._execute())
        self._task.add_done_callback(self._complete)

    def _cancel(self) -> None:
        if self._task is None:
            self._start()
        # Let the task enter the awaitable before delivering cancellation, even
        # when the caller is interrupted immediately after submitting it.
        self.loop.call_soon(self._deliver_cancel)

    def _deliver_cancel(self) -> None:
        assert self._task is not None
        if not self._task.cancelling():
            self._task.cancel()

    def cancel_and_wait(self) -> None:
        try:
            if not self.result.done():
                self.loop.call_soon_threadsafe(self._cancel)
            while not self.result.done():
                try:
                    self.result.result()
                except BaseException:
                    # A further caller interruption must not cut cleanup short.
                    pass
        finally:
            del self

    def wait[R](self, result: Future[R]) -> R:
        try:
            return result.result()
        except BaseException:
            if not result.done():
                if result is not self.result:
                    # Abandon the read, but keep the operation's completion waitable.
                    result.cancel()
                self.cancel_and_wait()
            raise
        finally:
            # Future.result() stores the same exception that passes through here.
            del self, result


def run_sync[T](awaitable: Awaitable[T]) -> T:
    """Run an awaitable on the shared background loop and wait for its result.

    Calls from the background event loop thread raise EngineStateError; await
    the coroutine instead. A rejected coroutine that has not started is closed
    without being awaited.
    If the caller is interrupted while waiting, cancel the operation and wait
    until its cleanup finishes before propagating the caller's interruption.
    """
    if _on_background_loop():
        if iscoroutine(awaitable) and getcoroutinestate(awaitable) == CORO_CREATED:
            awaitable.close()
        raise EngineStateError(
            "run_sync() cannot be called from the listenflow background event loop thread; "
            "await the coroutine instead"
        )
    operation = _Operation(awaitable)
    try:
        operation.start()
        return operation.wait(operation.result)
    finally:
        operation.cancel_and_wait()
        del operation


def iterate_sync[T](async_iterable: AsyncIterable[T]) -> Generator[T, None, None]:
    """Read an asynchronous iterable through a synchronous, closable iterator.

    Calls from the background event loop thread raise EngineStateError at the
    call to this function; use async for instead.
    Iteration and asynchronous close run on the shared background loop. Closing
    this iterator waits for the source's asynchronous cleanup when it has one.
    """
    if _on_background_loop():
        raise EngineStateError(
            "iterate_sync() cannot be called from the listenflow background event loop thread; "
            "use async for instead"
        )
    return _iterate_sync(async_iterable)


def _iterate_sync[T](async_iterable: AsyncIterable[T]) -> Generator[T, None, None]:
    requested = asyncio.Event()
    response: Future[T] | None = None

    async def produce(source: AsyncIterable[T]) -> None:
        iterator = aiter(source)
        try:
            while True:
                await requested.wait()
                requested.clear()
                assert response is not None
                value = await anext(iterator)
                if response.cancelled():
                    # The source may turn cancellation into one last value.
                    return
                response.set_result(value)
        except StopAsyncIteration:
            return
        finally:
            aclose = getattr(iterator, "aclose", None)
            try:
                if aclose is not None:
                    await aclose()
            finally:
                del iterator, aclose, source

    operation = _Operation(produce(async_iterable))

    def finish(result: Future[None]) -> None:
        pending = response
        try:
            if pending is not None and not pending.done():
                try:
                    result.result()
                except BaseException as error:
                    pending.set_exception(error)
                else:
                    pending.set_exception(StopAsyncIteration())
        finally:
            del result, pending

    operation.result.add_done_callback(finish)

    try:
        operation.start()
        while True:
            response = Future()
            if operation.result.done():
                operation.result.result()
                return
            operation.loop.call_soon_threadsafe(requested.set)
            try:
                value = operation.wait(response)
            except StopAsyncIteration:
                return
            yield value
    except GeneratorExit:
        operation.cancel_and_wait()
        try:
            operation.result.result()
        except asyncio.CancelledError:
            pass
        raise
    finally:
        if response is not None:
            response.cancel()
        operation.cancel_and_wait()
        # Release references held by this frame and the producer's closure cells.
        response = None
        del operation, async_iterable


async def iterate_async[T](iterable: Iterable[T]) -> AsyncIterator[T]:
    """Read a synchronous iterable in worker threads without blocking the loop."""
    iterator = await asyncio.to_thread(iter, iterable)
    end = object()
    while True:
        # StopIteration cannot be transported through an asyncio Future.
        value = await asyncio.to_thread(next, iterator, end)
        if value is end:
            return
        yield cast(T, value)
