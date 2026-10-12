from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
from collections.abc import AsyncIterator, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar

import pytest


@pytest.mark.req("SP-LF-051")
def test_sync_helpers_share_one_background_loop_across_threads() -> None:
    """複数の thread からの run_sync() と iterate_sync() は一つの背景 event loop で動く。"""
    from listenflow.sync import iterate_sync, run_sync

    barrier = threading.Barrier(8)

    async def observe() -> tuple[asyncio.AbstractEventLoop, int]:
        await asyncio.sleep(0)
        return asyncio.get_running_loop(), threading.get_ident()

    async def source() -> AsyncIterator[tuple[asyncio.AbstractEventLoop, int]]:
        yield await observe()

    def call(index: int) -> tuple[asyncio.AbstractEventLoop, int, int]:
        caller = threading.get_ident()
        barrier.wait(timeout=5)
        if index % 2:
            loop, thread = run_sync(observe())
        else:
            [(loop, thread)] = list(iterate_sync(source()))
        return loop, thread, caller

    with ThreadPoolExecutor(max_workers=8) as callers:
        observations = list(callers.map(call, range(8)))

    assert len({loop for loop, _, _ in observations}) == 1
    assert len({thread for _, thread, _ in observations}) == 1
    assert all(thread != caller for _, thread, caller in observations)


@pytest.mark.req("SP-LF-051")
async def test_sync_helpers_work_inside_a_running_event_loop() -> None:
    """利用側の実行中の event loop から run_sync() と iterate_sync() を呼べる。"""
    from listenflow.sync import iterate_sync, run_sync

    caller_loop = asyncio.get_running_loop()

    async def operation() -> tuple[str, asyncio.AbstractEventLoop]:
        await asyncio.sleep(0)
        return "result", asyncio.get_running_loop()

    async def source() -> AsyncIterator[int]:
        for value in (3, 1, 4):
            await asyncio.sleep(0)
            yield value

    value, background_loop = run_sync(operation())

    assert value == "result"
    assert background_loop is not caller_loop
    assert list(iterate_sync(source())) == [3, 1, 4]
    assert asyncio.get_running_loop() is caller_loop


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize("awaitable_kind", ["coroutine", "custom"])
def test_run_sync_rejects_background_loop_calls_without_starting_work(
    awaitable_kind: str,
) -> None:
    """背景 event loop の thread からの run_sync() は待たずに EngineStateError で断る。

    error は RuntimeError を兼ね、message は関数名、呼び出し元の thread、代わりの await を持つ。
    """
    source = """
        import asyncio
        import gc
        import sys
        import warnings

        from listenflow import EngineStateError, run_sync

        started = False

        async def operation():
            global started
            started = True
            return "result"

        class Operation:
            def __await__(self):
                return operation().__await__()

        async def nested():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                awaitable = operation() if sys.argv[1] == "coroutine" else Operation()
                try:
                    run_sync(awaitable)
                except EngineStateError as error:
                    assert isinstance(error, RuntimeError)
                    message = str(error)
                    assert "run_sync()" in message
                    assert "listenflow background event loop thread" in message
                    assert "await" in message
                else:
                    raise AssertionError("background loop call was accepted")
                del awaitable
                gc.collect()
                assert not caught, [str(warning.message) for warning in caught]
            await asyncio.sleep(0)
            assert not started, "rejected operation was started"
            return await operation()

        assert run_sync(nested()) == "result"
        assert started
        assert run_sync(operation()) == "result"
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source), awaitable_kind],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("run_sync() blocked the background event loop instead of rejecting the call")

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize("coroutine_state", ["suspended", "running"])
def test_run_sync_rejects_started_coroutines_with_the_state_error(coroutine_state: str) -> None:
    """開始済みや実行中の coroutine を背景 loop から渡しても、状態の error と await の案内を返す。"""
    source = """
        import asyncio
        import sys

        from listenflow import EngineStateError, run_sync

        async def operation():
            try:
                await asyncio.sleep(0)
            finally:
                raise LookupError("source cleanup")

        async def nested():
            suspended = sys.argv[1] == "suspended"
            if suspended:
                coroutine = operation()
                coroutine.send(None)
            else:
                coroutine = asyncio.current_task().get_coro()
            try:
                try:
                    run_sync(coroutine)
                except EngineStateError as error:
                    assert isinstance(error, RuntimeError)
                    message = str(error)
                    assert "run_sync()" in message
                    assert "listenflow background event loop thread" in message
                    assert "await" in message
                else:
                    raise AssertionError("background loop call was accepted")
            finally:
                if suspended:
                    try:
                        coroutine.close()
                    except LookupError:
                        pass
            await asyncio.sleep(0)
            return "still usable"

        assert run_sync(nested()) == "still usable"
        assert run_sync(nested()) == "still usable"
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source), coroutine_state],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("run_sync() blocked the background event loop instead of rejecting the call")

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.req("SP-LF-051")
def test_iterate_sync_rejects_background_loop_calls_before_requesting_items() -> None:
    """背景 event loop の thread からの iterate_sync() は呼んだ時点で EngineStateError で断る。

    error は RuntimeError を兼ね、message は関数名、呼び出し元の thread、代わりの async for を持つ。
    """
    source = """
        import asyncio

        from listenflow import EngineStateError, iterate_sync, run_sync

        started = False

        class Source:
            def __aiter__(self):
                global started
                started = True
                return self.values()

            async def values(self):
                yield "item"

        async def nested():
            source = Source()
            try:
                iterate_sync(source)
            except EngineStateError as error:
                assert isinstance(error, RuntimeError)
                message = str(error)
                assert "iterate_sync()" in message
                assert "listenflow background event loop thread" in message
                assert "async for" in message
            else:
                raise AssertionError("iterate_sync() returned before rejecting the call")
            await asyncio.sleep(0)
            assert not started, "rejected source was started"
            return [value async for value in source]

        assert run_sync(nested()) == ["item"]
        assert list(iterate_sync(Source())) == ["item"]
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "iterate_sync() blocked the background event loop instead of rejecting the call"
        )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.req("SP-LF-051")
def test_run_sync_accepts_an_awaitable_and_returns_its_result() -> None:
    """非同期の操作は coroutine に限らず awaitable を受けて元の値を返す。"""
    from listenflow.sync import run_sync

    expected = object()

    class Operation:
        def __await__(self) -> Generator[object, None, object]:
            async def execute() -> object:
                await asyncio.sleep(0)
                return expected

            return execute().__await__()

    assert run_sync(Operation()) is expected


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize(
    "error", [ValueError("operation failed"), asyncio.CancelledError("cancelled")]
)
def test_run_sync_preserves_the_operation_exception(error: BaseException) -> None:
    """非同期の操作の失敗と取り消しは元の例外のまま呼び出し元へ伝わる。"""
    from listenflow.sync import run_sync

    async def operation() -> None:
        await asyncio.sleep(0)
        raise error

    with pytest.raises(type(error)) as caught:
        run_sync(operation())

    assert caught.value is error


@pytest.mark.req("SP-LF-051")
def test_iterate_sync_reads_an_async_iterable_on_the_background_loop() -> None:
    """非同期の iterable の開始と読み取りは背景 loop で行い、値の順序を保つ。"""
    from listenflow.sync import iterate_sync, run_sync

    async def current_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    expected_loop = run_sync(current_loop())
    seen_loops = []

    class Source:
        def __aiter__(self) -> AsyncIterator[int]:
            seen_loops.append(asyncio.get_running_loop())
            return self.values()

        async def values(self) -> AsyncIterator[int]:
            for value in (7, 2, 5):
                seen_loops.append(asyncio.get_running_loop())
                await asyncio.sleep(0)
                yield value

    assert list(iterate_sync(Source())) == [7, 2, 5]
    assert seen_loops == [expected_loop] * 4


@pytest.mark.req("SP-LF-051")
def test_iterate_sync_propagates_failure_after_finishing_cleanup() -> None:
    """非同期の列は既読の値を保ち、終了処理の後に元の失敗を伝える。"""
    from listenflow.sync import iterate_sync

    error = LookupError("source failed")
    finished = threading.Event()

    async def source() -> AsyncIterator[str]:
        try:
            yield "first"
            raise error
        finally:
            await asyncio.sleep(0)
            finished.set()

    iterator = iterate_sync(source())
    assert next(iterator) == "first"

    with pytest.raises(LookupError) as caught:
        next(iterator)

    assert caught.value is error
    assert finished.is_set()
    assert list(iterator) == []


@pytest.mark.req("SP-LF-051")
def test_iterate_sync_close_waits_for_async_cleanup() -> None:
    """同期の列を途中で閉じると、非同期の列の終了処理が終わるまで待つ。"""
    from listenflow.sync import iterate_sync

    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    cleanup_finished = threading.Event()
    close_result: Future[None] = Future()

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        finally:
            cleanup_started.set()
            await asyncio.to_thread(release_cleanup.wait)
            cleanup_finished.set()

    iterator = iterate_sync(source())
    assert next(iterator) == 1

    def close() -> None:
        try:
            iterator.close()
        except BaseException as error:
            close_result.set_exception(error)
        else:
            close_result.set_result(None)

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    try:
        assert cleanup_started.wait(timeout=5)
        assert not close_result.done()
    finally:
        release_cleanup.set()

    close_result.result(timeout=5)
    closer.join(timeout=5)
    assert cleanup_finished.is_set()
    assert list(iterator) == []


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize("close_early", [False, True])
def test_iterate_sync_keeps_the_source_context_through_cleanup(close_early: bool) -> None:
    """非同期の列の読み取りと終了処理は、列が設定した context を引き継ぐ。"""
    from listenflow.sync import iterate_sync

    state: ContextVar[str] = ContextVar("source_state", default="outside")
    observations = []

    async def source() -> AsyncIterator[str]:
        token = state.set("inside")
        try:
            yield state.get()
            observations.append(state.get())
            yield state.get()
        finally:
            observations.append(state.get())
            state.reset(token)
            observations.append(state.get())

    iterator = iterate_sync(source())
    assert next(iterator) == "inside"
    if close_early:
        iterator.close()
        assert observations == ["inside", "outside"]
    else:
        assert list(iterator) == ["inside"]
        assert observations == ["inside", "inside", "outside"]
    assert state.get() == "outside"


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize("helper", ["run_sync", "iterate_sync"])
def test_caller_sigint_cancels_background_work_and_waits_for_cleanup(helper: str) -> None:
    """run_sync() と iterate_sync() は同期の呼び出し元の中断で背景の操作を取り消す。

    両関数は背景の操作の後始末の終了を待ってから、呼び出し元へ中断を伝える。
    """
    source = """
        import asyncio
        import os
        import signal
        import sys
        import threading

        from listenflow.sync import iterate_sync, run_sync

        helper = sys.argv[1]
        started = threading.Event()
        cancelled = threading.Event()
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        cleanup_finished = threading.Event()
        returned = threading.Event()
        controller_errors = []

        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                cleanup_started.set()
                await asyncio.to_thread(release_cleanup.wait)
                cleanup_finished.set()

        async def values():
            yield "first"
            await operation()

        def interrupt():
            try:
                assert started.wait(5), "operation did not start"
                os.kill(os.getpid(), signal.SIGINT)
                assert cleanup_started.wait(5), "cleanup did not start"
                assert not returned.is_set(), "caller returned before cleanup"
            except BaseException as error:
                controller_errors.append(error)
            finally:
                release_cleanup.set()

        controller = threading.Thread(target=interrupt, daemon=True)
        controller.start()
        try:
            if helper == "run_sync":
                run_sync(operation())
            else:
                iterator = iterate_sync(values())
                assert next(iterator) == "first"
                next(iterator)
        except KeyboardInterrupt:
            returned.set()
            assert cancelled.is_set(), "background operation was not cancelled"
            assert cleanup_finished.is_set(), "caller did not wait for cleanup"
        else:
            raise AssertionError("caller interrupt was not propagated")
        finally:
            release_cleanup.set()
            controller.join(6)

        assert not controller.is_alive(), "interrupt controller did not stop"
        assert not controller_errors, controller_errors
        async def after():
            return "still usable"
        assert run_sync(after()) == "still usable"
    """

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), helper],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize("helper", ["run_sync", "iterate_sync"])
@pytest.mark.parametrize("clear_cancellation", [False, True])
def test_caller_sigint_waits_for_sources_that_return_after_cancellation(
    helper: str, clear_cancellation: bool
) -> None:
    """自 subprocess への SIGINT を入口に、取り消し後に値を返す操作と列の終了待ちと再利用を確かめる。"""
    source = """
        import asyncio
        import os
        import signal
        import sys
        import threading

        from listenflow.sync import iterate_sync, run_sync

        helper = sys.argv[1]
        clear_cancellation = sys.argv[2] == "True"
        started = threading.Event()
        cancelled = threading.Event()
        returned_final_value = threading.Event()
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        cleanup_finished = threading.Event()
        caller_returned = threading.Event()
        controller_errors = []

        async def current_loop():
            return asyncio.get_running_loop()

        original_loop = run_sync(current_loop())

        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                if clear_cancellation:
                    asyncio.current_task().uncancel()
                await asyncio.sleep(0)
                return "final value"

        async def cleanup():
            cleanup_started.set()
            await asyncio.to_thread(release_cleanup.wait)
            cleanup_finished.set()

        async def scalar():
            try:
                return await operation()
            finally:
                await cleanup()

        async def values():
            try:
                value = await operation()
                returned_final_value.set()
                yield value
                raise AssertionError("source was read again after caller interruption")
            finally:
                await cleanup()

        def interrupt():
            try:
                assert started.wait(3), "operation did not start"
                os.kill(os.getpid(), signal.SIGINT)
                assert cleanup_started.wait(3), "source did not enter cleanup"
                assert not caller_returned.is_set(), "caller returned before cleanup"
            except BaseException as error:
                controller_errors.append(error)
                print(repr(error), flush=True)
            finally:
                release_cleanup.set()

        controller = threading.Thread(target=interrupt, daemon=True)
        controller.start()
        try:
            if helper == "run_sync":
                run_sync(scalar())
            else:
                iterator = iterate_sync(values())
                next(iterator)
        except KeyboardInterrupt:
            caller_returned.set()
            assert cancelled.is_set(), "background operation was not cancelled"
            assert cleanup_finished.is_set(), "caller did not wait for cleanup"
            if helper == "iterate_sync":
                assert returned_final_value.is_set(), "source did not yield its final value"
                assert list(iterator) == [], "interrupted iterator remained open"
        else:
            raise AssertionError("caller interrupt was replaced by the final value")
        finally:
            release_cleanup.set()
            controller.join(4)

        assert not controller.is_alive(), "interrupt controller did not stop"
        assert not controller_errors, controller_errors

        async def recovered():
            yield "new value", asyncio.get_running_loop()

        assert list(iterate_sync(recovered())) == [("new value", original_loop)]
        assert run_sync(current_loop()) is original_loop
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source), helper, str(clear_cancellation)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("caller did not return after the source handled cancellation")

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.req("SP-LF-051")
@pytest.mark.parametrize("values", [[], [9, 2, 6]])
async def test_iterate_async_reads_the_iterable_off_the_event_loop(values: list[int]) -> None:
    """同期の iterable の開始と読み取りは別の thread で行い、順序と正常な終端を保つ。"""
    from listenflow.sync import iterate_async

    loop_thread = threading.get_ident()
    read_threads = []

    class Source:
        def __iter__(self) -> Iterator[int]:
            read_threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread
            return self.values()

        def values(self) -> Iterator[int]:
            for value in values:
                read_threads.append(threading.get_ident())
                assert threading.get_ident() != loop_thread
                yield value

    assert [value async for value in iterate_async(Source())] == values
    assert len(read_threads) == len(values) + 1
    assert all(thread != loop_thread for thread in read_threads)


@pytest.mark.req("SP-LF-051")
async def test_iterate_async_preserves_the_iterable_exception() -> None:
    """同期の列の既読の値を保ち、失敗は元の例外のまま非同期の読み手へ伝える。"""
    from listenflow.sync import iterate_async

    error = OSError("input failed")

    def source() -> Iterator[str]:
        yield "first"
        raise error

    iterator = iterate_async(source())
    assert await anext(iterator) == "first"

    with pytest.raises(OSError) as caught:
        await anext(iterator)

    assert caught.value is error
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)
