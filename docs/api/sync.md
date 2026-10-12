# Synchronous helper API

These three helpers are available directly from `listenflow`. Engines expose asynchronous
operations; synchronous callers use the helpers to prepare an engine, supply PCM, read transcript
events, and close the engine.

`run_sync()` runs an awaitable and waits for its result. `iterate_sync()` reads an asynchronous
iterable through a synchronous iterator. Both run their operations on the same library-wide
background event loop. Use one engine instance on one event loop: keep its preparation,
recognition, and closure on that background loop when using these helpers.

`iterate_async()` adapts a synchronous iterable into an asynchronous iterator, reading the input
in worker threads so the event loop can continue running. Values retain their input order, and
input failures propagate to the caller.

The helpers support callers in multiple threads and inside a running event loop. Calling
`run_sync()` or `iterate_sync()` from the shared background event loop thread raises
`EngineStateError`; use `await` or `async for` there instead. If a synchronous
caller is interrupted while waiting in `run_sync()` or `iterate_sync()`, the background operation
is cancelled and its cleanup finishes before the interruption propagates. Closing an
`iterate_sync()` iterator waits for its source's asynchronous cleanup. Use `contextlib.closing`
around that iterator when stopping early, and call `run_sync(engine.aclose())` in a `finally`
block to release the engine's resources.

::: listenflow.sync
    options:
      members: [run_sync, iterate_sync, iterate_async]
