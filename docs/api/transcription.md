# Complete PCM transcription API

`transcribe(engine, pcm)` is an asynchronous function available directly from `listenflow`.
Pass a prepared `STTEngineBase` instance and the complete PCM audio as `bytes` in the engine's
accepted input format. The function feeds the PCM to `engine.stream()` and returns a `list[Final]`
containing the original final events in utterance order. Each event retains its utterance ID,
audio offset, text, and speaker label, including when an utterance produces multiple final events.

`Partial`, `UtteranceStart`, and `UtteranceEnd` events are excluded. A stream without final text
returns `[]`. Stream errors propagate unchanged, including a failure after some final events.
The function does not return partial results on failure.

The caller prepares and closes the engine and keeps all its operations on one event loop.
Audio file reading and format conversion belong to the application.

## Asynchronous callers

An asynchronous context prepares the engine before transcription and closes it afterwards:

```python
from listenflow import Final, STTEngineBase, transcribe


async def recognize_pcm(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    async with engine:
        return await transcribe(engine, pcm)
```

A caller that already owns a prepared engine can use `await transcribe(engine, pcm)` repeatedly
between its own preparation and closure. `transcribe()` performs neither operation.

## Synchronous callers

Use the existing `run_sync()` helper for preparation, transcription, and closure on the same
background event loop:

```python
from listenflow import Final, STTEngineBase, run_sync, transcribe


def recognize_pcm_sync(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    try:
        run_sync(engine.prepare())
        return run_sync(transcribe(engine, pcm))
    finally:
        run_sync(engine.aclose())
```

See the [synchronous helper API](sync.md) for cancellation and event loop restrictions.

## Function reference

::: listenflow.transcription.transcribe
