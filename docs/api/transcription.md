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

## One caller-selected utterance

`transcribe_utterance(engine, pcm) -> list[Final]` is a core asynchronous function for audio the
application has selected as one utterance. It calls
`engine.stream(pcm_chunks, segmentation=Segmentation.SINGLE)` and collects the original final
events. The engine must declare `EngineCapabilities.single_utterance=True`; otherwise the request
raises `ConfigurationError` before recognition. Supply 16 kHz mono PCM in the engine's declared
input format, with whole samples (s16le or little-endian float32).

```python
from listenflow import Final, STTEngineBase, transcribe_utterance


async def recognize_utterance(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    async with engine:
        return await transcribe_utterance(engine, pcm)
```

All supplied samples, including internal silence, belong to one decode. Chunk boundaries do not
change the submitted audio. Voice detection and `min_voiced_s` do not filter the input. Empty PCM
skips decoding and emits no events. Non-empty audio without recognized text still closes a single
`UtteranceStart` / `UtteranceEnd` pair, so this function returns `[]` in both cases.

Each call returns new results. Utterance IDs and offsets are relative to that call's PCM, starting
with utterance 1 at audio offset zero. Final offsets retain the recognizer's position within that
PCM; they do not all become zero. To recover the position in a longer recording, add the offset
where the application's retained range starts. Previously returned final events do not change.

The application retains audio and chooses a contiguous range when speech continues across a
previous boundary. The core retains only the audio supplied to the current call. Preparation,
closure, cancellation, and overlapping streams follow the same engine rules as `transcribe()`.
A concurrent request raises `EngineStateError` without disturbing the first stream. Incomplete
PCM samples raise `ConfigurationError`; recognition errors propagate unchanged without returning
partial results. Synchronous callers can use `run_sync(transcribe_utterance(engine, pcm))` on the
same background event loop as preparation and closure.

`transcribe(engine, pcm)` continues to collect finals using the engine's default segmentation.
Use `Segmentation.SINGLE` explicitly on `engine.stream()` when you also need start, partial, and
end events for one utterance. `Segmentation.AUTO` remains the default.

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

::: listenflow.transcription.transcribe_utterance
