# listenflow

Convert a stream of clean, 16 kHz mono PCM audio into asynchronous transcript events. `listenflow`
provides engine-neutral contracts, shared utterance boundaries, and health snapshots for speech
recognition engines.

The core package has no runtime dependencies and does not include a speech recognition model.
Engine modules are separate distributions and will be published individually as the needed modules
become ready. To transcribe audio, supply an engine that implements `STTEngineBase`.

The application supplies audio capture, audio cleanup, network transport, a persistent server, and
process supervision. It also decides readiness and recovery policy, and owns speech synthesis,
dialogue control, and LLM calls.

## Install

Requires Python 3.12 or 3.13. Licensed under
[MIT](https://github.com/sync-dev-org/listenflow/blob/main/LICENSE).

```console
pip install listenflow
```

## Use an engine

An engine exposes `capabilities`, loads and trials its model with `prepare()`, accepts an
`AsyncIterator[bytes]` through `stream()`, and reports progress through `health()`. Use `async with`
to prepare it on entry and release its resources on exit. Use one engine instance on one event loop.

```python
from collections.abc import AsyncIterator

from listenflow import Final, STTEngineBase


async def recognize_stream(engine: STTEngineBase, pcm_chunks: AsyncIterator[bytes]) -> None:
    async with engine:
        async for event in engine.stream(pcm_chunks):
            if isinstance(event, Final):
                print(event.utterance_id, event.ts_offset_s, event.text)

    snapshot = engine.health()
    print(snapshot.processed_pcm_chunks, snapshot.emitted_events)
```

Pass an engine instance and an audio source to `recognize_stream`. The core installation supplies the
contract; the engine supplies recognition. For the shared PCM helpers, select
`EngineInputFormat.PCM_S16LE` for signed 16-bit little-endian samples or `EngineInputFormat.PCM_F32`
for little-endian float32 samples. Both use 16 kHz mono audio, and each chunk must contain whole
samples. Match the audio representation to the engine's declared input format.

### Complete PCM input

For audio already held as `bytes`, use `transcribe(engine, pcm)` to collect the engine's `Final`
events in a `list[Final]`, in utterance order:

```python
from listenflow import Final, STTEngineBase, transcribe


async def recognize_pcm(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    async with engine:
        return await transcribe(engine, pcm)
```

The caller prepares and closes the engine; `transcribe()` only consumes its `stream()`.
Supply complete PCM matching the engine's input format. The function preserves each final event's
utterance ID, timestamp, text, and speaker label, excludes other event types, and returns `[]`
when no final text is emitted. Stream errors propagate unchanged. The application reads audio
files and converts formats before calling the function.

See the [complete PCM transcription API](https://sync-dev-org.github.io/listenflow/api/transcription/).

### Synchronous callers

Use the core's three helpers to call an engine from synchronous code:

- `run_sync()` runs an awaitable on the library's shared background event loop and waits for its result.
- `iterate_sync()` converts an asynchronous iterable into a synchronous iterator.
- `iterate_async()` converts a synchronous iterable into an asynchronous iterator, reading the input in worker threads.

Operations passed through `run_sync()` and `iterate_sync()` use the same background event loop,
so the following preparation, recognition, and closure keep the engine on one event loop:

```python
from collections.abc import Iterable
from contextlib import closing

from listenflow import Final, STTEngineBase, iterate_async, iterate_sync, run_sync


def transcribe_sync(engine: STTEngineBase, pcm_chunks: Iterable[bytes]) -> None:
    try:
        run_sync(engine.prepare())
        with closing(iterate_sync(engine.stream(iterate_async(pcm_chunks)))) as events:
            for event in events:
                if isinstance(event, Final):
                    print(event.utterance_id, event.ts_offset_s, event.text)
    finally:
        run_sync(engine.aclose())
```

The helpers support callers in multiple threads and inside an already running event loop.
Calling `run_sync()` or `iterate_sync()` from the shared background event loop thread raises
`EngineStateError`; use `await` or `async for` there instead.
Keep all asynchronous operations on a given engine on the same event loop. If a synchronous
caller is interrupted while waiting in `run_sync()` or `iterate_sync()`, the helper cancels the
background operation and waits for its cleanup before propagating the interruption. Closing an
`iterate_sync()` iterator also waits for asynchronous cleanup; `contextlib.closing` ensures this
happens when the loop stops early or its body raises an exception.

See the [synchronous helper API](https://sync-dev-org.github.io/listenflow/api/sync/).
For complete PCM input, use `run_sync(transcribe(engine, pcm))` between
`run_sync(engine.prepare())` and `run_sync(engine.aclose())` on the same background event loop.

## Transcript events

Every event has an `utterance_id`, a `ts_offset_s` measured in seconds from the start of the input
audio, and a `kind`.

| Event | Kind | Meaning |
| --- | --- | --- |
| `UtteranceStart` | `utterance_start` | An utterance has started. |
| `Partial` | `partial` | Provisional text, with an optional speaker label. |
| `Final` | `final` | Committed text, with an optional speaker label. |
| `UtteranceEnd` | `utterance_end` | The utterance has ended. |

Use `Partial` for a provisional display according to the engine's `partial_mode`. Append `Final`
events to the committed transcript; committed text is not revised. An utterance can contain more
than one `Final` event. `TranscriptEvent` is the union of the four event types.

## Capabilities and health

`EngineCapabilities` is an immutable, static declaration that can be read before preparation
and after closure:

| Field | Describes |
| --- | --- |
| `languages` | Accepted language codes, as a `frozenset[str]`. An empty set means the engine has no language argument and declares no languages. |
| `auto_language` | Whether a language of `None` selects automatic detection. |
| `transport` | In-process, WebSocket, gRPC, or HTTP per utterance. |
| `input_format` | The audio representation accepted by the engine. |
| `partial_mode` | Replace, append, commit-only, or no provisional output. |
| `finalization` | One or more ways the engine commits text or closes an utterance. |
| `timestamp_detail` | No timestamps, or segment, word, token, or sample offsets. |
| `diarization` | No speaker labels, segment labels, word labels, or mutable spans. |

`HealthSnapshot` reports held values immediately without I/O or waiting for model operations:

| Field | Meaning |
| --- | --- |
| `state` | `EngineState.UNPREPARED`, `READY`, or `CLOSED`. Loading and failed preparation remain `UNPREPARED`. Closing starts the `CLOSED` state. |
| `busy` | A model operation is in progress, including recognition cleanup. Waiting calls alone do not make it true. |
| `last_progress_monotonic_s` | Time in `time.monotonic()` seconds of the last processed PCM chunk or completed utterance decode; `None` before any progress. Event delivery does not update it. |
| `processed_pcm_chunks`, `emitted_events` | Recognition counters. |
| `decoding_utterance_id`, `pending_decode_utterances` | Current utterance decode and number of waiting utterances. |
| `bias_application` | Optional `BiasApplicationSnapshot` describing recognition prompt application. |

The application interprets these values and decides when an engine is ready or stalled.

## Operation lifetime and errors

`prepare()` loads the model and performs trial inference. A second successful preparation does
nothing; a failed preparation can be retried. Environment failures are reported by `prepare()`.
Model operations wait in arrival order, and a waiting caller can cancel its operation. Recognition
starts on the first event read. A second stream started during recognition or its cleanup raises
`EngineStateError` without disturbing the first stream.

`aclose()` cancels active and waiting operations, waits for native work to finish, then releases
resources once. The closed engine cannot be prepared or used for recognition again. Repeated
`aclose()` calls finish normally; overlapping calls wait for resource release. If release fails,
only the first call raises `InferenceError` with the original failure as its cause. Capabilities
and health remain readable after closure.

When stopping before the end of a stream, explicitly await the iterator's `aclose()` or use
`contextlib.aclosing`. Consumer cancellation and early closure deliver cancellation to inference
and wait for its cleanup before the engine can be reused. Native work may need to run to completion.
Caller cancellation remains `asyncio.CancelledError`; cancellation by the engine's `aclose()` and
operations after closure raise `EngineClosedError`. The caller supplies timeouts when needed.

All engine errors inherit from `STTError` and also support the corresponding standard handler:

| Error | Standard base | Meaning |
| --- | --- | --- |
| `ConfigurationError` | `ValueError` | Invalid engine configuration or input, including incomplete PCM samples. |
| `EngineStateError` | `RuntimeError` | Recognition before preparation or a concurrent stream. |
| `EngineClosedError` | `EngineStateError` | An operation on a closed engine or cancelled by engine closure. |
| `InferenceError` | `RuntimeError` | Runtime failure. |
| `EnvironmentUnavailableError` | `RuntimeError` | Missing native components, libraries, or devices. |
| `ModuleNotInstalledError` | `ImportError` | Attribute access to a missing module listed in the core's module table; includes installation guidance. |

The core's module table contains only engine modules published on PyPI.
Import an installed module outside that table explicitly with `import listenflow.<name>`.
Before that import, accessing an unlisted module as `listenflow.<name>` raises `AttributeError`.
If the module is missing, the explicit import raises Python's standard `ModuleNotFoundError`,
whose `name` identifies the missing module.

Standalone PCM, utterance, model-source, and capability helpers retain their standard validation
errors. An utterance without recognized text remains a normal start/end event pair.

## Build an engine module

Subclass `STTEngineBase` in a separate package and call `super().__init__()` in your constructor.
The parent implements `prepare()`, `stream()`, `aclose()`, `async with`, and common health fields.
The module owns its recognition runtime, configuration, and supported platforms and implements:

| Hook | Responsibility |
| --- | --- |
| `capabilities` | Return the static `EngineCapabilities` property. |
| `async _load()` | Load resources and run trial inference. Clean up resources from an unsuccessful load. |
| `async _release()` | Return resources from a successful load. |
| `_recognize(pcm_chunks)` | Return an asynchronous transcript iterator. Finish native work and close owned iterators before returning from cancellation or early closure. |
| `_health_details()` | Return a mapping of recognition fields only; the default is empty. |

Call the parent's `_mark_progress()` after processing a PCM chunk or completing an utterance decode.
Map native and upstream errors into the appropriate `STTError` subclass and chain the original
exception as the cause. A hook that cannot interrupt native inference must wait for it to finish.

For an engine that decodes complete utterances, use `UtteranceDecodePipeline` to share audio
segmentation and transcript event handling. Pass `on_progress=self._mark_progress` from the engine's
constructor to keep the progress timestamp in the parent, and return the pipeline's recognition
counters and decode fields from `_health_details()`. For an engine with its own streaming
recognition, use `UtteranceBoundarySession` to translate PCM and recognition observations into events.

## Utterance boundaries

- `has_voice` checks whether any sample reaches a normalized full-scale amplitude threshold.
  The threshold must be a finite `float` between `0.0` and `1.0`; the helper is amplitude-based.
- `UtteranceBoundarySettings` configures the silence threshold, voice threshold, preroll, and
  minimum voiced span (`min_voiced_s`, default `0.0`). The pipeline skips decoding shorter
  utterances and emits only `UtteranceStart` and `UtteranceEnd`; the start event remains immediate.
- `UtteranceBoundarySession` tracks audio offsets and utterance IDs. Feed PCM through
  `observe_pcm()` and recognition results through `observe_provisional()` or `observe_final()`.
  Call `finish_input()` when input ends, then drain `finalize_events()`.
- `UtteranceDecodePipeline` receives PCM while decoding closed utterances in order. Its async
  `decode_utterance` callback receives a `ClosedUtterance` and returns an iterable of
  `ProvisionalObservation` or `FinalObservation` records. `stream()` yields transcript events
  in utterance order and completes outstanding decoding when input ends normally. Failure or early
  closure cancels an active decode and waits for its cleanup before allowing reuse.
- `on_progress` optionally forwards PCM-processing and decode-completion notifications to an engine.
  With this callback, the pipeline stores no timestamp. Used on its own, the pipeline records
  `last_progress_monotonic_s`. Reading events does not advance progress.

See the [utterance API](https://sync-dev-org.github.io/listenflow/api/utterances/) for the session and
pipeline interfaces.

## Offline models

`OfflineModelDirectorySource` identifies a model directory with an absolute path:

```python
from pathlib import Path

from listenflow import OfflineModelDirectorySource

model_source = OfflineModelDirectorySource(str(Path("models/speech").resolve()))
```

Pass this value to an engine that accepts an offline model source. The engine defines the directory
layout and validates its contents. The application provisions the files; the core does not download
models.

## Documentation

The [documentation guide](https://sync-dev-org.github.io/listenflow/) links to the API reference for
engines, complete PCM transcription, synchronous helpers, transcript events, utterance boundaries,
and offline models.

## Development

Run from the repository root:

```console
uv sync
uv lock --check
uv run pytest
uv run ruff check
uv run ruff format --check
uv build
uv run --group docs mkdocs build --strict
```
