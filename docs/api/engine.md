# Engines and health API

These names are available directly from `listenflow`. An engine module implements the native
hooks; the core owns preparation, operation ordering, cancellation, closure, and common health.
Use one engine instance on one event loop. The application interprets capabilities and health
and chooses readiness, recovery, and timeout policy.

## Engine contract

`prepare()` loads and trials the model, retries after failure, and does nothing after success.
Recognition starts on the first read; a concurrent stream raises `EngineStateError`. Model
operations wait in arrival order. `aclose()` cancels active and waiting operations, waits for native
cleanup, and releases resources once. A closed engine cannot be reused. `async with` prepares on
entry and closes on exit, including exceptional exit.

Close a stream explicitly with its `aclose()` or `contextlib.aclosing` when stopping early.
Consumer cancellation remains `asyncio.CancelledError`; engine closure produces `EngineClosedError`.
Capabilities and health can be read while model operations run and after closure.

To collect final transcripts from complete PCM audio, use the core function
[`transcribe(engine, pcm)`](transcription.md). The caller prepares and closes the engine.

Subclass constructors call `super().__init__()`. `_load()` and `_release()` are asynchronous;
`_recognize()` returns an asynchronous iterator. A failed load cleans up its own partial resources.
Hooks map native errors into the error hierarchy and chain their causes. Cancelled hooks wait for
native work to actually finish. `_health_details()` returns recognition fields only, without
`state`, `busy`, or `last_progress_monotonic_s`. `_mark_progress()` records PCM processing and
utterance decode completion. Pass it as `on_progress` when constructing an `UtteranceDecodePipeline`
to give the parent sole ownership of the timestamp.

::: listenflow.base.STTEngineBase
    options:
      members: [capabilities, prepare, stream, aclose, health, _load, _release, _recognize, _health_details, _mark_progress]
      filters: ["!^__"]

## Errors

Engine errors inherit from `STTError` and the meaning's standard Python error. `EngineClosedError`
is an `EngineStateError`. Native errors are chained as causes. Release failure is an `InferenceError`
reported only by the first `aclose()`. Standalone helpers keep their standard validation errors.

::: listenflow.errors
    options:
      members: null
      filters: ["!^_"]

## Capabilities

`languages` is an immutable set of accepted language codes. Empty means the engine has no language
argument and makes no language declaration. `auto_language` states whether `None` requests language
detection. The declaration also keeps the recognition transport, input format, partial mode,
finalization modes, timestamp detail, and diarization fields.

::: listenflow.engine.EngineCapabilities

::: listenflow.engine.EngineTransport

::: listenflow.engine.EngineInputFormat

::: listenflow.engine.PartialMode

::: listenflow.engine.FinalizationMode

::: listenflow.engine.TimestampDetail

::: listenflow.engine.DiarizationMode

## Health

`state` stays `UNPREPARED` while loading and after failure, becomes `READY` after success, and is
`CLOSED` once closing starts. `busy` describes a running model operation, including stream cleanup;
waiting calls alone do not make it true. `last_progress_monotonic_s` is a `time.monotonic()` timestamp
or `None` before progress. PCM processing and utterance decode completion update it, including
silence and recognition without text. Consumer event reads do not update it.

Recognition fields are `processed_pcm_chunks`, `emitted_events`, `bias_application`,
`decoding_utterance_id`, and `pending_decode_utterances`. Health reads held values without I/O,
inference, or waiting for the model operation lock.

::: listenflow.health.EngineState

::: listenflow.health.HealthSnapshot

::: listenflow.health.BiasApplicationSnapshot
