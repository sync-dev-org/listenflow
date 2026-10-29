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

`stream(pcm_chunks, *, segmentation=Segmentation.AUTO)` accepts a keyword-only segmentation
choice. AUTO uses the engine's ordinary boundaries. SINGLE transcribes the entire supplied PCM
as one utterance at end of input, preserving silence and ignoring voice detection and
`min_voiced_s`. `transcribe_utterance(engine, pcm)` collects the final events of this SINGLE stream.
Unsupported SINGLE requests and values other than a `Segmentation` member raise
`ConfigurationError` on the first read, before calling the recognition hook or consuming input.
SINGLE uses the same preparation, exclusion, cancellation, and closure rules as AUTO.

Subclass constructors call `super().__init__()`. `_load()` and `_release()` are asynchronous;
`_recognize()` returns an asynchronous iterator. A failed load cleans up its own partial resources.
Hooks map native errors into the error hierarchy and chain their causes. Cancelled hooks wait for
native work to actually finish. `_health_details()` returns recognition fields only, without
`state`, `busy`, or `last_progress_monotonic_s`. `_mark_progress()` records PCM processing and
utterance decode completion. Pass it as `on_progress` when constructing an `UtteranceDecodePipeline`
to give the parent sole ownership of the timestamp.

### Supporting one caller-selected utterance

Set `single_utterance=True` in the engine's `EngineCapabilities` declaration, and accept the
`segmentation` keyword in `_recognize()`. Engines using the shared pipeline can forward it:

```python
from collections.abc import AsyncIterator
from contextlib import aclosing

from listenflow import Segmentation, TranscriptEvent


async def _recognize(
    self,
    pcm_chunks: AsyncIterator[bytes],
    *,
    segmentation: Segmentation = Segmentation.AUTO,
) -> AsyncIterator[TranscriptEvent]:
    async with aclosing(self._pipeline.stream(pcm_chunks, segmentation=segmentation)) as events:
        async for event in events:
            yield event
```

Use this method in the engine subclass with its configured `_pipeline`. The existing decode
callback receives one `ClosedUtterance` containing all non-empty SINGLE input. The callback's
observation offsets remain relative to the supplied PCM. An engine implementing its own decoding
must honor the same one-decode, empty-input, timestamp, and start/end event behavior.

The default capability is `False`. AUTO invokes `_recognize(pcm_chunks)` without the new keyword,
including when AUTO is passed explicitly to `stream()`. Existing one-argument hooks and capability
constructors continue to work unchanged. Declare support only after the hook implements SINGLE.

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
finalization modes, timestamp detail, and diarization fields. `single_utterance` is a boolean,
defaults to `False`, and states whether the engine supports `Segmentation.SINGLE`.

::: listenflow.engine.EngineCapabilities

::: listenflow.engine.EngineTransport

::: listenflow.engine.EngineInputFormat

::: listenflow.engine.Segmentation

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
