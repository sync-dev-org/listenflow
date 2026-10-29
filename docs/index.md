# listenflow

`listenflow` converts a stream of clean, 16 kHz mono PCM audio into asynchronous transcript events.
The core supplies engine contracts, shared utterance boundaries, and health snapshots. It has no
runtime dependencies and does not include a speech recognition model.

Engine modules are separate distributions and will be published individually as the needed modules
become ready. Supply an engine implementing `STTEngineBase` to perform recognition. The application
owns audio capture and cleanup, network transport, a persistent server, process supervision,
readiness, and recovery policy. Speech synthesis, dialogue control, and LLM calls also belong to
the application.

## Install

Use Python 3.12 or 3.13. `listenflow` is licensed under MIT.

```console
pip install listenflow
```

## Use an engine

Use one engine instance on one event loop. An asynchronous context prepares the model on entry
and closes the engine on exit, including when the body raises an exception:

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

The application supplies both arguments: an engine instance and an asynchronous source of audio
chunks. Match the chunks to `engine.capabilities.input_format`. The shared PCM helpers accept
`EngineInputFormat.PCM_S16LE` (signed 16-bit little-endian) and `EngineInputFormat.PCM_F32`
(little-endian float32). Both represent 16 kHz mono audio; every chunk must contain whole samples.

### Complete PCM input

Use `transcribe(engine, pcm)` when the complete PCM audio is already held as `bytes`:

```python
from listenflow import Final, STTEngineBase, transcribe


async def recognize_pcm(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    async with engine:
        return await transcribe(engine, pcm)
```

The function consumes `engine.stream()` and returns only its `Final` events in utterance order,
preserving their IDs, timestamps, text, and speaker labels. It returns `[]` when no final text is
emitted and propagates stream errors unchanged. The caller prepares and closes the engine and
supplies PCM matching its input format. Audio file reading and format conversion belong to the
application. See the [complete PCM transcription API](api/transcription.md).

For audio the application has already selected as one utterance, use
`await transcribe_utterance(engine, pcm)`. This core function requests
`engine.stream(pcm_chunks, segmentation=Segmentation.SINGLE)` and returns its final events.
`Segmentation` is available from `listenflow`; `AUTO` remains the default for ordinary streams.
The engine must declare `EngineCapabilities.single_utterance=True` (the default is `False`), or
the request raises `ConfigurationError` before recognition.

SINGLE submits all audio once at end of input, regardless of silence or chunk boundaries, and
bypasses voice detection and `min_voiced_s`. Empty PCM skips decoding and returns `[]`; audio
without text also returns `[]`. Results keep offsets from this call's PCM and do not change earlier
final events. The application retains audio and chooses the contiguous range to resubmit when
speech continues. The caller prepares and closes the engine, and concurrent streams are rejected
without disturbing the first. `transcribe()` keeps the engine's default segmentation behavior.

### Synchronous callers

For synchronous code, call `run_sync(engine.prepare())`, read
`iterate_sync(engine.stream(iterate_async(pcm_chunks)))`, and finish with
`run_sync(engine.aclose())`. These operations use the library's shared background event loop,
keeping the engine on one event loop. `run_sync()` waits for an awaitable; `iterate_sync()` turns
an asynchronous iterable into a synchronous iterator; `iterate_async()` reads a synchronous
iterable in worker threads and yields its values asynchronously.

The helpers can be called from multiple threads and inside a running event loop. If the caller
is interrupted while waiting, `run_sync()` and `iterate_sync()` cancel the background operation
and wait for cleanup before propagating the interruption. Use `contextlib.closing` around the
synchronous iterator to wait for cleanup when stopping early, and `try`/`finally` to close the
engine. See the [synchronous helper API](api/sync.md).
For complete PCM input, call `run_sync(transcribe(engine, pcm))` after preparation and before
closure on that same background event loop.

## Transcript events

Each event carries an utterance ID, an audio offset in seconds from the start of input, and a kind.

| Event | Kind | Content |
| --- | --- | --- |
| `UtteranceStart` | `utterance_start` | Start of an utterance. |
| `Partial` | `partial` | Provisional text and an optional speaker label. |
| `Final` | `final` | Committed text and an optional speaker label. |
| `UtteranceEnd` | `utterance_end` | End of an utterance. |

`TranscriptEvent` is the union of these four types. Interpret provisional text according to
`engine.capabilities.partial_mode`. Append `Final` events to the committed transcript; committed
text is not revised. An utterance may produce more than one `Final` event.

See the [transcript event API](api/events.md).

## Capabilities and health

`EngineCapabilities` declares accepted `languages` as a `frozenset[str]`, `auto_language`,
transport, input format, provisional output mode, finalization modes, timestamp detail, and speaker
labeling. Its values are immutable and readable before preparation and after closure.
`single_utterance` declares support for `Segmentation.SINGLE` and defaults to `False`.

`HealthSnapshot` reports `state` (`EngineState.UNPREPARED`, `READY`, or `CLOSED`), `busy`, and
`last_progress_monotonic_s`. Loading or a failed preparation remains `UNPREPARED`. Closing starts
the `CLOSED` state. Only a running model operation makes `busy` true. The timestamp advances after
PCM processing and utterance decode completion, including silence and empty recognition; reading
events does not change it. Recognition fields include processed PCM chunks, emitted events, active
decode, pending utterances, and an optional `BiasApplicationSnapshot` recording prompt application.
These synchronous reads use held values without I/O or waiting for a model operation.

See the [engine and health API](api/engine.md).

## Operation lifetime and errors

`prepare()` loads and trials the model once after success and can be retried after failure.
Model operations wait in arrival order; callers can cancel a waiting operation. Recognition starts
on the first event read. Concurrent streams are rejected with `EngineStateError`, including while
the first stream is cleaning up.

`aclose()` cancels active and waiting work, waits for native cleanup, and releases resources once.
Overlapping close calls wait for release. Only the first close reports a release failure, as
`InferenceError` chained to the original error. Further closes finish normally. A closed engine
cannot be reused, but its capabilities and health remain readable. Use `async with` for preparation
and closure, and `contextlib.aclosing` or the iterator's `aclose()` when stopping a stream early.
Consumer cancellation stays `asyncio.CancelledError`; engine closure cancels operations with
`EngineClosedError`. Timeouts are supplied by the caller.

`STTError` is the base for `ConfigurationError` (`ValueError`), `EngineStateError` (`RuntimeError`),
`EngineClosedError` (an `EngineStateError`), `InferenceError` (`RuntimeError`),
`EnvironmentUnavailableError` (`RuntimeError`), and `ModuleNotInstalledError` (`ImportError`).
Environment problems are reported during preparation. `ModuleNotInstalledError` provides installation
guidance and chains the original import error when accessing a missing module listed in the core's
module table as `listenflow.<name>`. The table contains only engine modules published on PyPI.

Import an installed module outside that table explicitly with `import listenflow.<name>`.
Before that import, accessing an unlisted module as `listenflow.<name>` raises `AttributeError`.
If the module is missing, the explicit import raises Python's standard `ModuleNotFoundError`,
whose `name` identifies the missing module. Standalone helper validation keeps standard errors.
Recognition without text remains a normal utterance start/end pair.

## Build an engine module

Subclass `STTEngineBase` in a separate distribution that depends on `listenflow`, and call
`super().__init__()` in the constructor. The parent owns the public operations and common health:

- `capabilities` returns the static `EngineCapabilities` declaration.
- `async _load()` loads resources and runs trial inference; an unsuccessful load cleans up its own resources.
- `async _release()` returns resources from a successful load.
- `_recognize(pcm_chunks)` returns an asynchronous iterator of `TranscriptEvent` values.
- `_health_details()` returns a mapping of recognition fields; it must omit the three common fields and defaults to an empty mapping.

Hooks finish native work and close owned iterators before returning from cancellation or early
closure. They map native and upstream errors into the corresponding `STTError` subclass with the
original error as the cause. Call `_mark_progress()` after a PCM chunk is processed or an utterance
is decoded. When using a pipeline, pass `on_progress=self._mark_progress` in the engine constructor;
the pipeline forwards notifications and keeps no duplicate timestamp. Return its counters and
decode fields through `_health_details()`.

The module owns model loading, recognition, runtime dependencies, configuration, and supported
platforms. Use the shared utterance helpers when they match the engine's recognition interface.

To support SINGLE, declare `single_utterance=True`, accept the keyword-only
`segmentation=Segmentation.AUTO` argument in `_recognize()`, and forward it to
`pipeline.stream(pcm_chunks, segmentation=segmentation)`. Close the pipeline iterator when the
consumer stops early. The parent omits the keyword for AUTO, so existing one-argument hooks work
unchanged. See the [engine hook example](api/engine.md#supporting-one-caller-selected-utterance).

## Utterance boundaries

Voice detection measures RMS over fixed 256-sample (16 ms) blocks at 16 kHz. A block is voiced
when its RMS reaches `voice_rms_threshold`, which defaults to `0.01` (about -40 dBFS). This value
is normalized full-scale RMS amplitude: s16le samples are divided by 32768, and f32le values are
used directly. Supply a finite `float` greater than `0.0` and at most `1.0`.

`has_voice(chunk, *, input_format=..., voice_rms_threshold=...)` returns true when any block
in its chunk is voiced. This helper is stateless: each call starts its own blocks. A final block
shorter than 256 samples uses its actual sample count, without zero padding. Sessions and pipelines
instead retain incomplete blocks across chunks, so chunk sizes and empty chunks do not change
the stream's block alignment or voice decisions.

`UtteranceBoundarySession` tracks audio offsets and converts `ProvisionalObservation` and
`FinalObservation` records into transcript events. Call `observe_pcm()` as audio arrives, feed the
recognizer's observations, and call `finish_input()` followed by `finalize_events()` at the end of
input. `finish_input()` returns `None`; if the last short block starts an utterance, its start
event is returned by `finalize_events()` before its end event. If a non-empty recognition observation
arrives first, its returned events begin with that held start. `maybe_end_event()` waits until
the start has been delivered. This direct path fits engines that
already stream recognition observations and leaves closing an utterance to `maybe_end_event()`.
Use the pipeline for chunk-independent segmentation when a chunk can contain several utterances.

`UtteranceDecodePipeline` fits engines that decode complete utterances. Configure silence, voice
threshold, preroll, and the minimum voiced span with `UtteranceBoundarySettings`, and supply an async
decoding callback. In AUTO mode, `min_voiced_s` defaults to `0.0`, which decodes every utterance.
A positive value skips decoding utterances whose voiced span is shorter than that many seconds, including utterances
closed at end of input. The span runs from the beginning of the first voiced block to the end of the
last voiced block, including any silence between them, and excludes preroll and trailing silence.
Skipped utterances emit only `UtteranceStart` and `UtteranceEnd`; the start event still arrives as
soon as the first voiced block is available. The example below uses a minimum voiced span of 0.2
seconds:

```python
from collections.abc import Awaitable, Callable, Iterable

from listenflow import (
    ClosedUtterance,
    EngineInputFormat,
    FinalObservation,
    ProvisionalObservation,
    UtteranceBoundarySettings,
    UtteranceDecodePipeline,
)


def make_pipeline(
    decode_utterance: Callable[
        [ClosedUtterance],
        Awaitable[Iterable[ProvisionalObservation | FinalObservation]],
    ],
) -> UtteranceDecodePipeline:
    return UtteranceDecodePipeline(
        boundary_settings=UtteranceBoundarySettings(
            silence_threshold_s=0.8,
            voice_rms_threshold=0.01,
            preroll_s=0.3,
            min_voiced_s=0.2,
        ),
        decode_utterance=decode_utterance,
        input_format=EngineInputFormat.PCM_S16LE,
    )
```

`silence_threshold_s` counts subthreshold audio after the last voiced block. The silence duration
is rounded up to blocks: `0.8` seconds closes after exactly 50 silent blocks, while `0.05` seconds
closes after four blocks (0.064 seconds). Audio passed to decoding ends at that closing block.
At end of input, an open utterance closes at the actual input end without adding silence.

The old peak threshold `voice_threshold` has been removed. Passing it to settings, sessions, or
`has_voice` raises `TypeError` with guidance to set `voice_rms_threshold` in normalized full-scale
RMS units. Replace the keyword and select an RMS value, such as the default `0.01`; do not copy
an old peak value unchanged, since peak and RMS have different meanings.
Settings and session constructors accept only `silence_threshold_s` positionally; all following
fields require keywords. A second positional argument also raises the migration `TypeError`,
so an old positional peak threshold cannot be interpreted as RMS silently.

The callback performs recognition on `ClosedUtterance.pcm`. A `FinalObservation.source_offset_s`
uses seconds from the beginning of the input audio, so convert any recognizer-relative offset
using the closed utterance's `start_offset_s`. The pipeline continues receiving PCM while decoding,
decodes utterances sequentially, and emits their events in utterance order. Iterate over
`pipeline.stream(pcm_chunks)` to consume the events. Normal end of input drains pending decoding;
decoder and input errors propagate to the consumer. Failure or early closure cancels an active
decode and waits for cleanup before reuse. When the optional `on_progress` callback is omitted,
the standalone pipeline records `last_progress_monotonic_s` after PCM processing and decode
completion. With the callback, the receiver owns the timestamp and the pipeline leaves it `None`.
Event reading does not advance progress.

Passing `segmentation=Segmentation.SINGLE` to the pipeline collects all PCM through end of input
and decodes it once. It bypasses voice detection and `min_voiced_s`, preserves every sample, and
closes a single start/end pair even when no text is recognized. Empty input emits nothing and is
not decoded. Event handling and cleanup are shared with AUTO.

See the [utterance boundary API](api/utterances.md).

## Offline models

Use `OfflineModelDirectorySource` to identify a model directory with an absolute path:

```python
from pathlib import Path

from listenflow import OfflineModelDirectorySource

model_source = OfflineModelDirectorySource(str(Path("models/speech").resolve()))
```

An engine that accepts this value defines the directory layout and validates its contents. The
application provisions the files; the core does not download models. See the
[offline model API](api/models.md).

## API reference

- [Engines, capabilities, and health](api/engine.md)
- [Complete PCM transcription](api/transcription.md)
- [Synchronous helpers](api/sync.md)
- [Transcript events](api/events.md)
- [Utterance boundaries and decoding](api/utterances.md)
- [Offline models](api/models.md)
