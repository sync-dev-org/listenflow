# Utterance boundary API

These names are available directly from `listenflow`. The PCM helpers accept 16 kHz mono raw
PCM in signed 16-bit little-endian or little-endian float32 format. Every chunk must contain whole
samples. Voice thresholds use normalized full-scale amplitude and must be finite `float` values
between `0.0` and `1.0`.

## Voice detection and settings

::: listenflow.engine.has_voice

::: listenflow.engine.UtteranceBoundarySettings

## Streaming observations

::: listenflow.engine.ProvisionalObservation

::: listenflow.engine.FinalObservation

::: listenflow.engine.UtteranceBoundarySession
    options:
      members: null
      filters: ["!^_"]

## Utterance decoding

The decoding callback receives a closed utterance and returns provisional or final observations.
Observation offsets use seconds from the start of input audio, including any preroll. The pipeline
receives further PCM while decoding and emits events in utterance order.

`stream(pcm_chunks, *, segmentation=Segmentation.AUTO)` uses automatic voice and silence boundaries
by default. `Segmentation` is exported from `listenflow`. In SINGLE mode, the pipeline collects
every sample through end of input and calls the decoder exactly once for non-empty PCM. Internal
silence and chunk boundaries do not split the utterance. Voice detection, preroll selection, and
`min_voiced_s` do not filter or trim the submitted range.

The first non-empty chunk produces `UtteranceStart(1, 0.0)`. The decoder receives a `ClosedUtterance`
with ID 1, start offset 0.0, and end offset equal to the complete input duration at 16 kHz. Decoding
begins only after input ends. Observations produce the ordinary partial and final events, and
`UtteranceEnd` closes the one pair at the complete input duration. Without recognized text, only
the start/end pair is emitted. Zero-length input, including a sequence of empty chunks, is not
decoded and emits no events. Each stream starts IDs and offsets again from its own PCM.

Both modes validate complete s16le or float32 samples. Invalid segmentation or incomplete samples
raise `ValueError` on this standalone helper; `STTEngineBase` reports invalid stream configuration
and samples as `ConfigurationError`. An engine module supporting SINGLE declares
`EngineCapabilities.single_utterance=True` and forwards `_recognize()`'s segmentation keyword to
the pipeline. See the [engine hook example](engine.md#supporting-one-caller-selected-utterance).

Normal input completion drains received utterances. Input or decoder failure and early consumer
closure cancel the active decode and wait for its cleanup before allowing reuse. Pending decodes
and undelivered events are discarded on failure. The first input or decode exception is returned
as the original object, even if cleanup fails later.

`on_progress` optionally receives a synchronous notification after processing a PCM chunk or
completing an utterance decode, including silence and recognition without text. In an engine module,
pass `on_progress=self._mark_progress` when constructing the pipeline and return its recognition
counters through `_health_details()`. With the callback, the parent owns the progress timestamp
and the pipeline's `last_progress_monotonic_s` stays `None`. A standalone pipeline without the
callback records its own `last_progress_monotonic_s`. Event delivery does not advance this value.

::: listenflow.engine.ClosedUtterance

::: listenflow.engine.UtteranceDecodePipeline
    options:
      members: null
      filters: ["!^_"]
