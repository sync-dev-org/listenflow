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
