# Utterance boundary API

These names are available directly from `listenflow`. The PCM helpers accept 16 kHz mono raw
PCM in signed 16-bit little-endian or little-endian float32 format. Every chunk must contain whole
samples. `voice_rms_threshold` uses normalized full-scale RMS amplitude, defaults to `0.01`
(about -40 dBFS), and must be a finite `float` greater than `0.0` and at most `1.0`.

## Voice detection and settings

Voice detection splits the input from its first sample into 256-sample (16 ms) blocks. A block
is voiced when the square root of its mean squared normalized samples reaches the threshold.
s16le samples are divided by 32768; f32le samples are used directly. There is no noise estimate,
normalization, or learned voice model. Sessions and pipelines carry incomplete blocks between
chunks. Their last short block uses its actual sample count at end of input, without zero padding.
Empty chunks do not end a block. Block alignment and voice decisions are independent of chunk sizes.

`has_voice` is stateless and returns true if any block within the supplied chunk is voiced.
Each call starts a fresh set of blocks and treats its final short block as the end of that input;
use a session or pipeline to preserve alignment across a stream.

The former peak keyword `voice_threshold` is rejected with `TypeError` by all three entry points:
`UtteranceBoundarySettings`, `UtteranceBoundarySession`, and `has_voice`. Use
`UtteranceBoundarySettings(voice_rms_threshold=0.01)`, and pass the same keyword to sessions or
`has_voice`. Choose a normalized full-scale RMS threshold; the previous peak value is not converted
and should not be reused as an equivalent RMS value.
Only `silence_threshold_s` can be positional in settings or session construction. Supply all
following fields by keyword. A second positional argument raises the migration `TypeError` too,
preventing a former positional peak value from becoming an RMS threshold silently.

::: listenflow.engine.has_voice

::: listenflow.engine.UtteranceBoundarySettings

## Streaming observations

`observe_pcm()` measures complete blocks and retains any remainder for the next chunk. The caller
feeds recognition observations and calls `maybe_end_event()` to close the current utterance.
This direct session API leaves closure timing to the caller. Use `UtteranceDecodePipeline` for
chunk-independent segmentation of several utterances within one chunk.

At input end, call `finish_input()` and then drain `finalize_events()`. `finish_input()` returns
`None`. If the last short block contains the first voice, `finalize_events()` returns its
`UtteranceStart` followed by its `UtteranceEnd`. If a non-empty recognition observation arrives
first, its returned events begin with that held start; `maybe_end_event()` waits until the start
has been delivered. An open utterance closes at the actual input end, without adding silence;
a provisional observation timestamp beyond input end remains a lower bound. An end already
confirmed by silence retains its timestamp.

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

AUTO closes at the first block boundary at or beyond `silence_threshold_s` of subthreshold audio
after the last voiced block, rounding the silence duration up to blocks. The default `0.8` seconds
is exactly 50 blocks. The PCM sent to decoding ends at the closing block. At end of input, any open
utterance closes at the actual input end. `min_voiced_s` measures the first voiced block's start
through the last voiced block's end, including intervening silence and excluding preroll and
trailing silence. Preroll retains the configured number of samples before the first voiced block,
limited by input start and the previous utterance's end.

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
