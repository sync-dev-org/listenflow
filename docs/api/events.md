# Transcript event API

These names are available directly from `listenflow`. All events carry `utterance_id` and
`ts_offset_s`, an offset in seconds from the start of input audio. Text events also carry an
optional speaker label. Append final text to the committed transcript and use partial text for
provisional display according to the engine's declared partial mode.

::: listenflow.events.TranscriptEvent

::: listenflow.events.UtteranceStart

::: listenflow.events.Partial

::: listenflow.events.Final

::: listenflow.events.UtteranceEnd
