"""Collect committed transcripts from a complete PCM input."""

from __future__ import annotations

from collections.abc import AsyncIterator

from listenflow.base import STTEngineBase
from listenflow.engine import Segmentation
from listenflow.events import Final


async def transcribe(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    """Return the engine's final transcript events in utterance order.

    Args:
        engine: An engine prepared by the caller on the current event loop.
        pcm: Complete audio in the engine's accepted input format.

    Returns:
        The original Final events, preserving their IDs, timestamps, text, and
        speaker labels. Returns an empty list when the stream emits no Final.

    Stream errors propagate unchanged. The caller owns preparation and closure
    of the engine. No audio conversion or file reading is performed.
    """

    async def pcm_chunks() -> AsyncIterator[bytes]:
        yield pcm

    return [event async for event in engine.stream(pcm_chunks()) if isinstance(event, Final)]


async def transcribe_utterance(engine: STTEngineBase, pcm: bytes) -> list[Final]:
    """Transcribe the entire PCM as one utterance, preserving internal silence.

    The caller prepares and closes the engine and retains any audio to join.
    The engine must declare single_utterance support; otherwise the stream raises
    ConfigurationError before recognition. Input uses the engine's PCM format.
    Offsets and utterance IDs are relative to this input, and previous results
    are unchanged. Voice detection and min_voiced_s do not filter this input.
    Empty PCM skips decoding. Input with no final text returns an empty list.
    Stream errors propagate unchanged; no partial results are returned on failure.
    """

    async def pcm_chunks() -> AsyncIterator[bytes]:
        yield pcm

    return [
        event
        async for event in engine.stream(pcm_chunks(), segmentation=Segmentation.SINGLE)
        if isinstance(event, Final)
    ]
