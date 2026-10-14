"""Collect committed transcripts from a complete PCM input."""

from __future__ import annotations

from collections.abc import AsyncIterator

from listenflow.base import STTEngineBase
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
