"""listenflow core library: PCM stream input, transcript events output."""

from __future__ import annotations

import importlib

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from listenflow._version import __version__
from listenflow.engine import (
    ClosedUtterance,
    DiarizationMode,
    EngineCapabilities,
    EngineInputFormat,
    EngineTransport,
    FinalizationMode,
    FinalObservation,
    OfflineModelDirectorySource,
    PartialMode,
    ProvisionalObservation,
    TimestampDetail,
    UtteranceBoundarySettings,
    UtteranceBoundarySession,
    UtteranceDecodePipeline,
    has_voice,
)
from listenflow.base import STTEngineBase
from listenflow.events import Final, Partial, TranscriptEvent, UtteranceEnd, UtteranceStart
from listenflow.health import BiasApplicationSnapshot, EngineState, HealthSnapshot
from listenflow.sync import iterate_async, iterate_sync, run_sync
from listenflow.errors import (
    ConfigurationError,
    EngineClosedError,
    EngineStateError,
    EnvironmentUnavailableError,
    InferenceError,
    ModuleNotInstalledError,
    STTError,
)

__all__ = [
    "BiasApplicationSnapshot",
    "ClosedUtterance",
    "ConfigurationError",
    "DiarizationMode",
    "EngineCapabilities",
    "EngineClosedError",
    "EngineInputFormat",
    "EngineState",
    "EngineStateError",
    "EngineTransport",
    "EnvironmentUnavailableError",
    "Final",
    "FinalObservation",
    "FinalizationMode",
    "HealthSnapshot",
    "InferenceError",
    "ModuleNotInstalledError",
    "OfflineModelDirectorySource",
    "Partial",
    "PartialMode",
    "ProvisionalObservation",
    "STTEngineBase",
    "STTError",
    "TimestampDetail",
    "TranscriptEvent",
    "UtteranceEnd",
    "UtteranceStart",
    "UtteranceBoundarySettings",
    "UtteranceBoundarySession",
    "UtteranceDecodePipeline",
    "__version__",
    "has_voice",
    "iterate_async",
    "iterate_sync",
    "run_sync",
]

# Add engine modules here when they are published on PyPI.
_MODULE_DISTRIBUTIONS: dict[str, str] = {}


def __getattr__(name: str) -> object:
    try:
        distribution = _MODULE_DISTRIBUTIONS[name]
    except KeyError:
        raise AttributeError(name) from None

    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotInstalledError(
            f"why={module_name} is unavailable because {distribution} is not installed; "
            f"what={distribution} provides {module_name}; "
            f"how=pip install {distribution}"
        ) from exc
    globals()[name] = module
    return module
