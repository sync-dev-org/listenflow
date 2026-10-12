"""Errors shared by recognition engines and their callers."""


class STTError(Exception):
    """Base class for recognition engine errors."""


class ConfigurationError(STTError, ValueError):
    """An engine setting or input value is invalid."""


class EngineStateError(STTError, RuntimeError):
    """The engine cannot accept an operation in its current state."""


class EngineClosedError(EngineStateError):
    """The engine is closed or has cancelled an operation while closing."""


class InferenceError(STTError, RuntimeError):
    """An engine operation failed at runtime."""


class EnvironmentUnavailableError(STTError, RuntimeError):
    """The environment lacks a runtime component or device the engine needs."""


class ModuleNotInstalledError(STTError, ImportError):
    """The requested engine module is not installed."""
