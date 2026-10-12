import importlib
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import listenflow
from listenflow import (
    EngineInputFormat,
    Final,
    OfflineModelDirectorySource,
    Partial,
    UtteranceEnd,
    UtteranceStart,
    has_voice,
)


def _removed_engine_name() -> str:
    return "".join(("Whisper", "Live", "Kit", "Engine"))


def _removed_runtime_module() -> str:
    return "listenflow." + "".join(("whisper", "live", "kit_engine"))


def _removed_transcript_module() -> str:
    return "listenflow." + "".join(("trans", "cript"))


def _removed_model_source_name() -> str:
    return "".join(("Offline", "Model", "Source"))


@pytest.mark.req("SP-LF-016")
def test_removed_engine_and_transcript_surfaces_are_absent() -> None:
    """core は engine 固有の runtime を公開しない。"""
    engine_name = _removed_engine_name()

    assert engine_name not in listenflow.__all__
    with pytest.raises(AttributeError):
        getattr(listenflow, engine_name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(_removed_runtime_module())
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(_removed_transcript_module())


@pytest.mark.req("SP-LF-033")
def test_removed_model_source_is_absent_while_directory_source_remains(tmp_path: Path) -> None:
    """利用側は絶対 directory path を持つ変更できない model source を core から取得できる。"""
    model_source_name = _removed_model_source_name()
    engine_module = importlib.import_module("listenflow.engine")

    assert model_source_name not in listenflow.__all__
    with pytest.raises(AttributeError):
        getattr(listenflow, model_source_name)
    with pytest.raises(AttributeError):
        getattr(engine_module, model_source_name)
    for module_name in ("listenflow", "listenflow.engine"):
        with pytest.raises(ImportError):
            exec(f"from {module_name} import {model_source_name}", {})

    model_path = tmp_path / "speech-model"
    source = OfflineModelDirectorySource(directory_path=str(model_path))
    assert source.directory_path == str(model_path)
    with pytest.raises(FrozenInstanceError):
        source.directory_path = str(tmp_path / "other")
    with pytest.raises(ValueError, match="absolute path"):
        OfflineModelDirectorySource(directory_path="relative/model")


@pytest.mark.req("SP-LF-005")
@pytest.mark.req("SP-LF-007")
def test_transcript_events_are_frozen_and_discriminated_by_type() -> None:
    """発話の四種の通知は区別でき、話者の値は省略できる。"""
    events = [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        Partial(utterance_id=1, ts_offset_s=0.1, text="hello"),
        Final(utterance_id=1, ts_offset_s=0.2, text="hello world"),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.3),
    ]

    assert [event.kind for event in events] == [
        "utterance_start",
        "partial",
        "final",
        "utterance_end",
    ]
    with pytest.raises(Exception):
        events[1].utterance_id = 2

    for event_type in (Partial, Final):
        speaker = inspect.signature(event_type).parameters["speaker"]
        assert speaker.annotation == "int | str | None"
        assert speaker.default is None


@pytest.mark.req("SP-LF-001")
def test_ac_11_14_has_voice_is_a_typed_package_root_export() -> None:
    """利用側は package root の has_voice で PCM の発話有無を判定できる。"""
    assert "has_voice" in listenflow.__all__
    assert listenflow.has_voice is has_voice
    assert str(inspect.signature(has_voice)) == (
        "(chunk: 'bytes', *, input_format: 'EngineInputFormat', voice_threshold: 'float') -> 'bool'"
    )
    assert has_voice(
        b"\x00\x00",
        input_format=EngineInputFormat.PCM_S16LE,
        voice_threshold=0.0,
    )


@pytest.mark.req("SP-LF-051")
def test_sync_helpers_are_package_root_exports() -> None:
    """core は run_sync()、iterate_sync()、iterate_async() を package root で公開する。"""
    sync = importlib.import_module("listenflow.sync")
    for name in ("run_sync", "iterate_sync", "iterate_async"):
        assert name in listenflow.__all__
        assert getattr(listenflow, name) is getattr(sync, name)


@pytest.mark.req("SP-LF-063")
def test_package_root_does_not_publish_language_constants_or_validators() -> None:
    """core の公開名は言語の定数や検証の関数を持たない許可集合と一致する。

    公開名を足す時は、SP-LF-063 の境界に照らして許可集合を改める。
    """
    allowed_public_names = {
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
        "UtteranceBoundarySession",
        "UtteranceBoundarySettings",
        "UtteranceDecodePipeline",
        "UtteranceEnd",
        "UtteranceStart",
        "__version__",
        "annotations",
        "base",
        "engine",
        "errors",
        "events",
        "has_voice",
        "health",
        "importlib",
        "iterate_async",
        "iterate_sync",
        "run_sync",
        "sync",
    }
    public_names = set(listenflow.__all__) | {
        name for name in vars(listenflow) if not name.startswith("_")
    }

    assert public_names == allowed_public_names, (
        "Review public names against SP-LF-063 before updating the allowed set: "
        f"added={public_names - allowed_public_names}, "
        f"missing={allowed_public_names - public_names}"
    )
