from __future__ import annotations

import email
import os
import re
import runpy
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROLLBACK_TEST_EXCLUSIONS = ("--ignore", "--ignore-glob")
MOVED_TRACKED_PREFIXES = (
    "benchmarks/",
    "scripts/release/",
    "tests/fixtures/",
    "tests/test_stt_eval_",
)
MOVED_TRACKED_PATHS = {
    "tests/test_workspace_packaging.py",
    "tests/test_release_tooling.py",
}
PROHIBITED_DOCUMENTATION_TERMS = (
    "git clone --branch main",
    "nested workspace",
    "共有 lock",
    "composite identity",
    "#subdirectory=packages/listenflow",
)


def _private_distribution_checks() -> dict[str, list[str]]:
    path = REPOSITORY_ROOT / "tools/private_distribution_checks.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if (REPOSITORY_ROOT / "AGENTS.md").exists() or (REPOSITORY_ROOT / ".nf").exists():
            raise
        pytest.skip("the public snapshot does not include private engine module names")


def _run(*command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        env.pop(name, None)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _tracked_paths(stage_output: str) -> set[str]:
    # `git ls-files --stage` separates metadata from the path with a single tab;
    # whitespace inside the path is legal and must be preserved.
    return {line.split("\t", 1)[1] for line in stage_output.splitlines() if "\t" in line}


def _wheel_metadata(wheel: Path) -> tuple[email.message.Message, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        return email.message_from_bytes(archive.read(metadata_path)), set(archive.namelist())


@pytest.fixture
def core_python(core_wheel: Path, tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    _run("uv", "venv", "--python", sys.executable, str(venv), cwd=tmp_path)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        str(core_wheel),
        cwd=tmp_path,
    )
    return python


def test_root_project_owns_core_only_standalone_layout() -> None:
    config = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    expected_core_project = {
        "name": "listenflow",
        "dynamic": ["version"],
        "description": "Engine-neutral streaming speech-to-text contracts and runtime",
        "license": "MIT",
        "requires-python": ">=3.12,<3.14",
        "dependencies": [],
    }
    assert all(config["project"][key] == value for key, value in expected_core_project.items())
    assert config["build-system"] == {
        "requires": ["hatchling", "hatch-vcs"],
        "build-backend": "hatchling.build",
    }
    assert config["tool"]["hatch"]["version"] == {
        "source": "vcs",
        "raw-options": {
            "root": ".",
            "relative_to": "pyproject.toml",
            "tag_regex": "^v(?P<version>[^/]+)$",
            "git_describe_command": [
                "git",
                "describe",
                "--dirty",
                "--tags",
                "--long",
                "--match",
                "v*",
            ],
        },
    }
    assert config["tool"]["hatch"]["build"]["hooks"]["vcs"]["version-file"] == (
        "src/listenflow/_version.py"
    )
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/listenflow"]

    assert (REPOSITORY_ROOT / "src/listenflow/__init__.py").is_file()
    assert (REPOSITORY_ROOT / "tests/test_core_api_contract.py").is_file()
    assert (REPOSITORY_ROOT / "tests/test_engine_neutral.py").is_file()
    assert not (REPOSITORY_ROOT / "packages").exists()
    assert not (REPOSITORY_ROOT / ".gitmodules").exists()

    stage_output = _run("git", "ls-files", "--stage", cwd=REPOSITORY_ROOT).stdout
    tracked = stage_output.splitlines()
    tracked_paths = _tracked_paths(stage_output)
    assert not any(line.split(maxsplit=1)[0] == "160000" for line in tracked)
    assert not any(path.startswith("packages/") for path in tracked_paths)
    assert not any(path.startswith(MOVED_TRACKED_PREFIXES) for path in tracked_paths)
    assert MOVED_TRACKED_PATHS.isdisjoint(tracked_paths)


def test_tracked_path_parser_keeps_whitespace_inside_paths() -> None:
    stage_output = (
        "100644 0123456789abcdef0123456789abcdef01234567 0\tbenchmarks/example fixture.py\n"
        "100644 0123456789abcdef0123456789abcdef01234567 0\tsrc/listenflow/__init__.py\n"
    )

    assert _tracked_paths(stage_output) == {
        "benchmarks/example fixture.py",
        "src/listenflow/__init__.py",
    }


@pytest.mark.req("SP-LF-071")
def test_root_test_and_lint_configuration_is_core_only() -> None:
    """core の検査と文書の build は別の dependency group で依存を導入する。"""
    config = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        re.split(r"[<>=!~]", dependency, maxsplit=1)[0]
        for dependency in config["dependency-groups"]["dev"]
    }

    assert dependency_names == {
        "pytest",
        "pytest-asyncio",
        "pyyaml",
        "ruff",
    }
    docs_dependency_names = {
        re.split(r"[<>=!~]", dependency, maxsplit=1)[0]
        for dependency in config["dependency-groups"]["docs"]
    }
    assert docs_dependency_names == {"mkdocs-material", "mkdocstrings[python]"}
    assert "workspace" not in config.get("tool", {}).get("uv", {})
    assert "sources" not in config.get("tool", {}).get("uv", {})
    assert "override-dependencies" not in config.get("tool", {}).get("uv", {})
    assert "index" not in config.get("tool", {}).get("uv", {})

    pytest_config = config["tool"]["pytest"]["ini_options"]
    assert pytest_config["asyncio_mode"] == "auto"
    assert pytest_config["testpaths"] == ["tests"]
    assert "--strict-markers" in pytest_config["addopts"]
    assert not any(
        option.startswith(ROLLBACK_TEST_EXCLUSIONS) for option in pytest_config["addopts"]
    )
    assert pytest_config["markers"] == [
        "cpu: qualified CPU payload and model tests",
        "gpu: requires a real GPU and explicitly provisioned runtime assets",
        "performance: opt-in performance measurement excluded from the default test run",
        "req(id): parent requirement ID for an externally visible behavior test",
    ]

    ruff_config = config["tool"]["ruff"]
    assert ruff_config["line-length"] == 100
    assert ruff_config["src"] == ["src", "tests"]
    assert not any(path.startswith("packages/") for path in ruff_config["src"])
    assert not any(path.startswith("packages/") for path in ruff_config.get("exclude", []))
    assert "respect-gitignore" not in ruff_config


def test_public_readme_drops_development_and_private_module_references() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    prohibited = PROHIBITED_DOCUMENTATION_TERMS + (
        ".nf/docs/spec.md",
        "開発 checkout 内の path",
        "rollback test",
        "明示除外",
    )

    assert not any(term in readme for term in prohibited)
    assert "uv build" in readme


@pytest.mark.req("SP-LF-037")
def test_public_readme_omits_private_engine_module_references() -> None:
    """公開する README には非公開の engine module の distribution 名を載せない。"""
    checks = _private_distribution_checks()
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert not any(term in readme for term in checks["module_distribution_names"])


def test_development_docs_describe_distribution_routes() -> None:
    try:
        agents = (REPOSITORY_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        if (REPOSITORY_ROOT / ".nf").exists():
            raise
        pytest.skip("the public snapshot does not include development instructions")
    assert not any(term in agents for term in PROHIBITED_DOCUMENTATION_TERMS)
    checks = _private_distribution_checks()
    assert not any("packages/" + name in agents for name in checks["module_distribution_names"])
    assert all(route in agents for route in checks["distribution_routes"])


def test_core_gitignore_drops_nested_module_paths() -> None:
    ignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "src/listenflow/_version.py" in ignore
    assert "packages/listenflow/src/listenflow/_version.py" not in ignore
    assert "/datasets/" in ignore


@pytest.mark.req("SP-LF-016")
def test_engine_modules_remain_outside_core(core_wheel: Path) -> None:
    """core の checkout と wheel は別の repository が所有する engine module を持たない。"""
    checks = _private_distribution_checks()
    tracked = _tracked_paths(_run("git", "ls-files", "--stage", cwd=REPOSITORY_ROOT).stdout)
    assert set(checks["removed_tracked_paths"]).isdisjoint(tracked)
    ignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert not any("packages/" + name in ignore for name in checks["module_distribution_names"])
    _, files = _wheel_metadata(core_wheel)
    assert not any(
        path.startswith(f"listenflow/{module}/")
        for path in files
        for module in checks["wheel_module_import_names"]
    )


@pytest.mark.req("SP-LF-016")
@pytest.mark.req("SP-LF-023")
def test_core_wheel_owns_namespace_root_without_runtime_dependencies(core_wheel: Path) -> None:
    """core の配布物は engine の runtime に依存しない pure Python の wheel になる。"""
    metadata, files = _wheel_metadata(core_wheel)

    assert metadata["Name"] == "listenflow"
    assert core_wheel.name.endswith("-py3-none-any.whl")
    assert metadata.get_all("Requires-Dist", []) == []
    assert "listenflow/__init__.py" in files
    assert "listenflow/_version.py" in files
    assert all(path.startswith(("listenflow/", "listenflow-")) for path in files)


@pytest.mark.req("SP-LF-016")
@pytest.mark.parametrize("module", ["test_engine", "unknown_engine"])
def test_core_only_wheel_preserves_public_imports_without_module_attributes(
    core_python: Path, tmp_path: Path, module: str
) -> None:
    """core だけを導入した利用側は公開 API を読め、表に無い engine module の属性参照は断られる。"""
    code = f"""
import listenflow
from listenflow import Final, OfflineModelDirectorySource, STTEngineBase

try:
    getattr(listenflow, {module!r})
except AttributeError:
    pass
else:
    raise AssertionError("unlisted engine module unexpectedly imported")
"""
    _run(str(core_python), "-I", "-c", code, cwd=tmp_path)


@pytest.mark.req("SP-LF-016")
@pytest.mark.parametrize("module", ["test_engine", "unknown_engine"])
def test_core_only_wheel_leaves_missing_module_imports_to_python(
    core_python: Path, tmp_path: Path, module: str
) -> None:
    """表に無い未導入の engine module の明示の import は、その名前を持つ標準の error になる。"""
    code = f"""
import listenflow
try:
    import listenflow.{module}
except ModuleNotFoundError as exc:
    assert type(exc) is ModuleNotFoundError
    assert exc.name == "listenflow.{module}"
    assert not isinstance(exc, listenflow.ModuleNotInstalledError)
else:
    raise AssertionError("missing engine module unexpectedly imported")
"""
    _run(str(core_python), "-I", "-c", code, cwd=tmp_path)


@pytest.mark.req("SP-LF-016")
@pytest.mark.req("SP-LF-009")
def test_installed_engine_module_is_explicitly_importable(
    core_python: Path, tmp_path: Path
) -> None:
    """利用側は、core の表に無い engine module も別の配布物から導入して明示の import で読める。"""
    project = tmp_path / "engine-module"
    module_path = project / "src/listenflow/test_engine"
    module_path.mkdir(parents=True)
    (module_path / "__init__.py").write_text(
        "from listenflow import EngineInputFormat\ninput_format = EngineInputFormat.PCM_S16LE\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        """[project]
name = "listenflow-test-engine"
version = "0.0.0"
dependencies = ["listenflow"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/listenflow"]
""",
        encoding="utf-8",
    )
    wheelhouse = tmp_path / "module-wheels"
    _run("uv", "build", "--wheel", "--out-dir", str(wheelhouse), cwd=project)
    wheel = next(wheelhouse.glob("listenflow_test_engine-*.whl"))
    _run(
        "uv", "pip", "install", "--python", str(core_python), "--no-deps", str(wheel), cwd=tmp_path
    )
    code = """
import listenflow
try:
    listenflow.test_engine
except AttributeError:
    pass
else:
    raise AssertionError("unlisted engine module was imported without an explicit import")

import listenflow.test_engine
from listenflow.test_engine import input_format

assert listenflow.test_engine.__name__ == "listenflow.test_engine"
assert input_format is listenflow.EngineInputFormat.PCM_S16LE
"""
    _run(str(core_python), "-I", "-c", code, cwd=tmp_path)


@pytest.mark.req("SP-LF-016")
@pytest.mark.parametrize(
    ("module", "distribution"),
    [
        ("faster_whisper", "listenflow-faster-whisper"),
        ("fastest_whisper", "listenflow-fastest-whisper"),
    ],
)
def test_core_only_wheel_guides_installation_of_published_engine_modules(
    core_python: Path, tmp_path: Path, module: str, distribution: str
) -> None:
    """core だけを導入した利用側は、公開済みの未導入 engine module の導入先と入れ方を案内される。"""
    code = f"""
import listenflow

try:
    getattr(listenflow, {module!r})
except listenflow.ModuleNotInstalledError as exc:
    assert isinstance(exc, listenflow.STTError)
    assert isinstance(exc, ImportError)
    assert isinstance(exc.__cause__, ModuleNotFoundError)
    assert exc.__cause__.name == "listenflow.{module}"
    guidance = dict(part.split("=", 1) for part in str(exc).split("; "))
    for field in ("why", "what"):
        assert {distribution!r} in guidance[field]
        assert "listenflow.{module}" in guidance[field]
    assert guidance["how"] == "pip install {distribution}"
else:
    raise AssertionError("missing published engine module unexpectedly imported")
"""
    _run(str(core_python), "-I", "-c", code, cwd=tmp_path)


@pytest.mark.req("SP-LF-016")
def test_registered_module_distinguishes_missing_module_and_runtime_dependency(
    core_python: Path, tmp_path: Path
) -> None:
    """表に載せた未導入の module は導入先を案内し、導入済みの module の依存の不足は元の error を保つ。"""
    code = """
import listenflow
listenflow._MODULE_DISTRIBUTIONS["test_engine"] = "listenflow-test-engine"
try:
    listenflow.test_engine
except ImportError as exc:
    assert isinstance(exc, listenflow.ModuleNotInstalledError)
    assert isinstance(exc, listenflow.STTError)
    assert isinstance(exc.__cause__, ModuleNotFoundError)
    assert exc.__cause__.name == "listenflow.test_engine"
    assert str(exc).split("; ") == [
        "why=listenflow.test_engine is unavailable because listenflow-test-engine is not installed",
        "what=listenflow-test-engine provides listenflow.test_engine",
        "how=pip install listenflow-test-engine",
    ]
else:
    raise AssertionError("missing registered engine module unexpectedly imported")

# An installed module whose runtime dependency is absent keeps the original error.
from pathlib import Path
module_path = Path("engine-modules/listenflow")
module_path.mkdir(parents=True)
(module_path / "test_engine.py").write_text("import absent_runtime_dependency_for_test\\n")
listenflow.__path__.append(str(module_path.resolve()))
try:
    listenflow.test_engine
except ModuleNotFoundError as exc:
    assert exc.name == "absent_runtime_dependency_for_test"
    assert not isinstance(exc, listenflow.ModuleNotInstalledError)
else:
    raise AssertionError("missing internal runtime dependency was hidden")
"""
    _run(str(core_python), "-I", "-c", code, cwd=tmp_path)


def _isolated_document_check(tmp_path: Path) -> Callable[[], None]:
    test_path = tmp_path / "tests/test_core_standalone.py"
    test_path.parent.mkdir()
    test_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "README.md").write_text("public documentation\n", encoding="utf-8")
    return runpy.run_path(str(test_path))["test_development_docs_describe_distribution_routes"]


@pytest.mark.req("SP-LF-036")
def test_public_snapshot_skips_the_development_document_check(tmp_path: Path) -> None:
    """A public snapshot without development files skips the development document check."""
    check = _isolated_document_check(tmp_path)

    with pytest.raises(pytest.skip.Exception):
        check()


@pytest.mark.req("SP-LF-036")
@pytest.mark.parametrize("missing", [False, True])
def test_development_checkout_requires_readable_agent_instructions(
    tmp_path: Path, missing: bool
) -> None:
    """A development checkout with missing or unreadable agent instructions fails its check."""
    (tmp_path / ".nf").mkdir()
    if not missing:
        (tmp_path / "AGENTS.md").mkdir()
    check = _isolated_document_check(tmp_path)

    with pytest.raises(FileNotFoundError if missing else IsADirectoryError):
        check()
