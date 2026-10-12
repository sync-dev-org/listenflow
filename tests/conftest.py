from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def public_snapshot_config() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    settings = root / "noren.yaml"
    if not settings.exists() and not (root / "AGENTS.md").exists() and not (root / ".nf").exists():
        pytest.skip("the public snapshot does not include development release tooling")
    return yaml.safe_load(settings.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def core_distributions(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("core-distributions")
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    completed = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


@pytest.fixture(scope="session")
def core_sdist(core_distributions: Path) -> Path:
    archives = list(core_distributions.glob("listenflow-*.tar.gz"))
    assert len(archives) == 1
    return archives[0]


@pytest.fixture(scope="session")
def core_wheel(core_distributions: Path) -> Path:
    wheels = list(core_distributions.glob("listenflow-*.whl"))
    assert len(wheels) == 1
    return wheels[0]
