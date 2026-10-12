from __future__ import annotations

import email
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from noren.git import Entry
from noren.publishing import match_lists


ROOT = Path(__file__).resolve().parents[1]


def _sdist_paths(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        members = [PurePosixPath(member.name) for member in archive.getmembers() if member.isfile()]
    assert len({member.parts[0] for member in members}) == 1
    return {str(PurePosixPath(*member.parts[1:])) for member in members}


@pytest.mark.req("SP-LF-037")
def test_sdist_contains_only_core_sources_and_build_inputs(core_sdist: Path) -> None:
    """PyPI の source 配布物には core の source と build に必要な公開 file だけが入る。"""
    files = _sdist_paths(core_sdist)
    build_inputs = {"README.md", "LICENSE", "pyproject.toml", "PKG-INFO", ".gitignore"}

    assert {"README.md", "LICENSE", "pyproject.toml", "PKG-INFO"} <= files
    assert "src/listenflow/__init__.py" in files
    assert "src/listenflow/_version.py" in files
    unexpected = {
        path
        for path in files
        if path not in build_inputs and not path.startswith("src/listenflow/")
    }
    assert not unexpected, f"non-distribution files in sdist: {sorted(unexpected)}"


@pytest.mark.req("SP-LF-036")
@pytest.mark.req("SP-LF-037")
def test_sdist_omits_every_private_snapshot_exclusion(
    public_snapshot_config: dict[str, str], core_sdist: Path
) -> None:
    """公開用 repository から除く file は、PyPI の source 配布物からも除かれる。"""
    # Matching needs paths and file kinds, not Git object IDs. Check generated
    # files such as PKG-INFO against exclude without requiring an include rule.
    entries = [Entry(path, "100644", "blob", "") for path in _sdist_paths(core_sdist)]
    private = match_lists(entries, public_snapshot_config)["exclude"]
    assert not private, f"private snapshot files in sdist: {sorted(private)}"


@pytest.mark.req("SP-LF-017")
@pytest.mark.req("SP-LF-021")
def test_wheel_describes_the_project_on_pypi(core_wheel: Path) -> None:
    """wheel は README、公開 repository と API 文書の URL、著者、分類を含む metadata と license の file を持つ。"""
    with zipfile.ZipFile(core_wheel) as archive:
        metadata_path = next(path for path in archive.namelist() if path.endswith("/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_path))
        license_path = next(
            path for path in archive.namelist() if path.endswith("/licenses/LICENSE")
        )
        assert archive.read(license_path) == (ROOT / "LICENSE").read_bytes()

    assert metadata["Description-Content-Type"] == "text/markdown"
    assert metadata.get_payload() == (ROOT / "README.md").read_text(encoding="utf-8")
    assert set(metadata.get_all("Project-URL", [])) == {
        "Repository, https://github.com/sync-dev-org/listenflow",
        "Documentation, https://sync-dev-org.github.io/listenflow/",
    }
    assert metadata["Author-email"] == "minamik <mia@sync.dev>"
    assert set(metadata["Keywords"].split(",")) == {"stt", "speech-to-text", "speech-recognition"}
    assert set(metadata.get_all("Classifier", [])) == {
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    }
