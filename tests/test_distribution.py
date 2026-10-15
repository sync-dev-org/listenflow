from __future__ import annotations

import email
import tarfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _sdist_files(sdist: Path) -> dict[str, bytes]:
    with tarfile.open(sdist) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        assert len({PurePosixPath(member.name).parts[0] for member in members}) == 1
        return {
            str(PurePosixPath(*PurePosixPath(member.name).parts[1:])): archive.extractfile(
                member
            ).read()
            for member in members
        }


@pytest.mark.req("SP-LF-037")
def test_sdist_contains_only_core_sources_and_build_inputs(core_sdist: Path) -> None:
    """PyPI の source 配布物には core の source と build に必要な公開 file だけが入る。"""
    files = _sdist_files(core_sdist).keys()
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
    exclusions = [
        PurePosixPath(line.strip().strip("/"))
        for line in public_snapshot_config["exclude"].splitlines()
        if line.strip()
    ]
    paths = {PurePosixPath(path) for path in _sdist_files(core_sdist)}
    private = {
        str(path)
        for path in paths
        if any(path == excluded or excluded in path.parents for excluded in exclusions)
    }
    assert not private, f"private snapshot files in sdist: {sorted(private)}"


@pytest.mark.req("SP-LF-037")
def test_distributions_contain_no_restricted_terms(core_sdist: Path, core_wheel: Path) -> None:
    """wheel と source 配布物の file 名と内容には、非公開の語を載せない。"""
    word_list = ROOT / "noren-words.txt"
    if not word_list.exists() and not (ROOT / "AGENTS.md").exists() and not (ROOT / ".nf").exists():
        pytest.skip("the public snapshot does not include the private word list")
    words = [
        unicodedata.normalize("NFKC", word.strip()).casefold()
        for word in word_list.read_text(encoding="utf-8").splitlines()
        if word.strip()
    ]
    files = {"sdist": _sdist_files(core_sdist)}
    with zipfile.ZipFile(core_wheel) as archive:
        files["wheel"] = {
            name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
        }
    violations = []
    for kind, contents in files.items():
        for path, data in contents.items():
            normalized = unicodedata.normalize(
                "NFKC", path + "\n" + data.decode("utf-8", errors="replace")
            ).casefold()
            if any(word in normalized for word in words):
                violations.append(f"{kind}: {path}")
    assert not violations, f"restricted terms in distribution files: {violations}"


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
