from __future__ import annotations

import ast
import re
import runpy
from collections.abc import Callable
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _requirement_ids_in_tests() -> set[str]:
    ids = set()
    for path in (REPOSITORY_ROOT / "tests").rglob("test_*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                try:
                    marker = ast.unparse(decorator.func)
                except ValueError:
                    continue
                if marker != "pytest.mark.req":
                    continue
                assert len(decorator.args) == 1 and not decorator.keywords, (
                    f"{path}:{node.lineno} must pass one requirement ID to pytest.mark.req"
                )
                value = decorator.args[0]
                assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
                    f"{path}:{node.lineno} must pass a literal requirement ID to pytest.mark.req"
                )
                ids.add(value.value)
    return ids


def test_requirement_markers_refer_to_existing_requirements() -> None:
    try:
        spec = (REPOSITORY_ROOT / ".nf/docs/spec.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        if (REPOSITORY_ROOT / "AGENTS.md").exists() or (REPOSITORY_ROOT / ".nf").exists():
            raise
        pytest.skip("the public snapshot does not include the private spec document")
    existing_ids = set(re.findall(r"^\*\*(SP-LF-\d{3})\*\* ", spec, re.MULTILINE))
    marked_ids = _requirement_ids_in_tests()

    assert marked_ids, "no parent requirement IDs are attached to tests"
    assert marked_ids <= existing_ids, (
        f"unknown parent requirement IDs: {marked_ids - existing_ids}"
    )


def _isolated_requirement_check(tmp_path: Path) -> Callable[[], None]:
    test_path = tmp_path / "tests/test_requirement_markers.py"
    test_path.parent.mkdir()
    test_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    return runpy.run_path(str(test_path))["test_requirement_markers_refer_to_existing_requirements"]


@pytest.mark.req("SP-LF-036")
def test_public_snapshot_skips_the_private_requirement_check(tmp_path: Path) -> None:
    """A public snapshot without development files skips the private requirement check."""
    check = _isolated_requirement_check(tmp_path)

    with pytest.raises(pytest.skip.Exception):
        check()


@pytest.mark.req("SP-LF-036")
@pytest.mark.parametrize("development_file", ["AGENTS.md", ".nf/other.md"])
def test_development_checkout_requires_its_requirement_document(
    tmp_path: Path, development_file: str
) -> None:
    """A development checkout with a missing requirement document fails its check."""
    path = tmp_path / development_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("development file\n", encoding="utf-8")
    check = _isolated_requirement_check(tmp_path)

    with pytest.raises(FileNotFoundError):
        check()


@pytest.mark.req("SP-LF-036")
def test_unreadable_requirement_document_fails_its_check(tmp_path: Path) -> None:
    """An existing requirement path that cannot be read fails instead of skipping."""
    (tmp_path / ".nf/docs/spec.md").mkdir(parents=True)
    check = _isolated_requirement_check(tmp_path)

    with pytest.raises(IsADirectoryError):
        check()
