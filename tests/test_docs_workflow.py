from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/docs.yml"
PUBLIC_REPOSITORY_CONDITION = "github.repository == 'sync-dev-org/listenflow'"


@pytest.fixture
def workflow() -> dict:
    return yaml.safe_load((ROOT / WORKFLOW).read_text(encoding="utf-8"))


@pytest.mark.req("SP-LF-072")
def test_docs_workflow_grants_only_checkout_pages_and_identity_permissions(workflow: dict) -> None:
    """文書の公開には内容の読み取り、Pages と OIDC token の書き込みの権限だけを与える。"""
    assert workflow["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert all("permissions" not in job for job in workflow["jobs"].values())


@pytest.mark.req("SP-LF-072")
def test_docs_workflow_uses_no_secrets() -> None:
    """文書の変換と公開は利用者が保管した secret を参照しない。"""
    assert "secrets." not in (ROOT / WORKFLOW).read_text(encoding="utf-8")


@pytest.mark.req("SP-LF-072")
def test_docs_workflow_runs_only_documentation_commands(workflow: dict) -> None:
    """文書の workflow は文書用の依存の導入と site の変換だけを実行する。"""
    commands = [
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    ]
    assert commands == [
        "uv sync --locked --only-group docs",
        "uv run --no-sync mkdocs build --strict",
    ]
    actions = [
        step["uses"].split("@", 1)[0]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert actions == [
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/configure-pages",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    ]


@pytest.mark.req("SP-LF-071")
def test_docs_workflow_runs_only_in_public_repository(workflow: dict) -> None:
    """文書の workflow は公開用の repository でだけ job を開始する。"""
    assert workflow["jobs"]
    assert all(job.get("if") == PUBLIC_REPOSITORY_CONDITION for job in workflow["jobs"].values())


@pytest.mark.req("SP-LF-071")
def test_docs_workflow_publishes_main_snapshots_and_manual_runs(workflow: dict) -> None:
    """main に送られた snapshot と手動の実行で、生成した site を Pages に公開する。"""
    assert workflow["on"] == {"push": {"branches": ["main"]}, "workflow_dispatch": None}
    (job,) = workflow["jobs"].values()
    assert job["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    upload = next(step for step in job["steps"] if "upload-pages-artifact@" in step.get("uses", ""))
    assert upload["with"]["path"] == "site/"
    deploy = job["steps"][-1]
    assert deploy["id"] == "deployment"
    assert deploy["uses"].startswith("actions/deploy-pages@")
    references = {
        step["uses"].split("@", 1)[0]: step["uses"].split("@", 1)[1]
        for step in job["steps"]
        if "uses" in step
    }
    # astral-sh/setup-uv publishes only full release tags, so a major-only tag does not resolve.
    assert re.fullmatch(r"v\d+\.\d+\.\d+", references.pop("astral-sh/setup-uv"))
    assert all(re.fullmatch(r"v\d+", version) for version in references.values())


@pytest.mark.req("SP-LF-071")
def test_public_snapshot_includes_only_the_documentation_workflow(
    public_snapshot_config: dict[str, str], tmp_path: Path
) -> None:
    """snapshot は文書の workflow だけを送り、ほかの GitHub 用 file を除く。"""
    github_includes = {
        line.strip() for line in public_snapshot_config["include"].splitlines() if ".github" in line
    }
    assert github_includes == {f"/{WORKFLOW}"}
    candidates = [
        WORKFLOW,
        ".github/ISSUE_TEMPLATE/bug.md",
        ".github/README.md",
        ".github/workflows/test.yml",
        ".github/workflows/other/nested.yml",
    ]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(public_snapshot_config["exclude"], encoding="utf-8")
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(candidates) + "\n",
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(completed.stdout.splitlines()) == set(candidates) - {WORKFLOW}
