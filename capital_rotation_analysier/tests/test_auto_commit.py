"""O1: auto-commit reports — exhaustive skip-path coverage.

Each test sets up a temporary git repo and verifies exactly one path through
`commit_reports`. The function must NEVER raise — every failure mode lowers
to a `skipped:<reason>` status."""
from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from rotation.auto_commit import commit_reports


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd),
                          capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Initialise a bare-bones git repo with the `reports/` directory the
    auto-commit path expects to find."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(["init", "-q"], root)
    _git(["config", "user.email", "test@example.com"], root)
    _git(["config", "user.name", "Test"], root)
    _git(["config", "commit.gpgsign", "false"], root)
    (root / "reports").mkdir()
    # Seed an initial commit so HEAD is symbolic (otherwise git status behaves
    # differently — and `_head_detached` returns true on an empty repo).
    (root / "README.md").write_text("seed\n")
    _git(["add", "README.md"], root)
    _git(["commit", "-q", "-m", "seed"], root)
    return root


def _seed_artifacts(root: Path, iso: str) -> list[Path]:
    paths = [
        root / "reports" / f"{iso}_daily.md",
        root / "reports" / f"{iso}_daily.pdf",
        root / "reports" / f"{iso}_daily_hk.md",
        root / "reports" / f"{iso}_daily_hk.pdf",
    ]
    for p in paths:
        p.write_bytes(b"contents")
    return paths


def _last_commit_msg(root: Path) -> str:
    cp = _git(["log", "-1", "--pretty=%B"], root)
    return cp.stdout


def _staged_files(root: Path) -> list[str]:
    cp = _git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"], root)
    return cp.stdout.strip().splitlines()


# ============================================================
# Skip paths
# ============================================================

def test_skip_when_disabled(repo: Path):
    _seed_artifacts(repo, "2026-06-09")
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=False,
                         enabled=False, skip_on_degraded=True)
    assert res["status"] == "skipped:disabled"
    assert res["sha"] is None


def test_skip_when_pipeline_failed(repo: Path):
    _seed_artifacts(repo, "2026-06-09")
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=True, degraded=False,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "skipped:pipeline_failed"


def test_skip_when_degraded_and_flag_set(repo: Path):
    _seed_artifacts(repo, "2026-06-09")
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=True,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "skipped:degraded"


def test_commit_proceeds_when_degraded_but_flag_off(repo: Path):
    _seed_artifacts(repo, "2026-06-09")
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=True,
                         enabled=True, skip_on_degraded=False)
    assert res["status"] == "committed"


def test_skip_when_not_a_repo(tmp_path: Path):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    (plain / "reports").mkdir()
    _seed_artifacts(plain, "2026-06-09")
    res = commit_reports(date(2026, 6, 9), plain,
                         pipeline_failed=False, degraded=False,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "skipped:not_a_repo"


def test_skip_when_no_artifacts(repo: Path):
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=False,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "skipped:no_artifacts"


def test_skip_when_no_reports_dir(tmp_path: Path):
    root = tmp_path / "norepo"
    root.mkdir()
    _git(["init", "-q"], root)
    _git(["config", "user.email", "t@e.c"], root)
    _git(["config", "user.name", "T"], root)
    res = commit_reports(date(2026, 6, 9), root,
                         pipeline_failed=False, degraded=False,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "skipped:no_reports_dir"


def test_skip_when_no_diff(repo: Path):
    """Re-running on a day whose artifacts are already committed should no-op."""
    iso = "2026-06-09"
    _seed_artifacts(repo, iso)
    first = commit_reports(date(2026, 6, 9), repo,
                           pipeline_failed=False, degraded=False,
                           enabled=True, skip_on_degraded=True)
    assert first["status"] == "committed"
    # Same content, no edits -> no diff
    again = commit_reports(date(2026, 6, 9), repo,
                           pipeline_failed=False, degraded=False,
                           enabled=True, skip_on_degraded=True)
    assert again["status"] == "skipped:no_diff"


# ============================================================
# Happy paths
# ============================================================

def test_commit_only_stages_reports_files(repo: Path):
    """The .env / DB / scratch scripts in the working tree must NOT be
    swept into the commit. Only `reports/<date>_daily*.{md,pdf}`."""
    iso = "2026-06-09"
    artifacts = _seed_artifacts(repo, iso)

    # Drop a couple of "secrets" + scratch files at the repo root so we'd
    # notice if `git add -A` were used by accident.
    (repo / ".env").write_text("FRED_API_KEY=secret\n")
    (repo / "scratch.py").write_text("print('debug')\n")
    (repo / "data").mkdir()
    (repo / "data" / "rotation.duckdb").write_bytes(b"db")

    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=False,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "committed"
    assert res["sha"] is not None

    committed = _staged_files(repo)
    # All four artifacts committed
    for name in ("2026-06-09_daily.md", "2026-06-09_daily.pdf",
                 "2026-06-09_daily_hk.md", "2026-06-09_daily_hk.pdf"):
        assert any(name in c for c in committed), f"{name} missing from commit"
    # .env / scratch.py / DB NOT committed
    for forbidden in (".env", "scratch.py", "data/rotation.duckdb"):
        assert not any(forbidden in c for c in committed), \
            f"{forbidden} should never appear in an auto-commit"


def test_commit_message_format(repo: Path):
    iso = "2026-06-09"
    _seed_artifacts(repo, iso)
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=False,
                         enabled=True, skip_on_degraded=True,
                         body_summary="Regime: Risk-On Expansion")
    assert res["status"] == "committed"
    msg = _last_commit_msg(repo)
    assert msg.startswith(f"reports: {iso} daily (US+HK)")
    assert "Regime: Risk-On Expansion" in msg
    assert "Auto-committed by the daily pipeline" in msg


def test_commit_handles_partial_artifact_set(repo: Path):
    """Only US PDF produced (e.g. HK render crashed) — still committable."""
    iso = "2026-06-09"
    (repo / "reports" / f"{iso}_daily.md").write_text("us only\n")
    (repo / "reports" / f"{iso}_daily.pdf").write_bytes(b"pdf")
    res = commit_reports(date(2026, 6, 9), repo,
                         pipeline_failed=False, degraded=False,
                         enabled=True, skip_on_degraded=True)
    assert res["status"] == "committed"
    assert len(res["files"]) == 2
