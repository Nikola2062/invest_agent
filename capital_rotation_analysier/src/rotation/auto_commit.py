"""Auto-commit generated report files (design note O1).

After both the US and HK PDFs have shipped to Telegram, this module makes a
narrow `git add` + `git commit` of just that day's `reports/<date>_daily*.{md,pdf}`
artifacts. It is OFF by default — the user opts in via `reports.auto_commit` in
config.yaml.

Design notes (kept here so the code matches the
agreed scope verbatim):
  - Commit ONLY the day's report artifacts. Never `git add -A` (the working
    tree may include .env, the DB, scratch scripts).
  - Commit message: `reports: <YYYY-MM-DD> daily (US+HK)` with a body
    summarising the run (regime, headline rotation, run status).
  - Skip when the run was `degraded` (signals not published) — degraded
    reports shouldn't pollute git history.
  - Skip when there is no diff (re-runs of an already committed day).
  - Skip when git isn't available / cwd isn't a repo / HEAD is detached.
    Log + return a skip reason; never crash the pipeline.
  - DO NOT push automatically. The user pushes manually.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Returns (returncode, stdout, stderr). Never raises; if git is missing
    we report exit 127 with the message in stderr so the caller can branch."""
    try:
        cp = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True,
            check=False,
        )
        return cp.returncode, (cp.stdout or "").strip(), (cp.stderr or "").strip()
    except FileNotFoundError as exc:
        return 127, "", f"git not found: {exc}"


def _is_git_repo(cwd: Path) -> bool:
    rc, out, _ = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return rc == 0 and out == "true"


def _head_detached(cwd: Path) -> bool:
    rc, out, _ = _run_git(["symbolic-ref", "-q", "HEAD"], cwd)
    return rc != 0  # detached HEAD has no symbolic ref


def _has_pending_diff(cwd: Path, files: list[Path]) -> bool:
    """True if any of `files` has uncommitted changes (modified or untracked)."""
    if not files:
        return False
    rel = [str(p) for p in files]
    rc, out, _ = _run_git(["status", "--porcelain", "--", *rel], cwd)
    return rc == 0 and bool(out.strip())


def commit_reports(
    asof: date,
    repo_root: Path,
    *,
    pipeline_failed: bool,
    degraded: bool,
    enabled: bool,
    skip_on_degraded: bool,
    body_summary: str = "",
) -> dict:
    """Commit the day's report artifacts after a successful Telegram send.

    Returns a status dict:
      {"status": "committed" | "skipped:<reason>", "sha": "...", "files": [...]}

    Never raises — every failure path returns a `skipped:<reason>` status."""
    out: dict = {"status": "skipped:disabled", "sha": None, "files": []}

    if not enabled:
        return out
    if pipeline_failed:
        out["status"] = "skipped:pipeline_failed"
        return out
    if degraded and skip_on_degraded:
        out["status"] = "skipped:degraded"
        return out
    if not _is_git_repo(repo_root):
        out["status"] = "skipped:not_a_repo"
        return out
    if _head_detached(repo_root):
        out["status"] = "skipped:detached_head"
        return out

    # Build the narrow file list. Glob in `reports/` to catch both the daily
    # report and the `_daily_hk` companion, with .md and .pdf for each. Resolve
    # relative to repo_root since `git add` wants repo-relative paths.
    reports_dir = repo_root / "reports"
    if not reports_dir.exists():
        out["status"] = "skipped:no_reports_dir"
        return out

    iso = asof.isoformat()
    patterns = [
        f"{iso}_daily.md", f"{iso}_daily.pdf",
        f"{iso}_daily_hk.md", f"{iso}_daily_hk.pdf",
    ]
    candidates = [reports_dir / p for p in patterns]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        out["status"] = "skipped:no_artifacts"
        return out

    rel_paths = [p.relative_to(repo_root) for p in existing]
    if not _has_pending_diff(repo_root, rel_paths):
        out["status"] = "skipped:no_diff"
        out["files"] = [str(p) for p in rel_paths]
        return out

    # Stage only the report files (no `git add -A`).
    rc, _, err = _run_git(["add", "--", *[str(p) for p in rel_paths]], repo_root)
    if rc != 0:
        out["status"] = f"skipped:add_failed:{err[:120]}"
        return out

    # Verify staging actually picked up something — if `git add` no-op'd
    # because patterns didn't match (gitignore, etc.), bail without committing.
    rc, staged, _ = _run_git(["diff", "--cached", "--name-only", "--",
                              *[str(p) for p in rel_paths]], repo_root)
    if rc != 0 or not staged.strip():
        out["status"] = "skipped:nothing_staged"
        return out

    # Build the commit message.
    headline = f"reports: {iso} daily (US+HK)"
    body_lines = []
    if body_summary:
        body_lines.extend(body_summary.splitlines())
    body_lines.append("")
    body_lines.append("Auto-committed by the daily pipeline (reports.auto_commit=true).")
    msg = headline + ("\n\n" + "\n".join(body_lines) if body_lines else "")

    rc, _, err = _run_git(["commit", "-m", msg], repo_root)
    if rc != 0:
        out["status"] = f"skipped:commit_failed:{err[:120]}"
        return out

    # Capture the new SHA.
    rc, sha, _ = _run_git(["rev-parse", "HEAD"], repo_root)
    out["status"] = "committed"
    out["sha"] = sha if rc == 0 else None
    out["files"] = [str(p) for p in rel_paths]
    return out
