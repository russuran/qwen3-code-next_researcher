"""Git versioning: branch management for task runs and variants."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GitVersioning:
    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repo_path)] + list(args),
            capture_output=True, text=True, check=check,
        )

    def init_if_missing(self) -> bool:
        git_dir = self.repo_path / ".git"
        if git_dir.exists():
            return False
        self._run("init")
        self._run("add", "-A")
        self._run("commit", "-m", "Initial commit (auto-generated)")
        logger.info("Initialized git repo at %s", self.repo_path)
        return True

    def current_branch(self) -> str:
        result = self._run("branch", "--show-current", check=False)
        return result.stdout.strip()

    def create_task_branch(self, run_id: str, slug: str = "") -> str:
        name = f"ai/task/{run_id[:8]}" + (f"-{slug}" if slug else "")
        return self.create_branch(name)

    def create_variant_branch(self, run_id: str, variant_id: str) -> str:
        return self.create_branch(f"ai/variant/{run_id[:8]}-{variant_id}")

    def create_explore_branch(self, run_id: str, slug: str = "") -> str:
        return self.create_branch(f"ai/explore/{run_id[:8]}" + (f"-{slug}" if slug else ""))

    def create_branch(self, branch_name: str, checkout: bool = True) -> str:
        self._run("checkout", "-b", branch_name)
        logger.info("Created branch: %s", branch_name)
        return branch_name

    def checkout(self, branch: str) -> None:
        self._run("checkout", branch)

    def commit(self, message: str, run_id: str = "", task_id: str = "") -> str:
        self._run("add", "-A")
        result = self._run("commit", "-m", message, "--allow-empty", check=False)
        sha = self._run("rev-parse", "HEAD").stdout.strip()
        logger.info("Committed: %s (%s)", message[:50], sha[:8])
        return sha

    def diff(self, base: str = "main", head: str = "HEAD") -> str:
        result = self._run("diff", f"{base}...{head}", check=False)
        return result.stdout

    def list_branches(self) -> list[str]:
        result = self._run("branch", "--list", "--format=%(refname:short)")
        return [b.strip() for b in result.stdout.splitlines() if b.strip()]

    def get_log(self, n: int = 10) -> list[dict[str, str]]:
        result = self._run("log", f"-{n}", "--format=%H|%s|%an|%ai", check=False)
        commits = []
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "sha": parts[0], "message": parts[1],
                    "author": parts[2], "date": parts[3],
                })
        return commits
