"""Repository ingestion: clone/load repo and build manifest."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RepoManifest(BaseModel):
    path: str
    name: str
    languages: dict[str, int] = {}  # language -> file count
    total_files: int = 0
    total_lines: int = 0
    entry_points: list[str] = []
    config_files: list[str] = []
    test_dirs: list[str] = []
    structure: dict[str, Any] = {}


_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
    ".md": "markdown", ".txt": "text",
}

_CONFIG_PATTERNS = [
    "pyproject.toml", "setup.py", "setup.cfg", "package.json",
    "Cargo.toml", "go.mod", "Makefile", "Dockerfile", "docker-compose.yml",
    ".env.example", "requirements.txt",
]

_ENTRY_PATTERNS = ["main.py", "app.py", "cli.py", "manage.py", "index.js", "main.go"]
_TEST_PATTERNS = ["tests", "test", "spec", "__tests__"]


class RepoIngest:
    def __init__(self, repos_dir: str = "repos") -> None:
        self.repos_dir = Path(repos_dir)
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    def clone(self, url: str, branch: str = "main") -> Path:
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        dest = self.repos_dir / name
        if dest.exists():
            logger.info("Repo already exists: %s", dest)
            return dest

        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
            check=True, capture_output=True, text=True,
        )
        logger.info("Cloned %s -> %s", url, dest)
        return dest

    def ingest(self, path: str | Path) -> RepoManifest:
        repo_path = Path(path)
        manifest = RepoManifest(path=str(repo_path), name=repo_path.name)

        languages: dict[str, int] = {}
        total_lines = 0

        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            rel = str(f.relative_to(repo_path))

            # Skip hidden/vendor dirs
            if any(part.startswith(".") or part in ("node_modules", "__pycache__", "venv", ".venv")
                   for part in f.parts):
                continue

            manifest.total_files += 1
            ext = f.suffix.lower()
            lang = _LANG_MAP.get(ext, "other")
            languages[lang] = languages.get(lang, 0) + 1

            # Count lines for code files
            if ext in (".py", ".js", ".ts", ".go", ".rs", ".java"):
                try:
                    total_lines += len(f.read_text(errors="ignore").splitlines())
                except Exception:
                    pass

            # Detect config files
            if f.name in _CONFIG_PATTERNS:
                manifest.config_files.append(rel)

            # Detect entry points
            if f.name in _ENTRY_PATTERNS:
                manifest.entry_points.append(rel)

            # Detect test directories
            for test_pat in _TEST_PATTERNS:
                if test_pat in f.parts:
                    test_dir = str(Path(*f.parts[:f.parts.index(test_pat) + 1]).relative_to(repo_path))
                    if test_dir not in manifest.test_dirs:
                        manifest.test_dirs.append(test_dir)

        manifest.languages = languages
        manifest.total_lines = total_lines
        logger.info("Ingested %s: %d files, %d lines", repo_path.name, manifest.total_files, total_lines)
        return manifest
