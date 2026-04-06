"""Repo manifest builder: builds a detailed manifest using RepoIngest."""
from __future__ import annotations

import logging
from pathlib import Path

from repo_adaptation.repo_ingest import RepoIngest, RepoManifest

logger = logging.getLogger(__name__)


def build_manifest(repo_path: str | Path, repos_dir: str = "repos") -> RepoManifest:
    """Build a complete repository manifest by ingesting the repo.

    Delegates to RepoIngest for the heavy lifting: language detection,
    entry point discovery, config file identification, and test dir scanning.
    """
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository not found: {repo_path}")

    ingest = RepoIngest(repos_dir=repos_dir)
    manifest = ingest.ingest(repo_path)

    # Enrich with additional metadata
    manifest.structure = _build_tree(repo_path, max_depth=3)

    logger.info(
        "Built manifest for %s: %d files, %d languages, %d entry points",
        manifest.name, manifest.total_files,
        len(manifest.languages), len(manifest.entry_points),
    )
    return manifest


def _build_tree(root: Path, max_depth: int = 3, _depth: int = 0) -> dict:
    """Build a shallow directory tree representation."""
    if _depth >= max_depth:
        return {}

    tree: dict = {}
    try:
        for item in sorted(root.iterdir()):
            if item.name.startswith(".") or item.name in (
                "node_modules", "__pycache__", "venv", ".venv",
            ):
                continue
            if item.is_dir():
                tree[item.name + "/"] = _build_tree(item, max_depth, _depth + 1)
            else:
                tree[item.name] = None
    except PermissionError:
        pass
    return tree
