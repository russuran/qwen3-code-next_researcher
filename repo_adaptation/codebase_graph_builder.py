"""Codebase graph builder: convenience wrapper around CodebaseGraphBuilder."""
from __future__ import annotations

import logging
from pathlib import Path

from repo_adaptation.codebase_graph import CodebaseGraph, CodebaseGraphBuilder

logger = logging.getLogger(__name__)


def build_graph(repo_path: str | Path) -> CodebaseGraph:
    """Build a codebase graph from a repository path.

    Delegates to CodebaseGraphBuilder for AST-based analysis of Python files,
    extracting entities (modules, classes, functions) and edges (calls, imports,
    inheritance).
    """
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository not found: {repo_path}")

    builder = CodebaseGraphBuilder()
    graph = builder.build(repo_path)

    logger.info(
        "Built codebase graph: %d entities, %d edges, %d files, %d errors",
        len(graph.entities), len(graph.edges),
        graph.files_analyzed, len(graph.errors),
    )
    return graph
