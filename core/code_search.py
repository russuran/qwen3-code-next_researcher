"""Code search: search over AST-based codebase graph."""
from __future__ import annotations

import logging
from typing import Any

from repo_adaptation.codebase_graph import CodebaseGraph, CodeEntity

logger = logging.getLogger(__name__)


class CodeSearchResult:
    def __init__(self, entity: CodeEntity, score: float, match_type: str):
        self.entity = entity
        self.score = score
        self.match_type = match_type


class CodeSearch:
    """Search across codebase graph by name, type, calls, and imports."""

    def __init__(self, graph: CodebaseGraph) -> None:
        self.graph = graph

    def search_by_name(self, query: str, limit: int = 10) -> list[CodeSearchResult]:
        query_lower = query.lower()
        results = []
        for entity in self.graph.entities:
            name_lower = entity.name.lower()
            if query_lower in name_lower:
                score = 1.0 if name_lower == query_lower else 0.5
                results.append(CodeSearchResult(entity, score, "name_match"))
            elif entity.docstring and query_lower in entity.docstring.lower():
                results.append(CodeSearchResult(entity, 0.3, "docstring_match"))
        return sorted(results, key=lambda r: -r.score)[:limit]

    def search_by_type(self, entity_type: str) -> list[CodeEntity]:
        return [e for e in self.graph.entities if e.entity_type == entity_type]

    def search_callers(self, function_name: str) -> list[CodeEntity]:
        return self.graph.get_callers(function_name)

    def search_imports_of(self, module: str) -> list[CodeEntity]:
        return self.graph.get_imports_of(module)

    def search_by_file(self, file_path: str) -> list[CodeEntity]:
        return [e for e in self.graph.entities if file_path in e.file_path]

    def find_entry_points(self) -> list[CodeEntity]:
        entry_names = {"main", "app", "create_app", "cli"}
        return [e for e in self.graph.entities
                if e.entity_type == "function" and e.name in entry_names]

    def find_tests(self) -> list[CodeEntity]:
        return [e for e in self.graph.entities if "test" in e.file_path.lower()]
