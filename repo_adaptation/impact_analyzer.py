"""Impact analyzer: determines what's affected by a proposed change."""
from __future__ import annotations

import logging

from pydantic import BaseModel

from repo_adaptation.codebase_graph import CodebaseGraph

logger = logging.getLogger(__name__)


class ImpactResult(BaseModel):
    target_entity: str
    directly_affected: list[str] = []
    transitively_affected: list[str] = []
    test_files_affected: list[str] = []
    risk_level: str = "low"  # low | medium | high
    summary: str = ""


class ImpactAnalyzer:
    def __init__(self, graph: CodebaseGraph) -> None:
        self.graph = graph
        self._reverse_edges: dict[str, list[str]] = {}
        self._build_reverse_edges()

    def _build_reverse_edges(self) -> None:
        for edge in self.graph.edges:
            self._reverse_edges.setdefault(edge["to"], []).append(edge["from"])

    def analyze(self, target: str, max_depth: int = 3) -> ImpactResult:
        result = ImpactResult(target_entity=target)

        # Direct dependents
        direct = self._reverse_edges.get(target, [])
        result.directly_affected = list(set(direct))

        # Transitive dependents (BFS)
        visited = set(direct)
        frontier = list(direct)
        depth = 0
        while frontier and depth < max_depth:
            next_frontier = []
            for entity in frontier:
                dependents = self._reverse_edges.get(entity, [])
                for dep in dependents:
                    if dep not in visited and dep != target:
                        visited.add(dep)
                        next_frontier.append(dep)
            frontier = next_frontier
            depth += 1

        result.transitively_affected = [e for e in visited if e not in result.directly_affected]

        # Find affected test files
        for entity_name in [target] + result.directly_affected:
            entity = self.graph.get_entity(entity_name)
            if entity and "test" in entity.file_path.lower():
                result.test_files_affected.append(entity.file_path)

        for entity in self.graph.entities:
            if "test" in entity.file_path.lower():
                if target in entity.calls or any(d in entity.calls for d in result.directly_affected):
                    if entity.file_path not in result.test_files_affected:
                        result.test_files_affected.append(entity.file_path)

        # Risk assessment
        total_affected = len(result.directly_affected) + len(result.transitively_affected)
        if total_affected > 10:
            result.risk_level = "high"
        elif total_affected > 3:
            result.risk_level = "medium"
        else:
            result.risk_level = "low"

        result.summary = (
            f"Changing '{target}' affects {len(result.directly_affected)} direct "
            f"and {len(result.transitively_affected)} transitive dependents. "
            f"Risk: {result.risk_level}. Tests: {len(result.test_files_affected)} files."
        )

        logger.info(result.summary)
        return result
