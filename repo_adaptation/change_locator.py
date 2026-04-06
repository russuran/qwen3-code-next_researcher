"""Change locator: finds entities and files that need modification for a task."""
from __future__ import annotations

import logging

from pydantic import BaseModel

from repo_adaptation.architecture_mapper import ArchitectureMap
from repo_adaptation.codebase_graph import CodebaseGraph

logger = logging.getLogger(__name__)


class ChangeLocation(BaseModel):
    file_path: str
    entity_name: str
    entity_type: str
    reason: str = ""
    confidence: float = 0.0


def locate_changes(
    task: str,
    graph: CodebaseGraph,
    architecture_map: ArchitectureMap | None = None,
) -> list[ChangeLocation]:
    """Locate entities that need to change for a given task description.

    Uses keyword matching against entity names, docstrings, and file paths.
    """
    task_lower = task.lower()
    keywords = [w for w in task_lower.split() if len(w) > 3]

    locations: list[ChangeLocation] = []

    for entity in graph.entities:
        score = _relevance_score(entity.name, entity.docstring, entity.file_path, keywords)
        if score > 0:
            locations.append(ChangeLocation(
                file_path=entity.file_path,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                reason=f"Matched {score} keywords from task",
                confidence=min(score / max(len(keywords), 1), 1.0),
            ))

    # Boost entities in relevant architecture layers
    if architecture_map:
        _boost_by_layer(locations, architecture_map, keywords)

    locations.sort(key=lambda loc: -loc.confidence)
    logger.info("Located %d change candidates for task", len(locations))
    return locations


def _relevance_score(
    name: str, docstring: str, file_path: str, keywords: list[str],
) -> int:
    searchable = f"{name} {docstring} {file_path}".lower()
    return sum(1 for kw in keywords if kw in searchable)


def _boost_by_layer(
    locations: list[ChangeLocation],
    arch: ArchitectureMap,
    keywords: list[str],
) -> None:
    relevant_layers = set()
    for layer in arch.layers:
        if any(kw in layer.name.lower() for kw in keywords):
            relevant_layers.update(layer.modules)

    for loc in locations:
        if loc.entity_name in relevant_layers:
            loc.confidence = min(loc.confidence + 0.2, 1.0)
