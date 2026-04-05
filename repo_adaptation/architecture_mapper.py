"""Architecture mapper: recovers architectural layers and boundaries from codebase graph."""
from __future__ import annotations

import logging

from pydantic import BaseModel

from repo_adaptation.codebase_graph import CodebaseGraph

logger = logging.getLogger(__name__)


class ArchitectureLayer(BaseModel):
    name: str
    modules: list[str] = []
    description: str = ""
    dependencies: list[str] = []


class ArchitectureMap(BaseModel):
    layers: list[ArchitectureLayer] = []
    entry_points: list[str] = []
    boundaries: list[dict[str, str]] = []  # {from_layer, to_layer, via}
    patterns_detected: list[str] = []


# Heuristic layer detection for Python projects
_LAYER_PATTERNS = {
    "api": ["api", "routes", "views", "endpoints", "handlers", "controllers"],
    "core": ["core", "domain", "models", "entities", "schemas"],
    "services": ["services", "service", "use_cases", "usecases"],
    "data": ["db", "database", "repositories", "repo", "storage", "dal"],
    "infrastructure": ["infra", "config", "settings", "middleware"],
    "cli": ["cli", "commands", "cmd"],
    "tests": ["tests", "test", "spec"],
    "utils": ["utils", "helpers", "common", "lib", "shared"],
}


class ArchitectureMapper:
    def map(self, graph: CodebaseGraph) -> ArchitectureMap:
        arch = ArchitectureMap()

        # Group entities by detected layer
        layer_entities: dict[str, list[str]] = {}
        for entity in graph.entities:
            if entity.entity_type != "module":
                continue
            layer = self._detect_layer(entity.file_path)
            layer_entities.setdefault(layer, []).append(entity.name)

        for layer_name, modules in sorted(layer_entities.items()):
            arch.layers.append(ArchitectureLayer(
                name=layer_name,
                modules=modules,
                description=f"{len(modules)} modules",
            ))

        # Detect entry points
        for entity in graph.entities:
            if entity.entity_type == "function" and entity.name in ("main", "app", "create_app"):
                arch.entry_points.append(f"{entity.file_path}:{entity.name}")
            if entity.entity_type == "module" and any(
                ep in entity.file_path for ep in ("main.py", "cli.py", "app.py", "__main__.py")
            ):
                arch.entry_points.append(entity.file_path)

        # Detect boundaries (cross-layer imports)
        for edge in graph.edges:
            if edge["type"] == "imports":
                from_layer = self._detect_layer_for_entity(edge["from"], graph)
                to_layer = self._detect_layer_for_entity(edge["to"], graph)
                if from_layer != to_layer and from_layer and to_layer:
                    boundary = {"from_layer": from_layer, "to_layer": to_layer, "via": edge["from"]}
                    if boundary not in arch.boundaries:
                        arch.boundaries.append(boundary)

        # Detect patterns
        patterns = []
        entity_names = {e.name.lower() for e in graph.entities}
        if any("factory" in n for n in entity_names):
            patterns.append("factory_pattern")
        if any("registry" in n for n in entity_names):
            patterns.append("registry_pattern")
        if any("singleton" in n for n in entity_names):
            patterns.append("singleton_pattern")
        if any("middleware" in n for n in entity_names):
            patterns.append("middleware_pattern")
        if any(e.entity_type == "class" and e.bases for e in graph.entities):
            patterns.append("inheritance")
        arch.patterns_detected = patterns

        logger.info("Architecture: %d layers, %d entry points, %d patterns",
                     len(arch.layers), len(arch.entry_points), len(arch.patterns_detected))
        return arch

    @staticmethod
    def _detect_layer(file_path: str) -> str:
        parts = file_path.lower().replace("\\", "/").split("/")
        for part in parts:
            for layer, keywords in _LAYER_PATTERNS.items():
                if part in keywords:
                    return layer
        return "other"

    def _detect_layer_for_entity(self, entity_name: str, graph: CodebaseGraph) -> str:
        entity = graph.get_entity(entity_name)
        if entity:
            return self._detect_layer(entity.file_path)
        return ""
