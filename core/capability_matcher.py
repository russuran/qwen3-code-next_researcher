"""Capability matcher: maps required capabilities to available tools."""
from __future__ import annotations

import logging

from core.task_model_builder import TaskModel
from core.tools import ToolDefinition, registry

logger = logging.getLogger(__name__)

_CAPABILITY_TOOL_MAP: dict[str, list[str]] = {
    "web_search": ["search_arxiv", "search_semantic_scholar", "search_github"],
    "web_browse": ["browse_web"],
    "pdf_read": ["download_pdf", "parse_pdf"],
    "code_analysis": ["inspect_code"],
    "sandbox": ["run_sandbox"],
    "latex": ["parse_latex"],
}


class CapabilityMatcher:
    """Matches required capabilities from a TaskModel to available tools."""

    def __init__(
        self,
        capability_map: dict[str, list[str]] | None = None,
    ) -> None:
        self._map = capability_map or _CAPABILITY_TOOL_MAP

    def match(self, task_model: TaskModel) -> list[ToolDefinition]:
        """Return list of tool definitions that satisfy the task's capabilities."""
        matched_names: set[str] = set()
        for cap in task_model.capabilities:
            tool_names = self._map.get(cap, [])
            if not tool_names:
                logger.warning("No tools mapped for capability '%s'", cap)
            matched_names.update(tool_names)

        tools: list[ToolDefinition] = []
        for name in sorted(matched_names):
            defn = registry.get(name)
            if defn:
                tools.append(defn)
            else:
                logger.debug("Tool '%s' not registered, skipping", name)

        logger.info(
            "Matched %d tools for %d capabilities",
            len(tools), len(task_model.capabilities),
        )
        return tools
