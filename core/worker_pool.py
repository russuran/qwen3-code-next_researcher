"""Worker pool: executes task nodes by dispatching to the right handler."""
from __future__ import annotations

import logging
from typing import Any

from core.llm import LLM, LLMMode
from core.planner import Planner
from core.prompts import ANALYSIS, COMPARISON_TABLE
from core.task_graph import NodeType, TaskNode
from core.tools import registry

logger = logging.getLogger(__name__)


class WorkerPool:
    """Routes TaskNode execution to the appropriate handler."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.planner = Planner(llm)
        self._handlers = {
            NodeType.SEARCH: self._handle_search,
            NodeType.FETCH: self._handle_fetch,
            NodeType.PARSE: self._handle_parse,
            NodeType.ANALYZE: self._handle_analyze,
            NodeType.COMPARE: self._handle_compare,
            NodeType.SYNTHESIZE: self._handle_synthesize,
        }

    async def execute(self, node: TaskNode) -> Any:
        handler = self._handlers.get(node.node_type)
        if handler is None:
            raise ValueError(f"No handler for node type: {node.node_type}")
        return await handler(node)

    async def _handle_search(self, node: TaskNode) -> Any:
        tool_name = node.params.get("tool", "search_arxiv")
        query = node.params.get("query", "")
        max_results = node.params.get("max_results", 10)
        result = await registry.execute(tool_name, query=query, max_results=max_results)
        return result.data if result.success else []

    async def _handle_fetch(self, node: TaskNode) -> Any:
        url = node.params.get("url", "")
        tool = node.params.get("tool", "browse_web")
        result = await registry.execute(tool, url=url)
        return result.data if result.success else None

    async def _handle_parse(self, node: TaskNode) -> Any:
        file_path = node.params.get("file_path", "")
        result = await registry.execute("parse_pdf", file_path=file_path)
        return result.data if result.success else None

    async def _handle_analyze(self, node: TaskNode) -> Any:
        content = node.params.get("content", "")
        source_type = node.params.get("source_type", "unknown")
        prompt = (
            ANALYSIS
            .replace("{{ source_type }}", source_type)
            .replace("{{ content }}", content[:4000])
        )
        raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
        return self.planner._extract_json(raw)

    async def _handle_compare(self, node: TaskNode) -> Any:
        methods = node.params.get("methods", "[]")
        prompt = COMPARISON_TABLE.replace("{{ methods }}", methods)
        return await self.llm.generate(prompt, mode=LLMMode.FAST)

    async def _handle_synthesize(self, node: TaskNode) -> Any:
        prompt = node.params.get("prompt", "")
        return await self.llm.generate(prompt, mode=LLMMode.THINKING)
