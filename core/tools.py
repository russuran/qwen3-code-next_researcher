from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    params: list[ToolParam]
    fn: Callable[..., Awaitable[ToolResult]] | None = field(default=None, repr=False)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None

    def to_observation(self, max_chars: int = 8000) -> str:
        if not self.success:
            return f"Error: {self.error}"
        text = json.dumps(self.data, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        return text


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        params: list[ToolParam] | None = None,
    ) -> Callable:
        def decorator(fn: Callable[..., Awaitable[ToolResult]]) -> Callable[..., Awaitable[ToolResult]]:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                params=params or [],
                fn=fn,
            )
            logger.debug("Registered tool: %s", name)
            return fn
        return decorator

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        lines = ["Available tools:"]
        for tool in self._tools.values():
            params_str = ", ".join(
                f"{p.name}: {p.type}" + (f" = {p.default}" if not p.required else "")
                for p in tool.params
            )
            lines.append(f"- {tool.name}({params_str}): {tool.description}")
        return "\n".join(lines)

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: {name}",
            )
        if tool.fn is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' has no implementation",
            )
        try:
            logger.info("Executing tool: %s(%s)", name, kwargs)
            result = await tool.fn(**kwargs)
            return result
        except Exception as e:
            logger.error("Tool '%s' failed: %s", name, e, exc_info=True)
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )


# Module-level singleton
registry = ToolRegistry()
