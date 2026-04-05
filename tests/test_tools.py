from __future__ import annotations

import pytest

from core.tools import ToolParam, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_and_list(clean_registry: ToolRegistry):
    @clean_registry.register(
        name="dummy",
        description="A dummy tool",
        params=[ToolParam(name="x", type="int", description="a number")],
    )
    async def dummy(x: int) -> ToolResult:
        return ToolResult(tool_name="dummy", success=True, data=x * 2)

    defs = clean_registry.list_definitions()
    assert len(defs) == 1
    assert defs[0].name == "dummy"
    assert defs[0].params[0].name == "x"


def test_register_multiple(clean_registry: ToolRegistry):
    for name in ("tool_a", "tool_b", "tool_c"):
        @clean_registry.register(name=name, description=f"desc {name}")
        async def fn() -> ToolResult:
            return ToolResult(tool_name=name, success=True)

    assert len(clean_registry.list_definitions()) == 3


def test_get_existing(clean_registry: ToolRegistry):
    @clean_registry.register(name="hello", description="says hello")
    async def hello() -> ToolResult:
        return ToolResult(tool_name="hello", success=True, data="hi")

    assert clean_registry.get("hello") is not None
    assert clean_registry.get("hello").name == "hello"


def test_get_missing(clean_registry: ToolRegistry):
    assert clean_registry.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_success(clean_registry: ToolRegistry):
    @clean_registry.register(
        name="add",
        description="adds two numbers",
        params=[
            ToolParam(name="a", type="int", description="first"),
            ToolParam(name="b", type="int", description="second"),
        ],
    )
    async def add(a: int, b: int) -> ToolResult:
        return ToolResult(tool_name="add", success=True, data=a + b)

    result = await clean_registry.execute("add", a=3, b=7)
    assert result.success is True
    assert result.data == 10


@pytest.mark.asyncio
async def test_execute_unknown_tool(clean_registry: ToolRegistry):
    result = await clean_registry.execute("unknown_tool")
    assert result.success is False
    assert "Unknown tool" in result.error


@pytest.mark.asyncio
async def test_execute_handles_exception(clean_registry: ToolRegistry):
    @clean_registry.register(name="crasher", description="always crashes")
    async def crasher() -> ToolResult:
        raise ValueError("boom")

    result = await clean_registry.execute("crasher")
    assert result.success is False
    assert "ValueError" in result.error
    assert "boom" in result.error


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_for_prompt(clean_registry: ToolRegistry):
    @clean_registry.register(
        name="search",
        description="Search for stuff",
        params=[
            ToolParam(name="query", type="str", description="search query"),
            ToolParam(name="limit", type="int", description="max results", required=False, default=10),
        ],
    )
    async def search(query: str, limit: int = 10) -> ToolResult:
        return ToolResult(tool_name="search", success=True, data=[])

    text = clean_registry.format_for_prompt()
    assert "Available tools:" in text
    assert "search(query: str, limit: int = 10)" in text
    assert "Search for stuff" in text


# ---------------------------------------------------------------------------
# ToolResult.to_observation
# ---------------------------------------------------------------------------

def test_to_observation_success():
    r = ToolResult(tool_name="t", success=True, data={"key": "value"})
    obs = r.to_observation()
    assert '"key"' in obs
    assert '"value"' in obs


def test_to_observation_error():
    r = ToolResult(tool_name="t", success=False, error="something broke")
    obs = r.to_observation()
    assert obs == "Error: something broke"


def test_to_observation_truncation():
    r = ToolResult(tool_name="t", success=True, data="x" * 20000)
    obs = r.to_observation(max_chars=100)
    assert len(obs) < 200
    assert "[truncated]" in obs
