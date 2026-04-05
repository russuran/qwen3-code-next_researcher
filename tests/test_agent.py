from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.agent import (
    AgentConfig,
    JournalEntry,
    ResearchAgent,
    _parse_react,
)
from core.llm import LLM
from core.planner import ResearchPlan, SubQuestion
from core.tools import ToolResult


# ---------------------------------------------------------------------------
# ReAct parser
# ---------------------------------------------------------------------------

def test_parse_react_action():
    text = """Thought: I should search for papers.
Action: search_arxiv
Action Input: {"query": "transformers", "max_results": 5}"""
    action, inp, final = _parse_react(text)
    assert action == "search_arxiv"
    assert inp == {"query": "transformers", "max_results": 5}
    assert final is None


def test_parse_react_final_answer():
    text = """Thought: I have all the information.
Final Answer: The main approaches are X, Y, and Z."""
    action, inp, final = _parse_react(text)
    assert action is None
    assert inp is None
    assert "main approaches" in final


def test_parse_react_no_match():
    text = "Just some random text without proper format."
    action, inp, final = _parse_react(text)
    assert action is None
    assert inp is None
    assert final is None


def test_parse_react_bad_json():
    text = """Thought: search now
Action: search_github
Action Input: {bad json here}"""
    action, inp, final = _parse_react(text)
    assert action == "search_github"
    assert inp == {}  # failed to parse, returns empty dict


# ---------------------------------------------------------------------------
# JournalEntry
# ---------------------------------------------------------------------------

def test_journal_entry_serialization():
    entry = JournalEntry(
        timestamp="2026-01-01T00:00:00Z",
        phase="search",
        action="tool_call",
        tool_name="search_arxiv",
        tool_input={"query": "test"},
        result_summary="5 results",
    )
    data = json.loads(entry.model_dump_json())
    assert data["phase"] == "search"
    assert data["tool_name"] == "search_arxiv"


# ---------------------------------------------------------------------------
# Agent: full pipeline (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_full_run(mock_llm: LLM, patch_acompletion, tmp_output: Path, tmp_journal: Path):
    # Mock planner to return a simple plan
    plan = ResearchPlan(
        topic="test topic",
        slug="test-topic",
        sub_questions=[
            SubQuestion(
                question="What is X?",
                priority=5,
                sources=["arxiv"],
                keywords=["test", "keyword"],
            ),
        ],
        scope_notes="test scope",
    )

    # Mock search results
    search_result = ToolResult(
        tool_name="search_arxiv",
        success=True,
        data=[{"title": "Paper A", "abstract": "About testing", "authors": ["Author"]}],
    )

    # Mock analysis LLM response
    analysis_json = json.dumps({
        "title": "Paper A",
        "approach": "Testing approach",
        "key_contributions": ["contrib1"],
        "strengths": ["strong"],
        "weaknesses": ["weak"],
        "relevant_code": "",
        "tags": ["testing"],
    })

    # Mock synthesis LLM response
    synthesis_md = "# Test Topic\n\n## Overview\nThis is a test report."

    # Mock comparison
    comparison_md = "| Method | Approach |\n|--------|----------|\n| A | test |"

    config = AgentConfig(
        output_dir=str(tmp_output),
        journal_dir=str(tmp_journal),
        sources=["arxiv"],
        max_results_per_source=5,
        parallel_search=False,
        verbose=False,
        intervene=False,
    )

    agent = ResearchAgent(config=config, llm=mock_llm)

    # Patch planner
    agent.planner.generate_plan = AsyncMock(return_value=plan)

    # Patch tool execution
    agent.registry.execute = AsyncMock(side_effect=[
        # search_arxiv call
        search_result,
        # compare_methods call
        ToolResult(tool_name="compare_methods", success=True, data={"table": comparison_md}),
    ])

    # Patch LLM.generate for analysis and synthesis
    call_count = 0

    async def mock_generate(prompt, mode=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return analysis_json  # analysis
        return synthesis_md  # synthesis

    mock_llm.generate = mock_generate

    result_path = await agent.run("test topic")

    # Verify output structure
    assert result_path.exists()
    assert (result_path / "01_plan.json").exists()
    assert (result_path / "02_sources.json").exists()
    assert (result_path / "07_synthesis.md").exists()
    assert (result_path / "06_comparison.md").exists()
    assert (result_path / "08_references.md").exists()

    # Verify synthesis content
    report = (result_path / "07_synthesis.md").read_text()
    assert "test report" in report

    # Verify journal
    journal_file = tmp_journal / "test-topic.jsonl"
    assert journal_file.exists()
    lines = journal_file.read_text().strip().split("\n")
    assert len(lines) >= 2  # at least plan + synthesis entries


# ---------------------------------------------------------------------------
# Agent: intervention mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_intervention_abort(mock_llm: LLM, patch_acompletion, tmp_output: Path, tmp_journal: Path):
    plan = ResearchPlan(
        topic="abort test",
        slug="abort-test",
        sub_questions=[SubQuestion(question="Q?", priority=3)],
    )

    config = AgentConfig(
        output_dir=str(tmp_output),
        journal_dir=str(tmp_journal),
        intervene=True,
    )

    agent = ResearchAgent(config=config, llm=mock_llm)
    agent.planner.generate_plan = AsyncMock(return_value=plan)

    # Simulate user typing "abort"
    with patch("core.agent.asyncio.to_thread", new_callable=AsyncMock, return_value="abort"):
        result_path = await agent.run("abort test")

    # Should have stopped after plan phase
    assert (result_path / "01_plan.json").exists()
    assert not (result_path / "02_sources.json").exists()


# ---------------------------------------------------------------------------
# Agent: react loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_react_loop(mock_llm: LLM):
    config = AgentConfig(journal_dir="/tmp/test_journal")
    agent = ResearchAgent(config=config, llm=mock_llm)
    agent._journal_path = Path("/tmp/test_journal/test.jsonl")
    agent._journal_path.parent.mkdir(parents=True, exist_ok=True)

    call_count = 0

    async def fake_generate(prompt, mode=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                "Thought: Let me search.\n"
                "Action: search_arxiv\n"
                'Action Input: {"query": "test"}'
            )
        return "Thought: Done.\nFinal Answer: Found interesting results."

    mock_llm.generate = fake_generate
    agent.registry.execute = AsyncMock(return_value=ToolResult(
        tool_name="search_arxiv", success=True, data=[{"title": "Paper"}],
    ))

    answer = await agent._react_loop("Find papers about testing")
    assert "interesting results" in answer
    assert call_count == 2
