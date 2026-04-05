from __future__ import annotations

import json

import pytest

from core.llm import LLM
from core.planner import Planner, ResearchPlan, SubQuestion


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    text = '{"key": "value"}'
    assert Planner._extract_json(text) == {"key": "value"}


def test_extract_json_markdown_block():
    text = 'Here is the plan:\n```json\n{"key": "value"}\n```\nDone.'
    assert Planner._extract_json(text) == {"key": "value"}


def test_extract_json_embedded():
    text = 'Some preamble {"key": "value"} trailing text'
    assert Planner._extract_json(text) == {"key": "value"}


def test_extract_json_invalid():
    text = "no json here at all"
    assert Planner._extract_json(text) == {}


def test_extract_json_nested_braces():
    text = '{"a": {"b": 1}, "c": [1, 2]}'
    result = Planner._extract_json(text)
    assert result["a"]["b"] == 1
    assert result["c"] == [1, 2]


# ---------------------------------------------------------------------------
# generate_plan (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_plan(mock_llm: LLM, patch_acompletion):
    plan_json = json.dumps({
        "topic": "vector databases",
        "sub_questions": [
            {
                "question": "What are the main vector database implementations?",
                "priority": 5,
                "sources": ["arxiv", "github"],
                "keywords": ["vector database", "similarity search"],
            },
            {
                "question": "How do indexing algorithms compare?",
                "priority": 4,
                "sources": ["arxiv", "semantic_scholar"],
                "keywords": ["HNSW", "IVF", "LSH"],
            },
        ],
        "scope_notes": "Focus on open-source solutions",
    })

    with patch_acompletion(plan_json):
        planner = Planner(mock_llm)
        plan = await planner.generate_plan("vector databases")

    assert isinstance(plan, ResearchPlan)
    assert plan.topic == "vector databases"
    assert plan.slug == "vector-databases"
    assert len(plan.sub_questions) == 2
    assert plan.sub_questions[0].priority == 5
    assert "HNSW" in plan.sub_questions[1].keywords


@pytest.mark.asyncio
async def test_generate_plan_handles_bad_json(mock_llm: LLM, patch_acompletion):
    with patch_acompletion("I'm not sure, here is some random text"):
        planner = Planner(mock_llm)
        plan = await planner.generate_plan("something")

    assert isinstance(plan, ResearchPlan)
    assert plan.topic == "something"
    assert len(plan.sub_questions) == 0


# ---------------------------------------------------------------------------
# refine_plan (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refine_plan(mock_llm: LLM, patch_acompletion):
    refinement_json = json.dumps({
        "refined_queries": [
            {"query": "FAISS vs Milvus benchmark", "source": "github", "reason": "Need direct comparison"},
        ]
    })

    plan = ResearchPlan(
        topic="vector databases",
        slug="vector-databases",
        sub_questions=[SubQuestion(question="q1", priority=3)],
    )

    with patch_acompletion(refinement_json):
        planner = Planner(mock_llm)
        refined = await planner.refine_plan(plan, "Found papers about HNSW and IVF")

    assert len(refined) == 1
    assert refined[0].source == "github"
    assert "FAISS" in refined[0].query
