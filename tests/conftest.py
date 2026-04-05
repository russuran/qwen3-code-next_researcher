from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.llm import LLM, LLMConfig
from core.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures: config / paths
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    j = tmp_path / "journal"
    j.mkdir()
    return j


# ---------------------------------------------------------------------------
# Fixtures: LLM (mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(provider="ollama", model="test-model", host="http://localhost:11434")


@pytest.fixture
def mock_llm(llm_config: LLMConfig) -> LLM:
    """LLM instance with mocked acompletion — returns canned responses."""
    llm = LLM(llm_config)
    return llm


def make_llm_response(content: str):
    """Helper: build a fake litellm response object."""
    class Choice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class Response:
        def __init__(self, text):
            self.choices = [Choice(text)]

    return Response(content)


@pytest.fixture
def patch_acompletion():
    """Context-manager fixture that patches litellm.acompletion.

    Usage in tests:
        def test_something(patch_acompletion):
            with patch_acompletion("response text") as mock:
                ...
    """
    def _patch(response_text: str):
        mock = AsyncMock(return_value=make_llm_response(response_text))
        return patch("litellm.acompletion", mock)

    return _patch


# ---------------------------------------------------------------------------
# Fixtures: ToolRegistry (clean)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_registry() -> ToolRegistry:
    """A fresh ToolRegistry with no tools registered."""
    return ToolRegistry()
