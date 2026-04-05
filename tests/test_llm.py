from __future__ import annotations

import pytest
from pydantic import BaseModel

from core.llm import LLM, LLMConfig, LLMMode, ModeConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_config():
    cfg = LLMConfig()
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen3:8b"
    assert "thinking" in cfg.modes
    assert "fast" in cfg.modes


def test_custom_config():
    cfg = LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="sk-test",
        modes={"thinking": ModeConfig(temperature=0.1, max_tokens=8192)},
    )
    assert cfg.provider == "openai"
    assert cfg.modes["thinking"].temperature == 0.1


# ---------------------------------------------------------------------------
# Model string building
# ---------------------------------------------------------------------------

def test_model_string_ollama():
    llm = LLM(LLMConfig(provider="ollama", model="qwen3:8b"))
    assert llm._model_string == "ollama/qwen3:8b"


def test_model_string_openai():
    llm = LLM(LLMConfig(provider="openai", model="gpt-4o"))
    assert llm._model_string == "openai/gpt-4o"


def test_model_string_anthropic():
    llm = LLM(LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"))
    assert llm._model_string == "anthropic/claude-sonnet-4-20250514"


def test_model_string_vllm():
    llm = LLM(LLMConfig(provider="vllm", model="my-model", host="http://gpu-server:8000"))
    assert llm._model_string == "openai/my-model"
    assert llm._api_base == "http://gpu-server:8000"


def test_model_string_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        LLM(LLMConfig(provider="unknown_provider", model="x"))


# ---------------------------------------------------------------------------
# API base resolution
# ---------------------------------------------------------------------------

def test_api_base_local_providers():
    for provider in ("ollama", "vllm", "lmstudio"):
        llm = LLM(LLMConfig(provider=provider, model="m", host="http://localhost:1234"))
        assert llm._api_base == "http://localhost:1234"


def test_api_base_cloud_providers():
    for provider in ("openai", "anthropic", "gemini"):
        llm = LLM(LLMConfig(provider=provider, model="m"))
        assert llm._api_base is None


# ---------------------------------------------------------------------------
# Generate (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_returns_text(mock_llm: LLM, patch_acompletion):
    with patch_acompletion("Hello from LLM"):
        result = await mock_llm.generate("Say hello")
    assert result == "Hello from LLM"


@pytest.mark.asyncio
async def test_generate_with_system(mock_llm: LLM, patch_acompletion):
    with patch_acompletion("response") as ctx:
        result = await mock_llm.generate("prompt", system="You are a helper")
    assert result == "response"


@pytest.mark.asyncio
async def test_generate_uses_mode_config(mock_llm: LLM, patch_acompletion):
    with patch_acompletion("fast response") as ctx:
        result = await mock_llm.generate("quick task", mode=LLMMode.FAST)
    assert result == "fast response"


# ---------------------------------------------------------------------------
# Generate structured (mocked)
# ---------------------------------------------------------------------------

class SampleSchema(BaseModel):
    name: str
    count: int


@pytest.mark.asyncio
async def test_generate_structured_parses_json(mock_llm: LLM, patch_acompletion):
    json_response = '{"name": "test", "count": 42}'
    with patch_acompletion(json_response):
        result = await mock_llm.generate_structured("Give me data", schema=SampleSchema)
    assert isinstance(result, SampleSchema)
    assert result.name == "test"
    assert result.count == 42


@pytest.mark.asyncio
async def test_generate_structured_handles_code_block(mock_llm: LLM, patch_acompletion):
    response = '```json\n{"name": "wrapped", "count": 7}\n```'
    with patch_acompletion(response):
        result = await mock_llm.generate_structured("Give me data", schema=SampleSchema)
    assert result.name == "wrapped"
    assert result.count == 7
