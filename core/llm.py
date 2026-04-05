from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

import litellm
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Suppress litellm's noisy logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLMMode(str, Enum):
    THINKING = "thinking"
    FAST = "fast"


class ModeConfig(BaseModel):
    temperature: float = 0.5
    max_tokens: int = 4096


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen3:8b"
    host: str | None = "http://localhost:11434"
    api_key: str | None = None
    modes: dict[str, ModeConfig] = {
        "thinking": ModeConfig(temperature=0.2, max_tokens=4096),
        "fast": ModeConfig(temperature=0.7, max_tokens=2048),
    }


# Provider -> litellm model prefix mapping
_PROVIDER_MAP: dict[str, str] = {
    "ollama": "ollama/{model}",
    "vllm": "openai/{model}",
    "lmstudio": "openai/{model}",
    "openai": "openai/{model}",
    "anthropic": "anthropic/{model}",
    "gemini": "gemini/{model}",
}


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model_string = self._build_model_string()
        self._api_base = self._resolve_api_base()
        self._api_key = config.api_key or None

    def _build_model_string(self) -> str:
        template = _PROVIDER_MAP.get(self.config.provider)
        if template is None:
            raise ValueError(
                f"Unknown provider '{self.config.provider}'. "
                f"Supported: {', '.join(_PROVIDER_MAP)}"
            )
        return template.format(model=self.config.model)

    def _resolve_api_base(self) -> str | None:
        if self.config.provider in ("ollama", "vllm", "lmstudio"):
            return self.config.host
        return None

    def _get_mode_config(self, mode: LLMMode) -> ModeConfig:
        return self.config.modes.get(mode.value, ModeConfig())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((litellm.exceptions.RateLimitError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        mode: LLMMode = LLMMode.THINKING,
        system: str | None = None,
    ) -> str:
        mode_cfg = self._get_mode_config(mode)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model_string,
            "messages": messages,
            "temperature": mode_cfg.temperature,
            "max_tokens": mode_cfg.max_tokens,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        logger.debug("LLM request: model=%s mode=%s", self._model_string, mode.value)
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        logger.debug("LLM response: %d chars", len(content))
        return content

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        mode: LLMMode = LLMMode.THINKING,
        system: str | None = None,
    ) -> BaseModel:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"No additional text, only the JSON object."
        )

        raw = await self.generate(structured_prompt, mode=mode, system=system)

        # Extract JSON from possible markdown code block
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return schema.model_validate_json(text)
