"""Patch generator: produces patch candidates from a change plan."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from pydantic import BaseModel

from core.llm import LLM, LLMMode
from repo_adaptation.patch_editor import PatchCandidate

logger = logging.getLogger(__name__)


class ChangePlanStep(BaseModel):
    file_path: str
    description: str
    priority: int = 0


_PATCH_PROMPT = """\
Generate a code modification for this file. File: {file_path}
```
{content}
```
Change request: {description}
Output ONLY the complete modified file in ```python ... ```.
"""


async def generate_patches(
    change_plan: list[ChangePlanStep],
    repo_path: str | Path,
    llm: LLM,
) -> list[PatchCandidate]:
    """Generate patch candidates for each step in the change plan."""
    repo = Path(repo_path)
    patches: list[PatchCandidate] = []

    for step in change_plan:
        target = repo / step.file_path
        if not target.exists():
            logger.warning("File not found: %s, generating new file", step.file_path)
            content = ""
        else:
            content = target.read_text(encoding="utf-8", errors="ignore")

        prompt = _PATCH_PROMPT.format(
            file_path=step.file_path,
            content=content[:8000],
            description=step.description,
        )

        try:
            raw = await llm.generate(prompt, mode=LLMMode.THINKING)
            modified = _extract_code(raw) or raw
            patches.append(PatchCandidate(
                patch_id=str(uuid.uuid4())[:8], file_path=step.file_path,
                original=content, modified=modified, rationale=step.description,
            ))
        except Exception as e:
            logger.error("Patch generation failed for %s: %s", step.file_path, e)

    logger.info("Generated %d patches from %d steps", len(patches), len(change_plan))
    return patches


def _extract_code(text: str) -> str:
    if "```" in text:
        blocks = text.split("```")
        for block in blocks[1::2]:
            block = block.strip()
            if block.startswith("python"):
                return block[6:].strip()
            if block:
                return block.strip()
    return ""
