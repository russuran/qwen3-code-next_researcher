"""Patch editor: generates code patches using LLM."""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class PatchCandidate(BaseModel):
    patch_id: str = ""
    file_path: str
    original: str
    modified: str
    rationale: str = ""


PATCH_PROMPT = """\
You are a code editor. Given the following file and a change request, \
produce the modified file content.

File: {file_path}
```
{content}
```

Change request: {request}

Rules:
- Output ONLY the full modified file content.
- Wrap in a code block: ```python ... ```
- Do not explain, only output the code.
"""


class PatchEditor:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def generate_patch(
        self,
        file_path: str,
        content: str,
        change_request: str,
    ) -> PatchCandidate:
        prompt = PATCH_PROMPT.format(
            file_path=file_path,
            content=content[:8000],
            request=change_request,
        )

        response = await self.llm.generate(prompt, mode=LLMMode.THINKING)

        # Extract code from markdown block
        modified = response
        if "```" in response:
            blocks = response.split("```")
            for block in blocks[1::2]:
                if block.strip().startswith("python"):
                    modified = block.strip()[6:].strip()
                    break
                elif block.strip():
                    modified = block.strip()
                    break

        return PatchCandidate(
            file_path=file_path,
            original=content,
            modified=modified,
            rationale=change_request,
        )

    def apply_patch(self, repo_path: str | Path, patch: PatchCandidate) -> None:
        target = Path(repo_path) / patch.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch.modified, encoding="utf-8")
        logger.info("Applied patch to %s", patch.file_path)
