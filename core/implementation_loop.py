"""Implementation loop: hypothesis -> code generation -> sandbox -> eval -> PR.

Full cycle:
1. Take top-priority hypothesis from research output
2. Generate implementation code via LLM (patch_editor)
3. Create git branch ai/task/{run_id}-{slug}
4. Run tests in sandbox
5. Evaluate results
6. Update hypothesis status (validated/rejected)
7. Package PR if validated
8. Move to next hypothesis
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.hypothesis_loop import HypothesisRegistry, HypothesisStatus, TrackedHypothesis
from core.llm import LLM, LLMMode
from repo_adaptation.git_versioning import GitVersioning
from repo_adaptation.patch_editor import PatchEditor
from repo_adaptation.test_oracle import TestOracle, TestResult
from repo_adaptation.pr_packager import PRPackager
from sandbox.evaluator import Evaluator

logger = logging.getLogger(__name__)


IMPLEMENTATION_PROMPT = """\
You are a software engineer. Based on the following research hypothesis, \
generate a complete Python implementation.

Hypothesis: {title}
Description: {description}
Approach: {approach}
Expected outcome: {expected_outcome}

Generate a Python module that implements this hypothesis.
The code should be:
- Self-contained (minimal external dependencies)
- Well-documented with docstrings
- Include a simple test/demo function called `demo()` that shows usage
- Include type hints

Output ONLY the Python code wrapped in ```python ... ```
"""

BENCHMARK_PROMPT = """\
You are a software engineer. Generate a benchmark/test script for the following implementation.

Hypothesis: {title}
Approach: {approach}
Validation method: {validation_method}

Generate a Python test script that:
- Tests the core functionality
- Measures performance metrics (accuracy, speed, etc.)
- Returns results as a JSON-serializable dict
- Uses pytest-compatible test functions

Output ONLY the Python code wrapped in ```python ... ```
"""


class ImplementationResult(BaseModel):
    hypothesis_id: str
    status: str = "pending"  # pending | implemented | tested | validated | rejected
    branch_name: str = ""
    commit_sha: str = ""
    code_path: str = ""
    test_result: dict[str, Any] = {}
    eval_metrics: dict[str, Any] = {}
    pr_path: str = ""
    error: str = ""


class ImplementationLoop:
    """Full cycle: hypothesis -> implementation -> test -> eval -> PR."""

    def __init__(
        self,
        llm: LLM,
        workspace: str = "workspace",
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.patch_editor = PatchEditor(llm)
        self.evaluator = Evaluator()
        self.results: list[ImplementationResult] = []

    async def run(
        self,
        registry: HypothesisRegistry,
        on_progress: Any = None,
    ) -> list[ImplementationResult]:
        """Execute implementation loop for all draft hypotheses."""
        iteration = 0
        max_iter = min(len([h for h in registry.hypotheses if h.status == HypothesisStatus.DRAFT]), 5)

        while iteration < max_iter:
            hyp = registry.get_next()
            if not hyp:
                break

            iteration += 1
            logger.info("Implementation loop %d/%d: %s", iteration, max_iter, hyp.title)

            result = await self._implement_hypothesis(hyp, registry)
            self.results.append(result)

            if on_progress:
                await on_progress(hyp, result)

        return self.results

    async def _implement_hypothesis(
        self,
        hyp: TrackedHypothesis,
        registry: HypothesisRegistry,
    ) -> ImplementationResult:
        result = ImplementationResult(hypothesis_id=hyp.id)
        slug = "".join(c if c.isalnum() or c == "-" else "-" for c in hyp.title.lower()[:30]).strip("-")
        project_dir = self.workspace / slug
        project_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Mark as running
            registry.update_status(hyp.id, HypothesisStatus.RUNNING)

            # 2. Init git repo
            git = GitVersioning(project_dir)
            git.init_if_missing()
            branch = f"ai/task/{hyp.id}-{slug}"
            try:
                git.create_branch(branch)
            except Exception:
                git.checkout(branch)
            result.branch_name = branch

            # 3. Generate implementation code
            logger.info("Generating implementation for: %s", hyp.title)
            code = await self._generate_code(hyp)
            code_path = project_dir / "implementation.py"
            code_path.write_text(code, encoding="utf-8")
            result.code_path = str(code_path)
            result.status = "implemented"

            # 4. Generate benchmark/test
            test_code = await self._generate_benchmark(hyp)
            test_path = project_dir / "test_implementation.py"
            test_path.write_text(test_code, encoding="utf-8")

            # 5. Commit
            sha = git.commit(f"ai: implement {hyp.title[:50]}", run_id=hyp.id)
            result.commit_sha = sha

            # 6. Run tests
            logger.info("Running tests for: %s", hyp.title)
            oracle = TestOracle(project_dir)
            test_result = oracle.run_tests(
                command=["python", "-m", "pytest", "test_implementation.py", "-v", "--tb=short"],
                timeout=120,
            )
            result.test_result = test_result.model_dump()
            result.status = "tested"

            # 7. Evaluate
            if test_result.passed:
                result.eval_metrics = {
                    "tests_passed": True,
                    "total_tests": test_result.total,
                    "failures": test_result.failures,
                }
                result.status = "validated"
                registry.update_status(
                    hyp.id, HypothesisStatus.VALIDATED,
                    validation_result=result.eval_metrics,
                )
                logger.info("H%s VALIDATED: tests passed", hyp.id)
            else:
                result.eval_metrics = {
                    "tests_passed": False,
                    "total_tests": test_result.total,
                    "failures": test_result.failures,
                    "output": test_result.output[-1000:],
                }
                result.status = "rejected"
                registry.update_status(
                    hyp.id, HypothesisStatus.REJECTED,
                    validation_result=result.eval_metrics,
                )
                logger.info("H%s REJECTED: tests failed", hyp.id)

            # 8. Package PR (for validated hypotheses)
            if result.status == "validated":
                packager = PRPackager(git)
                pr = packager.package(
                    title=f"ai: {hyp.title}",
                    description=f"{hyp.description}\n\nApproach: {hyp.approach}\n\nBased on: {', '.join(hyp.based_on)}",
                    base_branch="main",
                    candidate_branch=branch,
                    test_results=result.test_result,
                )
                pr_dir = project_dir / "pr_package"
                packager.save(pr, pr_dir)
                result.pr_path = str(pr_dir)
                logger.info("PR packaged: %s", pr_dir)

        except Exception as e:
            logger.error("Implementation failed for H%s: %s", hyp.id, e, exc_info=True)
            result.status = "rejected"
            result.error = str(e)
            registry.update_status(
                hyp.id, HypothesisStatus.REJECTED,
                validation_result={"error": str(e)},
            )

        return result

    async def _generate_code(self, hyp: TrackedHypothesis) -> str:
        prompt = IMPLEMENTATION_PROMPT.format(
            title=hyp.title,
            description=hyp.description,
            approach=hyp.approach,
            expected_outcome=hyp.expected_outcome,
        )
        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        return self._extract_code(raw)

    async def _generate_benchmark(self, hyp: TrackedHypothesis) -> str:
        prompt = BENCHMARK_PROMPT.format(
            title=hyp.title,
            approach=hyp.approach,
            validation_method=hyp.validation_method,
        )
        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        return self._extract_code(raw)

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```" in text:
            blocks = text.split("```")
            for block in blocks[1::2]:
                block = block.strip()
                if block.startswith("python"):
                    return block[6:].strip()
                elif block.strip():
                    return block.strip()
        return text
