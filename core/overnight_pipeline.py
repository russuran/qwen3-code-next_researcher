"""Overnight pipeline: autonomous research -> implement -> benchmark -> report.

Single entry point that runs overnight:
1. Deep research on the topic
2. Generate real implementations using actual libraries
3. Create synthetic test data
4. Run benchmarks with real metrics (CER, accuracy, F1)
5. Iterate: analyze failures, improve, re-test
6. Produce final report with ranked approaches
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.agent import AgentConfig, ResearchAgent
from core.llm import LLM, LLMMode
from core.hypothesis_loop import HypothesisRegistry, HypothesisStatus, TrackedHypothesis
from repo_adaptation.git_versioning import GitVersioning
from repo_adaptation.test_oracle import TestOracle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts for REAL code generation (not stubs)
# ---------------------------------------------------------------------------

REAL_IMPLEMENTATION_PROMPT = """\
You are a senior Python engineer. Write a COMPLETE, WORKING implementation.

Task: {title}
Description: {description}
Approach: {approach}

CRITICAL REQUIREMENTS:
- Use REAL libraries: {libraries}
- Code must be RUNNABLE without modification
- Include all imports
- Include a `run(input_path: str) -> dict` function that:
  - Takes a path to input data (image or text file)
  - Returns a dict with extracted fields and confidence scores
- Include a `benchmark(data_dir: str) -> dict` function that:
  - Takes a directory with test data
  - Returns metrics: accuracy, cer (character error rate), processing_time
- Handle errors gracefully (try/except with meaningful messages)
- If a library is not installed, print instructions and return empty results

Output ONLY Python code in ```python ... ```
"""

SYNTHETIC_DATA_PROMPT = """\
You are a data engineer. Generate a Python script that creates synthetic test data \
for {task}.

The script should:
- Create a directory `test_data/` with sample files
- Generate {num_samples} synthetic samples
- Each sample should have:
  - Input data (text file simulating OCR output, or image if possible)
  - Ground truth labels (JSON file with correct field values)
- For passport fields, generate realistic:
  - Full name (Russian names in Cyrillic)
  - Passport number (format: XX XX XXXXXX)
  - Date of birth (DD.MM.YYYY)
  - Gender (M/F)
  - Issue date, authority code
- Add realistic OCR noise: character substitutions (0↔O, 1↔I, З↔3), missing chars, extra spaces

Include a `generate()` function that creates the data and returns the path.

Output ONLY Python code in ```python ... ```
"""

REAL_BENCHMARK_PROMPT = """\
You are a QA engineer. Write a benchmark script for {title}.

The benchmark must:
- Load test data from `test_data/` directory
- Run the implementation's `run()` function on each sample
- Compare output to ground truth
- Calculate REAL metrics:
  - Character Error Rate (CER): edit_distance(predicted, actual) / len(actual)
  - Field-level accuracy: exact match per field
  - Overall accuracy: all fields correct
  - Processing time per sample
- Save results to `benchmark_results.json`
- Use pytest for test assertions

Required imports: pytest, json, time, pathlib
Use Levenshtein distance for CER (implement inline if python-Levenshtein not available).

Output ONLY Python code in ```python ... ```
"""

IMPROVEMENT_PROMPT = """\
You are a senior engineer reviewing benchmark results. Suggest code improvements.

Implementation: {code_snippet}
Benchmark results: {results}
Failures: {failures}

Provide an IMPROVED version of the implementation that:
- Fixes the specific failures identified
- Improves accuracy based on the error patterns
- Keeps the same interface (run, benchmark functions)

Output ONLY the improved Python code in ```python ... ```
"""


class OvernightPipeline:
    """Autonomous overnight run: research -> implement -> benchmark -> improve -> report."""

    def __init__(
        self,
        llm: LLM,
        workspace: str = "workspace/overnight",
        max_improvement_iterations: int = 3,
    ) -> None:
        self.llm = llm
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_improvement_iterations
        self.log: list[dict[str, Any]] = []

    async def run(
        self,
        topic: str,
        libraries: str = "pytesseract, easyocr, Pillow, opencv-python",
        num_samples: int = 50,
        sources: list[str] | None = None,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Full overnight pipeline. Returns final report."""
        start_time = datetime.now(timezone.utc)
        self._log("start", f"Overnight pipeline: {topic}")

        results = {
            "topic": topic,
            "started_at": start_time.isoformat(),
            "research": {},
            "implementations": [],
            "best_approach": None,
            "final_metrics": {},
        }

        try:
            # Phase 1: Research
            self._log("phase", "Phase 1: Deep Research")
            if on_progress:
                await on_progress("research", "Starting deep research...")

            research_output = await self._run_research(topic, sources)
            results["research"] = research_output
            hypotheses = research_output.get("hypotheses", [])
            self._log("research_done", f"{len(hypotheses)} hypotheses generated")

            if not hypotheses:
                self._log("error", "No hypotheses generated, aborting")
                results["error"] = "No hypotheses from research"
                return results

            # Phase 2: Generate synthetic test data
            self._log("phase", "Phase 2: Generating test data")
            if on_progress:
                await on_progress("data", f"Generating {num_samples} synthetic samples...")

            data_dir = await self._generate_test_data(topic, num_samples)
            self._log("data_done", f"Test data in {data_dir}")

            # Phase 3: Implement each hypothesis
            self._log("phase", "Phase 3: Implementing hypotheses")
            implementations = []

            for hyp in hypotheses[:3]:  # Top 3 hypotheses
                self._log("implement", f"Implementing: {hyp.get('title', '')}")
                if on_progress:
                    await on_progress("implement", f"Implementing H{hyp.get('id', '?')}: {hyp.get('title', '')}")

                impl = await self._implement_and_benchmark(
                    hyp, libraries, data_dir
                )
                implementations.append(impl)

            results["implementations"] = implementations

            # Phase 4: Iterate improvements on best approach
            self._log("phase", "Phase 4: Iterative improvement")
            best = max(implementations, key=lambda x: x.get("metrics", {}).get("accuracy", 0))

            if best.get("metrics", {}).get("accuracy", 0) > 0:
                for iteration in range(self.max_iterations):
                    if on_progress:
                        await on_progress("improve", f"Improvement iteration {iteration + 1}/{self.max_iterations}")

                    improved = await self._improve_implementation(best, data_dir, libraries)
                    if improved.get("metrics", {}).get("accuracy", 0) > best.get("metrics", {}).get("accuracy", 0):
                        best = improved
                        self._log("improved", f"Iteration {iteration + 1}: accuracy improved to {best['metrics']['accuracy']:.2%}")
                    else:
                        self._log("plateau", f"Iteration {iteration + 1}: no improvement, stopping")
                        break

            results["best_approach"] = best
            results["final_metrics"] = best.get("metrics", {})

            # Phase 5: Final report
            self._log("phase", "Phase 5: Final report")
            report = await self._generate_final_report(results)
            report_path = self.workspace / "final_report.md"
            report_path.write_text(report, encoding="utf-8")
            results["report_path"] = str(report_path)

        except Exception as e:
            self._log("error", str(e))
            results["error"] = str(e)

        results["finished_at"] = datetime.now(timezone.utc).isoformat()
        results["log"] = self.log

        # Save full results
        (self.workspace / "overnight_results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

        self._log("done", f"Pipeline complete. Best accuracy: {results.get('final_metrics', {}).get('accuracy', 'N/A')}")
        return results

    # ------------------------------------------------------------------
    # Phase 1: Research
    # ------------------------------------------------------------------

    async def _run_research(self, topic: str, sources: list[str] | None) -> dict:
        research_dir = self.workspace / "research"
        config = AgentConfig(
            output_dir=str(research_dir),
            sources=sources or ["github", "arxiv"],
            max_results_per_source=7,
            stages=["plan", "search", "filter", "deep_fetch", "analyze", "hypotheses"],
        )
        agent = ResearchAgent(config=config, llm=self.llm)
        output_path = await agent.run(topic)

        # Load hypotheses
        hyp_path = list(research_dir.rglob("04_hypotheses.json"))
        if hyp_path:
            return json.loads(hyp_path[0].read_text(encoding="utf-8"))
        return {"hypotheses": []}

    # ------------------------------------------------------------------
    # Phase 2: Synthetic data
    # ------------------------------------------------------------------

    async def _generate_test_data(self, topic: str, num_samples: int) -> str:
        data_dir = self.workspace / "test_data"
        data_dir.mkdir(exist_ok=True)

        prompt = SYNTHETIC_DATA_PROMPT.format(task=topic, num_samples=num_samples)
        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        code = self._extract_code(raw)

        gen_path = self.workspace / "generate_data.py"
        gen_path.write_text(code, encoding="utf-8")

        # Run data generation
        import subprocess
        try:
            subprocess.run(
                [sys.executable, str(gen_path)],
                cwd=str(self.workspace),
                capture_output=True, text=True, timeout=120,
            )
        except Exception as e:
            logger.warning("Data generation script failed: %s, creating minimal data", e)
            self._create_minimal_test_data(data_dir, num_samples)

        # Verify data exists
        if not list(data_dir.glob("*")):
            self._create_minimal_test_data(data_dir, num_samples)

        return str(data_dir)

    def _create_minimal_test_data(self, data_dir: Path, count: int) -> None:
        """Fallback: create minimal synthetic passport data."""
        import random
        names = ["ИВАНОВ ИВАН ИВАНОВИЧ", "ПЕТРОВ ПЕТР ПЕТРОВИЧ", "СИДОРОВА МАРИЯ АЛЕКСАНДРОВНА",
                 "КОЗЛОВ ДМИТРИЙ СЕРГЕЕВИЧ", "НОВИКОВА АННА ВЛАДИМИРОВНА"]
        for i in range(min(count, 20)):
            name = random.choice(names)
            series = f"{random.randint(10,99)} {random.randint(10,99)}"
            number = f"{random.randint(100000,999999)}"
            dob = f"{random.randint(1,28):02d}.{random.randint(1,12):02d}.{random.randint(1960,2005)}"

            # Ground truth
            gt = {"name": name, "series": series, "number": number, "dob": dob}

            # Simulate OCR noise
            noisy_name = name.replace("О", "0").replace("З", "3") if random.random() > 0.5 else name
            noisy_number = number.replace("0", "O") if random.random() > 0.5 else number
            ocr_text = f"ФИО: {noisy_name}\nСерия: {series}\nНомер: {noisy_number}\nДата рождения: {dob}"

            sample_dir = data_dir / f"sample_{i:03d}"
            sample_dir.mkdir(exist_ok=True)
            (sample_dir / "ocr_output.txt").write_text(ocr_text, encoding="utf-8")
            (sample_dir / "ground_truth.json").write_text(
                json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # ------------------------------------------------------------------
    # Phase 3: Implement + benchmark
    # ------------------------------------------------------------------

    async def _implement_and_benchmark(
        self, hyp: dict, libraries: str, data_dir: str,
    ) -> dict:
        slug = "".join(c if c.isalnum() or c == "-" else "-" for c in hyp.get("title", "impl")[:30].lower()).strip("-")
        impl_dir = self.workspace / slug
        impl_dir.mkdir(parents=True, exist_ok=True)

        # Generate implementation
        prompt = REAL_IMPLEMENTATION_PROMPT.format(
            title=hyp.get("title", ""),
            description=hyp.get("description", ""),
            approach=hyp.get("approach", ""),
            libraries=libraries,
        )
        code = self._extract_code(await self.llm.generate(prompt, mode=LLMMode.THINKING))
        (impl_dir / "implementation.py").write_text(code, encoding="utf-8")

        # Generate benchmark
        bench_prompt = REAL_BENCHMARK_PROMPT.format(title=hyp.get("title", ""))
        bench_code = self._extract_code(await self.llm.generate(bench_prompt, mode=LLMMode.THINKING))
        (impl_dir / "test_benchmark.py").write_text(bench_code, encoding="utf-8")

        # Copy test data reference
        (impl_dir / "data_dir.txt").write_text(data_dir, encoding="utf-8")

        # Git init + branch
        git = GitVersioning(impl_dir)
        git.init_if_missing()
        git.create_branch(f"ai/task/{slug}")
        git.commit(f"implement: {hyp.get('title', '')[:50]}")

        # Run benchmark
        metrics = await self._run_benchmark(impl_dir)

        return {
            "hypothesis": hyp,
            "code_path": str(impl_dir / "implementation.py"),
            "metrics": metrics,
            "slug": slug,
        }

    async def _run_benchmark(self, impl_dir: Path) -> dict:
        """Run benchmark and return metrics."""
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_benchmark.py", "-v", "--tb=short", "-x"],
                cwd=str(impl_dir),
                capture_output=True, text=True, timeout=300,
            )

            # Try to load benchmark_results.json if generated
            results_path = impl_dir / "benchmark_results.json"
            if results_path.exists():
                return json.loads(results_path.read_text())

            return {
                "passed": result.returncode == 0,
                "accuracy": 1.0 if result.returncode == 0 else 0.0,
                "output": result.stdout[-2000:],
            }
        except Exception as e:
            return {"passed": False, "accuracy": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Phase 4: Improve
    # ------------------------------------------------------------------

    async def _improve_implementation(self, best: dict, data_dir: str, libraries: str) -> dict:
        code_path = best.get("code_path", "")
        if not code_path or not Path(code_path).exists():
            return best

        code = Path(code_path).read_text(encoding="utf-8")
        metrics = best.get("metrics", {})
        failures = metrics.get("output", "")[:2000]

        prompt = IMPROVEMENT_PROMPT.format(
            code_snippet=code[:4000],
            results=json.dumps(metrics, indent=2)[:1000],
            failures=failures[:1000],
        )

        improved_code = self._extract_code(await self.llm.generate(prompt, mode=LLMMode.THINKING))

        # Save improved version
        impl_dir = Path(code_path).parent
        (impl_dir / "implementation.py").write_text(improved_code, encoding="utf-8")

        git = GitVersioning(impl_dir)
        git.commit("improve: based on benchmark feedback")

        new_metrics = await self._run_benchmark(impl_dir)

        return {
            **best,
            "metrics": new_metrics,
            "improved": True,
        }

    # ------------------------------------------------------------------
    # Phase 5: Report
    # ------------------------------------------------------------------

    async def _generate_final_report(self, results: dict) -> str:
        implementations = results.get("implementations", [])
        best = results.get("best_approach", {})

        report_prompt = f"""\
Write a technical report summarizing the overnight research and implementation run.

Topic: {results.get('topic', '')}
Hypotheses tested: {len(implementations)}
Best approach: {best.get('hypothesis', {}).get('title', 'N/A')}
Best accuracy: {best.get('metrics', {}).get('accuracy', 'N/A')}

Results per hypothesis:
"""
        for impl in implementations:
            hyp = impl.get("hypothesis", {})
            metrics = impl.get("metrics", {})
            report_prompt += f"- {hyp.get('title', '')}: accuracy={metrics.get('accuracy', 'N/A')}\n"

        report_prompt += """
Write sections: Summary, Methodology, Results, Best Approach Details, Recommendations, Next Steps.
"""
        return await self.llm.generate(report_prompt, mode=LLMMode.THINKING)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, action: str, message: str) -> None:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "action": action, "message": message}
        self.log.append(entry)
        logger.info("[overnight] %s: %s", action, message)

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```" in text:
            for block in text.split("```")[1::2]:
                block = block.strip()
                if block.startswith("python"):
                    return block[6:].strip()
                elif block.strip():
                    return block.strip()
        return text
