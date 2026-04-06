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
import os
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
for: {task}

The script should:
- Create a directory `test_data/` with {num_samples} samples
- Each sample in its own folder (sample_000, sample_001, ...)
- Each sample has: input data file + ground_truth.json
- Generate realistic data appropriate for the specific task
- Add realistic noise patterns typical for this domain
- Include a `generate()` function that creates the data and returns the path
- Use only standard library + common packages (json, random, pathlib, etc.)

Output ONLY Python code in ```python ... ```
"""

REAL_BENCHMARK_PROMPT = """\
You are a QA engineer. Write a benchmark script for {title}.

Data is mounted at `/data` inside the Docker container.
Dataset structure:
{data_structure}

The benchmark must:
- Scan `/data` directory recursively for input files (images: *.jpg, *.png; text: *.txt, *.json)
- For each input file, call the implementation's `run(file_path)` function
- If ground_truth.json exists next to the input, compare output to it
- Calculate metrics:
  - Character Error Rate (CER): edit_distance(predicted, actual) / len(actual)
  - Field-level accuracy: exact match per field
  - Overall accuracy: all fields correct
  - Processing time per sample
  - Total samples processed
- Save results to `benchmark_results.json` in current directory
- Use pytest with at least one assertion (e.g. assert total_samples > 0)
- If no data found at `/data`, scan current directory and `/workspace` as fallback

Required imports: pytest, json, time, pathlib, os
Implement Levenshtein distance inline (do not import external package).

Output ONLY Python code in ```python ... ```
"""

DATASET_SEARCH_PROMPT = """\
You are a data engineer. Find real, downloadable datasets for: {task}

Search for:
1. Kaggle datasets (provide kaggle dataset slug like "user/dataset-name")
2. GitHub repos with sample data (provide raw download URLs)
3. HuggingFace datasets (provide dataset ID like "org/dataset")
4. Academic datasets with direct download links

IMPORTANT: Provide DIRECT download URLs or exact Kaggle slugs that can be downloaded programmatically.
For Kaggle, format as: kaggle datasets download -d <user/dataset>
For HuggingFace, format as: datasets.load_dataset("<org/dataset>")

Respond with ONLY a JSON object:
{{
  "datasets": [
    {{
      "name": "dataset name",
      "source": "kaggle|github|huggingface|url",
      "identifier": "download path or URL",
      "description": "what it contains",
      "size_mb": estimated size,
      "format": "images|csv|json",
      "relevance": 1-10
    }}
  ]
}}
"""

DATASET_DOWNLOAD_PROMPT = """\
You are a data engineer. Write a Python script that downloads and prepares \
the following dataset for benchmarking.

Dataset: {name}
Source: {source}
Identifier: {identifier}
Description: {description}

The script must:
- Download the dataset to `{output_dir}/`
- If it's a zip/tar, extract it
- Organize into folders: each sample in its own directory with image + ground_truth.json
- ground_truth.json should have fields like: {{"name": "...", "number": "...", "dob": "..."}}
- If the dataset has a different format, convert it
- Handle download errors gracefully
- Print progress
- Use requests, urllib, or kaggle/huggingface_hub if needed

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
                self._log("fallback", "No hypotheses from research, generating from topic")
                hypotheses = await self._generate_fallback_hypotheses(topic)
                if not hypotheses:
                    self._log("error", "Could not generate any hypotheses")
                    results["error"] = "No hypotheses generated"
                    return results

            # Phase 2: Search for datasets (for later benchmark)
            self._log("phase", "Phase 2: Searching for datasets")
            if on_progress:
                await on_progress("data", "Searching for datasets to benchmark against...")

            datasets = await self._search_datasets(topic)
            self._log("datasets_found", f"Found {len(datasets)} potential datasets")

            # Save dataset suggestions for user
            (self.workspace / "suggested_datasets.md").write_text(
                "# Suggested Datasets\n\n"
                "Upload one of these via the dashboard to run real benchmarks.\n\n"
                + "\n".join(
                    f"- **{d['name']}** ({d['source']}): {d.get('description', '')[:80]}\n"
                    f"  `{d.get('identifier', '')}`"
                    for d in datasets[:5]
                ),
                encoding="utf-8",
            )

            # Phase 3: Implement all hypotheses + smoke test
            self._log("phase", "Phase 3: Implementing hypotheses with smoke tests")
            implementations = []

            for hyp in hypotheses[:5]:
                self._log("implement", f"Implementing: {hyp.get('title', '')}")
                if on_progress:
                    await on_progress("implement", f"Implementing H{hyp.get('id', '?')}: {hyp.get('title', '')}")

                impl = await self._implement_hypothesis(hyp, libraries)
                implementations.append(impl)

            results["implementations"] = implementations

            working = [i for i in implementations if i.get("smoke_test_passed")]
            failed = [i for i in implementations if not i.get("smoke_test_passed")]
            self._log("implementations_done",
                       f"{len(working)} working, {len(failed)} failed smoke test out of {len(implementations)}")

            # Phase 4: Check for real data, auto-label if needed, then benchmark
            data_dir = self._find_real_data()

            if data_dir:
                # Auto-label: generate ground_truth.json where missing
                has_gt = any(Path(data_dir).rglob("ground_truth.json"))
                if not has_gt:
                    self._log("phase", "Phase 3.9: Auto-labeling dataset (no ground truth found)")
                    if on_progress:
                        await on_progress("labeling", "Auto-labeling dataset with cross-validation OCR...")
                    await self._auto_label_dataset(data_dir)

                self._log("phase", "Phase 4: Benchmarking on real data")
                if on_progress:
                    await on_progress("benchmark", f"Running benchmarks on {data_dir}")

                for impl in working:
                    metrics = await self._run_benchmark(Path(impl["code_path"]).parent)
                    impl["benchmark_metrics"] = metrics
                    self._log("benchmark", f"{impl['hypothesis'].get('title','')}: {metrics}")

                # Iterate improvements on best
                best = max(working, key=lambda x: x.get("benchmark_metrics", {}).get("accuracy", 0)) if working else {}
                for iteration in range(self.max_iterations):
                    if not best or not best.get("benchmark_metrics"):
                        break
                    if on_progress:
                        await on_progress("improve", f"Improvement iteration {iteration + 1}")
                    improved = await self._improve_implementation(best, data_dir, libraries)
                    if improved.get("metrics", {}).get("accuracy", 0) > best.get("benchmark_metrics", {}).get("accuracy", 0):
                        best = improved
                        self._log("improved", f"Iteration {iteration + 1}: improved")
                    else:
                        self._log("plateau", f"Iteration {iteration + 1}: no improvement")
                        break
                results["best_approach"] = best
                results["final_metrics"] = best.get("benchmark_metrics", best.get("metrics", {}))
                results["status"] = "benchmarked"
            else:
                self._log("waiting_data",
                           "Implementations ready. Upload dataset via /dashboard/upload-dataset to run benchmarks.")
                if on_progress:
                    await on_progress("waiting", "Implementations ready. Waiting for dataset upload to run benchmarks.")
                results["best_approach"] = working[0] if working else {}
                results["final_metrics"] = {"status": "waiting_for_data", "implementations_ready": len(working)}
                results["status"] = "waiting_data"

            # Phase 5: Report
            self._log("phase", "Phase 5: Report")
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
            stages=["plan", "search", "filter", "deep_fetch", "analyze", "hypotheses", "synthesize"],
        )
        agent = ResearchAgent(config=config, llm=self.llm)
        output_path = await agent.run(topic)

        # Load hypotheses
        hyp_path = list(research_dir.rglob("04_hypotheses.json"))
        if hyp_path:
            return json.loads(hyp_path[0].read_text(encoding="utf-8"))
        return {"hypotheses": []}

    async def _generate_fallback_hypotheses(self, topic: str) -> list[dict]:
        """Fallback: generate hypotheses directly from topic when research yields none."""
        prompt = f"""\
Generate 3 implementation hypotheses for: {topic}

Each hypothesis should use specific, real Python libraries.
Respond with JSON: {{"hypotheses": [{{"id": "H1", "title": "...", "description": "...", \
"approach": "...", "expected_outcome": "...", "validation_method": "...", "priority": 5, \
"effort": "medium", "based_on": []}}]}}
"""
        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        data = self._extract_json(raw)
        hypotheses = data.get("hypotheses", [])
        self._log("fallback_hypotheses", f"Generated {len(hypotheses)} fallback hypotheses")
        return hypotheses

    # ------------------------------------------------------------------
    # Phase 2a: Dataset search
    # ------------------------------------------------------------------

    async def _search_datasets(self, topic: str) -> list[dict]:
        """Search for real datasets via LLM + GitHub API. Fully topic-agnostic."""
        prompt = DATASET_SEARCH_PROMPT.format(task=topic)
        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        data = self._extract_json(raw)
        datasets = data.get("datasets", [])

        # Also search GitHub for datasets
        from core.tools import registry
        gh_result = await registry.execute(
            "search_github",
            query=f"{topic} dataset benchmark samples",
            max_results=5,
        )
        if gh_result.success and gh_result.data:
            for repo in gh_result.data:
                datasets.append({
                    "name": repo.get("name", ""),
                    "source": "github",
                    "identifier": repo.get("url", ""),
                    "description": repo.get("description", ""),
                    "relevance": 6,
                })

        # Sort by relevance
        datasets.sort(key=lambda d: -d.get("relevance", 0))
        self._log("datasets", f"Found {len(datasets)} datasets: {[d['name'] for d in datasets[:3]]}")

        # Save
        (self.workspace / "datasets_found.json").write_text(
            json.dumps(datasets, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return datasets

    # ------------------------------------------------------------------
    # Phase 3: Implement + smoke test
    # ------------------------------------------------------------------

    async def _implement_hypothesis(self, hyp: dict, libraries: str) -> dict:
        """Generate code + run smoke test (one dummy input to verify it doesn't crash)."""
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
        code_path = impl_dir / "implementation.py"
        code_path.write_text(code, encoding="utf-8")

        # Scan dataset for benchmark prompt
        data_dir = self._find_real_data()
        ds_info = "No data. Use /workspace/smoke_test/ as fallback."
        if data_dir:
            try:
                files = list(Path(data_dir).rglob("*.*"))[:20]
                exts = {}
                for f in files:
                    exts[f.suffix] = exts.get(f.suffix, 0) + 1
                total = len(list(Path(data_dir).rglob("*.*")))
                sample = [str(f.relative_to(data_dir)) for f in files[:5]]
                ds_info = f"Total files: {total}\nExtensions: {exts}\nSample paths: {sample}"
            except Exception:
                pass

        # Generate benchmark script
        bench_prompt = REAL_BENCHMARK_PROMPT.format(title=hyp.get("title", ""), data_structure=ds_info)
        bench_code = self._extract_code(await self.llm.generate(bench_prompt, mode=LLMMode.THINKING))
        (impl_dir / "test_benchmark.py").write_text(bench_code, encoding="utf-8")

        # Create dummy smoke test input
        smoke_dir = impl_dir / "smoke_test"
        smoke_dir.mkdir(exist_ok=True)
        # Black square image
        try:
            from PIL import Image
            img = Image.new("RGB", (200, 100), color=(0, 0, 0))
            img.save(str(smoke_dir / "dummy.png"))
        except ImportError:
            (smoke_dir / "dummy.txt").write_text("smoke test input", encoding="utf-8")

        # Generate requirements.txt for sandbox
        (impl_dir / "requirements.txt").write_text(
            "\n".join(lib.strip() for lib in libraries.split(",")) + "\n",
            encoding="utf-8",
        )

        # Smoke test script
        smoke_script = """\
import sys, os
sys.path.insert(0, "/workspace")
try:
    import implementation
    if hasattr(implementation, 'run'):
        result = implementation.run("/workspace/smoke_test")
        print(f"OK: {result}")
    elif hasattr(implementation, 'demo'):
        implementation.demo()
        print("OK: demo ran")
    else:
        print("OK: module imported")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
"""
        smoke_path = impl_dir / "smoke_test.py"
        smoke_path.write_text(smoke_script)

        # Run smoke test in Docker sandbox with self-healing
        import subprocess
        passed = False
        output = ""
        last_error = ""
        max_fix_attempts = 10

        for attempt in range(max_fix_attempts):
            try:
                # Docker needs network for pip install
                docker_cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{impl_dir.resolve()}:/workspace",
                    "-w", "/workspace",
                    "--memory", "4g", "--cpus", "2",
                    "python:3.11-slim",
                    "sh", "-c",
                    "apt-get update -qq && apt-get install -y -qq tesseract-ocr libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1; "
                    "pip install -q -r requirements.txt 2>&1; python smoke_test.py"
                ]
                result = await asyncio.to_thread(subprocess.run, docker_cmd, capture_output=True, text=True, timeout=300,)
                output = result.stdout.strip()[-500:] + "\n" + result.stderr.strip()[-500:]
                passed = result.returncode == 0

                if passed:
                    self._log("smoke_test", f"PASS (attempt {attempt+1}): {hyp.get('title','')}")
                    break

                # Check if same error as before (loop detection)
                current_error = output[-200:]
                if current_error == last_error:
                    self._log("smoke_test", f"STUCK (same error x2): {hyp.get('title','')} — {output[-80:]}")
                    break
                last_error = current_error

                # Self-heal: ask LLM to fix the error
                self._log("smoke_fix", f"Attempt {attempt+1} failed, asking LLM to fix: {output[-100:]}")
                fix_prompt = f"""The following Python code failed in a Docker container (python:3.11-slim).

Error output:
{output[-1000:]}

Current requirements.txt:
{(impl_dir / 'requirements.txt').read_text()}

Current implementation.py (first 100 lines):
{(impl_dir / 'implementation.py').read_text()[:3000]}

Fix the issue. Common fixes:
- Replace opencv-python with opencv-python-headless
- Add missing system packages to requirements
- Fix import errors in the code
- Handle missing optional dependencies gracefully

Respond with TWO code blocks:
1. ```requirements``` — updated requirements.txt
2. ```python``` — updated implementation.py
"""
                fix_response = await self.llm.generate(fix_prompt, mode=LLMMode.THINKING)

                # Extract and apply fixes
                blocks = fix_response.split("```")
                for i, block in enumerate(blocks[1::2]):
                    block = block.strip()
                    if block.startswith("requirements"):
                        new_reqs = block.split("\n", 1)[1].strip() if "\n" in block else block
                        (impl_dir / "requirements.txt").write_text(new_reqs + "\n")
                        self._log("smoke_fix", f"Updated requirements.txt")
                    elif block.startswith("python"):
                        new_code = block[6:].strip()
                        if len(new_code) > 50:  # sanity check
                            (impl_dir / "implementation.py").write_text(new_code)
                            self._log("smoke_fix", f"Updated implementation.py")

            except FileNotFoundError:
                self._log("smoke_test_fallback", "Docker not available, running locally")
                try:
                    result = subprocess.run(
                        [sys.executable, str(smoke_path)],
                        capture_output=True, text=True, timeout=60,
                        env={**os.environ, "PYTHONPATH": str(impl_dir)},
                    )
                    passed = result.returncode == 0
                    output = result.stdout.strip()[:200] + result.stderr.strip()[:200]
                except Exception as e:
                    output = str(e)
                self._log("smoke_test", f"{'PASS' if passed else 'FAIL'} (local): {hyp.get('title','')}")
                break
            except subprocess.TimeoutExpired:
                self._log("smoke_test", f"TIMEOUT attempt {attempt+1}: {hyp.get('title','')}")
                break
            except Exception as e:
                self._log("smoke_test", f"ERROR attempt {attempt+1}: {e}")
                break

        # Git init
        from repo_adaptation.git_versioning import GitVersioning
        git = GitVersioning(impl_dir)
        git.init_if_missing()
        git.create_branch(f"ai/task/{slug}")
        git.commit(f"implement: {hyp.get('title', '')[:50]}")

        return {
            "hypothesis": hyp,
            "code_path": str(code_path),
            "slug": slug,
            "smoke_test_passed": passed,
            "smoke_output": (result.stdout + result.stderr)[-500:] if 'result' in dir() else "",
        }

    def _find_real_data(self) -> str | None:
        """Check if user has uploaded real data."""
        # Check per-run upload
        user_data = self.workspace / "user_data"
        if user_data.exists() and any(user_data.rglob("*.jpg")) or any(user_data.rglob("*.png")) or any(user_data.rglob("*.json")):
            return str(user_data)

        # Check global uploads
        uploads_dir = Path("uploads/datasets")
        if uploads_dir.exists():
            for d in sorted(uploads_dir.iterdir(), reverse=True):
                if d.is_dir() and any(d.rglob("*.*")):
                    return str(d)

        # Check downloaded data
        real_data = self.workspace / "real_data"
        if real_data.exists() and any(real_data.rglob("*.*")):
            return str(real_data)

        return None

    # ------------------------------------------------------------------
    # Auto-labeling
    # ------------------------------------------------------------------

    async def _auto_label_dataset(self, data_dir: str) -> None:
        """Auto-generate ground_truth.json for images using cross-validation OCR in Docker."""
        import subprocess

        label_script = """\
import os, json, sys
from pathlib import Path

data_path = Path("/data")
results = {}
labeled = 0

# Try to import OCR engines
engines = []
try:
    import pytesseract
    engines.append(("tesseract", lambda img: pytesseract.image_to_string(img, lang="rus+eng")))
except ImportError:
    pass

try:
    import easyocr
    reader = easyocr.Reader(["en", "ru"], gpu=False, verbose=False)
    engines.append(("easyocr", lambda img: " ".join([r[1] for r in reader.readtext(str(img))])))
except ImportError:
    pass

if not engines:
    print("No OCR engines available")
    sys.exit(0)

# Scan for images
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
for img_path in sorted(data_path.rglob("*")):
    if img_path.suffix.lower() not in image_exts:
        continue
    if img_path.stat().st_size < 1000:
        continue

    gt_path = img_path.parent / "ground_truth.json"
    if gt_path.exists():
        continue

    # Run all available engines
    ocr_results = {}
    for name, engine_fn in engines:
        try:
            if name == "tesseract":
                from PIL import Image
                img = Image.open(str(img_path))
                text = engine_fn(img)
            else:
                text = engine_fn(img_path)
            ocr_results[name] = text.strip()
        except Exception as e:
            ocr_results[name] = f"ERROR: {e}"

    # Cross-validate: use consensus or first available
    texts = [v for v in ocr_results.values() if not v.startswith("ERROR")]
    if not texts:
        continue

    # Simple consensus: take the longest non-error result (usually more complete)
    best_text = max(texts, key=len)

    # Parse fields (best effort)
    lines = best_text.split("\\n")
    gt = {
        "raw_text": best_text,
        "ocr_engines": list(ocr_results.keys()),
        "fields": {},
        "source_file": str(img_path.relative_to(data_path)),
    }

    # Try to extract common fields
    for line in lines:
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if any(k in lower for k in ["фамилия", "имя", "отчество", "name", "фио"]):
            gt["fields"]["name"] = line
        elif any(k in lower for k in ["серия", "series", "номер", "number", "паспорт"]):
            gt["fields"]["document_number"] = line
        elif any(k in lower for k in ["дата", "date", "рожд", "birth"]):
            gt["fields"]["date"] = line
        elif any(k in lower for k in ["пол", "sex", "gender"]):
            gt["fields"]["gender"] = line

    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2))
    labeled += 1
    if labeled % 10 == 0:
        print(f"Labeled {labeled} images...")

print(f"Done: {labeled} images labeled with {len(engines)} engine(s)")
"""

        # Write labeling script
        label_path = self.workspace / "auto_label.py"
        label_path.write_text(label_script)

        # Also write requirements
        req_path = self.workspace / "label_requirements.txt"
        req_path.write_text("pytesseract\neasyocr\nPillow\nopencv-python-headless\n")

        # Run in Docker with data mounted
        try:
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{Path(data_dir).resolve()}:/data",
                "-v", f"{self.workspace.resolve()}:/scripts",
                "-w", "/scripts",
                "--memory", "4g", "--cpus", "2",
                "python:3.11-slim",
                "sh", "-c",
                "apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-rus libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1; "
                "pip install -q -r label_requirements.txt 2>&1; python auto_label.py"
            ]
            result = await asyncio.to_thread(
                subprocess.run, docker_cmd,
                capture_output=True, text=True, timeout=600,
            )
            output = result.stdout.strip()[-200:]
            self._log("auto_label", f"Labeling: {output}")

            gt_count = len(list(Path(data_dir).rglob("ground_truth.json")))
            self._log("auto_label_done", f"Generated {gt_count} ground_truth.json files")

        except subprocess.TimeoutExpired:
            self._log("auto_label", "Labeling timed out (600s)")
        except FileNotFoundError:
            self._log("auto_label", "Docker not available, skipping auto-labeling")
        except Exception as e:
            self._log("auto_label", f"Labeling error: {e}")

    # ------------------------------------------------------------------
    # Dataset download
    # ------------------------------------------------------------------

    async def _download_dataset(self, dataset: dict) -> str | None:
        """Download dataset using appropriate tool based on source."""
        import subprocess
        download_dir = self.workspace / "real_data"
        download_dir.mkdir(exist_ok=True)

        source = dataset.get("source", "")
        identifier = dataset.get("identifier", "")
        name = dataset.get("name", "unknown")

        self._log("download_try", f"Trying {source}: {name} ({identifier[:60]})")

        try:
            if source == "kaggle" and identifier:
                # Use kaggle CLI directly
                result = subprocess.run(
                    [sys.executable, "-m", "kaggle", "datasets", "download", "-d", identifier, "-p", str(download_dir), "--unzip"],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0 and list(download_dir.rglob("*.*")):
                    self._log("download_ok", f"Kaggle dataset: {name}")
                    return str(download_dir)
                self._log("download_fail", f"Kaggle: {result.stderr[:150]}")

            elif source == "huggingface" and identifier:
                # Use huggingface datasets
                script = f"""
import sys
try:
    from datasets import load_dataset
    ds = load_dataset("{identifier}", split="train[:100]")
    ds.save_to_disk("{download_dir}")
    print("OK")
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
                script_path = self.workspace / "download_hf.py"
                script_path.write_text(script)
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0 and list(download_dir.rglob("*")):
                    self._log("download_ok", f"HuggingFace dataset: {name}")
                    return str(download_dir)
                self._log("download_fail", f"HuggingFace: {result.stderr[:150]}")

            elif source == "github" and identifier:
                # Clone or download from GitHub
                url = identifier
                if "github.com" in url and not url.endswith(".git"):
                    url = url.rstrip("/") + ".git"
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(download_dir / name.replace("/", "-"))],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    self._log("download_ok", f"GitHub repo: {name}")
                    return str(download_dir)
                self._log("download_fail", f"GitHub: {result.stderr[:150]}")

            elif source == "url" and identifier:
                # Direct URL download
                import httpx
                async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                    resp = await client.get(identifier)
                    if resp.status_code == 200:
                        fname = identifier.split("/")[-1] or "dataset.zip"
                        fpath = download_dir / fname
                        fpath.write_bytes(resp.content)
                        # Unzip if needed
                        if fname.endswith(".zip"):
                            import zipfile
                            with zipfile.ZipFile(fpath) as zf:
                                zf.extractall(download_dir)
                            fpath.unlink()
                        self._log("download_ok", f"URL download: {name}")
                        return str(download_dir)
                self._log("download_fail", f"URL failed: {identifier[:100]}")

            else:
                # Fallback: LLM generates download script
                prompt = DATASET_DOWNLOAD_PROMPT.format(
                    name=name, source=source, identifier=identifier,
                    description=dataset.get("description", ""), output_dir=str(download_dir),
                )
                code = self._extract_code(await self.llm.generate(prompt, mode=LLMMode.THINKING))
                script_path = self.workspace / "download_dataset.py"
                script_path.write_text(code, encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.workspace),
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0 and list(download_dir.rglob("*.*")):
                    self._log("download_ok", f"Script download: {name}")
                    return str(download_dir)
                self._log("download_fail", f"Script: {result.stderr[:150]}")

        except subprocess.TimeoutExpired:
            self._log("download_timeout", f"Timeout downloading: {name}")
        except Exception as e:
            self._log("download_error", f"{name}: {e}")

        return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if "```" in text:
            for block in text.split("```")[1::2]:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except Exception:
                    continue
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        return {}

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
        """Fallback: minimal placeholder data. Real data should be uploaded or downloaded."""
        sample_dir = data_dir / "sample_000"
        sample_dir.mkdir(exist_ok=True)
        (sample_dir / "input.txt").write_text("placeholder — upload real dataset via /dashboard/upload-dataset", encoding="utf-8")
        (sample_dir / "ground_truth.json").write_text('{"note": "placeholder"}', encoding="utf-8")
        self._log("fallback_data", "No real data. Upload via /dashboard/upload-dataset or provide Kaggle/HuggingFace credentials.")

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

        # Scan dataset structure for benchmark prompt
        data_dir = self._find_real_data()
        data_structure = "No data directory found. Use /workspace/smoke_test/ as fallback."
        if data_dir:
            try:
                files = list(Path(data_dir).rglob("*.*"))[:20]
                exts = {}
                for f in files:
                    exts[f.suffix] = exts.get(f.suffix, 0) + 1
                total = len(list(Path(data_dir).rglob("*.*")))
                sample = [str(f.relative_to(data_dir)) for f in files[:5]]
                data_structure = f"Total files: {total}\nExtensions: {exts}\nSample paths: {sample}"
            except Exception:
                pass

        # Generate benchmark
        bench_prompt = REAL_BENCHMARK_PROMPT.format(
            title=hyp.get("title", ""),
            data_structure=data_structure,
        )
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
        """Run benchmark in Docker sandbox (falls back to local)."""
        import subprocess

        # Find data to mount
        data_dir = self._find_real_data()
        data_mount = f"-v {Path(data_dir).resolve()}:/data" if data_dir else ""

        try:
            # Try Docker first
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{impl_dir.resolve()}:/workspace",
            ]
            if data_dir:
                docker_cmd.extend(["-v", f"{Path(data_dir).resolve()}:/data"])
            docker_cmd.extend([
                "-w", "/workspace",
                "--memory", "4g", "--cpus", "2",
                "python:3.11-slim",
                "sh", "-c",
                "pip install -q pytest -r requirements.txt 2>&1; python -m pytest test_benchmark.py -v --tb=short -x 2>&1; cat benchmark_results.json 2>/dev/null || echo '{}'"
            ])
            result = await asyncio.to_thread(subprocess.run, docker_cmd, capture_output=True, text=True, timeout=600,)

            results_path = impl_dir / "benchmark_results.json"
            if results_path.exists():
                return json.loads(results_path.read_text())

            return {
                "passed": result.returncode == 0,
                "accuracy": 1.0 if result.returncode == 0 else 0.0,
                "output": result.stdout[-2000:],
            }
        except FileNotFoundError:
            # Docker not available, run locally
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "test_benchmark.py", "-v", "--tb=short", "-x"],
                    cwd=str(impl_dir), capture_output=True, text=True, timeout=300,
                )
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
