"""Benchmark manager: runs evaluation benchmarks and tracks results."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BenchmarkCase(BaseModel):
    case_id: str
    query: str
    expected_sources: list[str] = []
    expected_keywords: list[str] = []
    metadata: dict[str, Any] = {}


class BenchmarkResult(BaseModel):
    case_id: str
    metrics: dict[str, float] = {}
    passed: bool = False
    details: str = ""


class BenchmarkSuite(BaseModel):
    name: str
    cases: list[BenchmarkCase] = []
    results: list[BenchmarkResult] = []

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)


class BenchmarkManager:
    def __init__(self, benchmarks_dir: str = "benchmarks") -> None:
        self.benchmarks_dir = Path(benchmarks_dir)

    def load_suite(self, name: str, split: str = "validation") -> BenchmarkSuite:
        """Load benchmark cases from JSONL file."""
        path = self.benchmarks_dir / name / f"{split}.jsonl"
        suite = BenchmarkSuite(name=f"{name}/{split}")

        if not path.exists():
            logger.warning("Benchmark file not found: %s", path)
            return suite

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    suite.cases.append(BenchmarkCase(**data))

        logger.info("Loaded benchmark %s: %d cases", suite.name, len(suite.cases))
        return suite

    def save_results(self, suite: BenchmarkSuite, output_path: str | None = None) -> Path:
        if output_path is None:
            out = self.benchmarks_dir / "results" / f"{suite.name.replace('/', '_')}.json"
        else:
            out = Path(output_path)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved results: %s (pass rate: %.1f%%)", out, suite.pass_rate * 100)
        return out

    def compare(self, baseline: BenchmarkSuite, candidate: BenchmarkSuite) -> dict[str, Any]:
        """Compare two benchmark runs."""
        return {
            "baseline_pass_rate": baseline.pass_rate,
            "candidate_pass_rate": candidate.pass_rate,
            "delta": candidate.pass_rate - baseline.pass_rate,
            "improved": candidate.pass_rate > baseline.pass_rate,
            "baseline_cases": len(baseline.results),
            "candidate_cases": len(candidate.results),
        }
