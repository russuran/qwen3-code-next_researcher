from __future__ import annotations

import pytest

from sandbox.evaluator import Evaluator
from sandbox.benchmark_manager import BenchmarkManager, BenchmarkSuite, BenchmarkResult
from sandbox.sandbox_runner import SandboxJob


# --- Evaluator ---

def test_evaluate_report():
    evaluator = Evaluator()
    report = """
# Test Report

## Methods
Method A uses https://github.com/example/repo for implementation.

```python
import torch
model = torch.nn.Linear(10, 5)
```

## Comparison
| Method | Score |
|--------|-------|
| A | 95% |
| B | 88% |

References: https://arxiv.org/abs/1234
"""
    sources = [
        {"source": "arxiv", "title": "Paper A"},
        {"source": "github", "title": "Repo B"},
        {"source": "semantic_scholar", "title": "Paper C"},
    ]
    questions = ["What methods exist?", "How do they compare?"]

    metrics = evaluator.evaluate_report(report, sources, questions)
    assert 0 <= metrics.overall <= 1
    assert metrics.source_diversity > 0
    assert metrics.code_presence > 0


def test_evaluate_patch():
    evaluator = Evaluator()
    diff = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
-    print("old")
+    print("new")
+    return True
"""
    metrics = evaluator.evaluate_patch(diff, test_passed=True)
    assert metrics.tests_pass is True
    assert metrics.files_changed == 1
    assert metrics.lines_added == 2
    assert metrics.lines_removed == 1


# --- Benchmark Manager ---

def test_benchmark_suite():
    suite = BenchmarkSuite(name="test")
    suite.results = [
        BenchmarkResult(case_id="1", passed=True),
        BenchmarkResult(case_id="2", passed=False),
        BenchmarkResult(case_id="3", passed=True),
    ]
    assert suite.pass_rate == pytest.approx(2 / 3)


def test_benchmark_compare():
    mgr = BenchmarkManager()
    baseline = BenchmarkSuite(name="baseline", results=[
        BenchmarkResult(case_id="1", passed=True),
        BenchmarkResult(case_id="2", passed=False),
    ])
    candidate = BenchmarkSuite(name="candidate", results=[
        BenchmarkResult(case_id="1", passed=True),
        BenchmarkResult(case_id="2", passed=True),
    ])
    result = mgr.compare(baseline, candidate)
    assert result["improved"] is True
    assert result["delta"] > 0


# --- Sandbox Job model ---

def test_sandbox_job_defaults():
    job = SandboxJob(run_id="test-run")
    assert job.image == "python:3.11-slim"
    assert job.timeout_sec == 600
    assert job.memory_mb == 4096
