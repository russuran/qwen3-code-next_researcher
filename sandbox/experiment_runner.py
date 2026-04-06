"""Experiment runner: runs hypothesis checks and benchmarks in sandbox."""
from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel

from sandbox.sandbox_runner import SandboxJob, SandboxRunner

logger = logging.getLogger(__name__)


class ExperimentResult(BaseModel):
    experiment_id: str = ""
    hypothesis: str = ""
    success: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
    metrics: dict[str, float] = {}


class ExperimentRunner:
    """Runs experiments (hypothesis checks, benchmarks) inside a sandbox."""

    def __init__(self, runner: SandboxRunner | None = None) -> None:
        self._runner = runner or SandboxRunner()

    def run_experiment(
        self,
        hypothesis: str,
        code_path: str,
        data_path: str | None = None,
        image: str = "python:3.11-slim",
        timeout_sec: int = 600,
    ) -> ExperimentResult:
        """Execute an experiment and return structured results."""
        experiment_id = str(uuid.uuid4())[:8]

        command = ["python", code_path]
        if data_path:
            command.extend(["--data", data_path])

        job = SandboxJob(
            job_id=experiment_id,
            job_type="hypothesis_check",
            image=image,
            command=command,
            mount_dir=code_path.rsplit("/", 1)[0] if "/" in code_path else ".",
            timeout_sec=timeout_sec,
        )

        logger.info("Running experiment %s: %s", experiment_id, hypothesis[:80])
        result = self._runner.run(job)
        metrics = self._parse_metrics(result.stdout)
        exp_result = ExperimentResult(
            experiment_id=experiment_id, hypothesis=hypothesis,
            success=result.success, exit_code=result.exit_code,
            stdout=result.stdout, stderr=result.stderr,
            duration_sec=result.duration_sec, metrics=metrics,
        )
        logger.info("Experiment %s: success=%s metrics=%d", experiment_id, result.success, len(metrics))
        return exp_result

    @staticmethod
    def _parse_metrics(stdout: str) -> dict[str, float]:
        """Parse key=value metrics from stdout lines prefixed with METRIC:."""
        metrics: dict[str, float] = {}
        for line in stdout.splitlines():
            if line.startswith("METRIC:"):
                parts = line[7:].strip().split("=", 1)
                if len(parts) == 2:
                    try:
                        metrics[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
        return metrics
