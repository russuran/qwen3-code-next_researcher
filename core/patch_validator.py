"""Patch validator: validates patches via tests and sandbox execution."""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from repo_adaptation.patch_editor import PatchCandidate
from sandbox.sandbox_runner import SandboxJob, SandboxResult, SandboxRunner

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    patch_id: str
    valid: bool = False
    tests_passed: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0


def validate_patch(
    patch: PatchCandidate,
    repo_path: str | Path,
    sandbox_runner: SandboxRunner | None = None,
) -> ValidationResult:
    """Validate a patch by applying it and running tests in a sandbox."""
    repo = Path(repo_path)
    runner = sandbox_runner or SandboxRunner()

    # Write patch to temporary location
    target = repo / patch.file_path
    backup = target.read_text(encoding="utf-8") if target.exists() else ""

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch.modified, encoding="utf-8")

        job = SandboxJob(
            job_type="test",
            image="python:3.11-slim",
            command=["python", "-m", "pytest", "--tb=short", "-q"],
            mount_dir=str(repo),
            timeout_sec=300,
        )

        result: SandboxResult = runner.run(job)

        return ValidationResult(
            patch_id=patch.patch_id,
            valid=result.success,
            tests_passed=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_sec=result.duration_sec,
        )
    except Exception as e:
        logger.error("Validation failed for patch %s: %s", patch.patch_id, e)
        return ValidationResult(patch_id=patch.patch_id, stderr=str(e))
    finally:
        # Restore original content
        if backup:
            target.write_text(backup, encoding="utf-8")
        elif target.exists():
            target.unlink()
