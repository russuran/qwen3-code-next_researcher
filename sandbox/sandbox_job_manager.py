"""Sandbox job manager: manages sandbox job lifecycle."""
from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel

from sandbox.sandbox_runner import SandboxJob, SandboxResult, SandboxRunner

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManagedJob(BaseModel):
    job_id: str
    job: SandboxJob
    status: JobStatus = JobStatus.QUEUED
    result: SandboxResult | None = None


class SandboxJobManager:
    """Manages the lifecycle of sandbox jobs."""

    def __init__(self, runner: SandboxRunner | None = None) -> None:
        self._runner = runner or SandboxRunner()
        self._jobs: dict[str, ManagedJob] = {}

    def submit(self, job: SandboxJob) -> str:
        """Submit a job for execution. Returns job_id."""
        if not job.job_id:
            job.job_id = str(uuid.uuid4())[:8]
        managed = ManagedJob(job_id=job.job_id, job=job)
        self._jobs[job.job_id] = managed
        logger.info("Submitted job %s (type=%s)", job.job_id, job.job_type)

        # Execute synchronously (async wrapper can be added)
        managed.status = JobStatus.RUNNING
        try:
            result = self._runner.run(job)
            managed.result = result
            managed.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
        except Exception as e:
            managed.status = JobStatus.FAILED
            managed.result = SandboxResult(
                job_id=job.job_id, success=False, stderr=str(e),
            )
            logger.error("Job %s failed: %s", job.job_id, e)

        return job.job_id

    def get_status(self, job_id: str) -> JobStatus | None:
        managed = self._jobs.get(job_id)
        return managed.status if managed else None

    def get_result(self, job_id: str) -> SandboxResult | None:
        managed = self._jobs.get(job_id)
        return managed.result if managed else None

    def cancel(self, job_id: str) -> bool:
        managed = self._jobs.get(job_id)
        if managed and managed.status == JobStatus.QUEUED:
            managed.status = JobStatus.CANCELLED
            logger.info("Cancelled job %s", job_id)
            return True
        return False

    def list_jobs(self) -> list[ManagedJob]:
        return list(self._jobs.values())
