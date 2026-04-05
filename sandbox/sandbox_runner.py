"""Sandbox runner: Docker-based isolated execution environment."""
from __future__ import annotations

import logging
import subprocess
import uuid

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SandboxJob(BaseModel):
    job_id: str = ""
    run_id: str = ""
    job_type: str = "test"  # test, benchmark, build, hypothesis_check
    image: str = "python:3.11-slim"
    command: list[str] = ["python", "-m", "pytest"]
    timeout_sec: int = 600
    memory_mb: int = 4096
    cpu_limit: float = 2.0
    network_enabled: bool = False
    mount_dir: str = ""
    env_vars: dict[str, str] = {}


class SandboxResult(BaseModel):
    job_id: str
    success: bool
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0


class SandboxRunner:
    """Runs jobs in Docker containers with resource limits."""

    def run(self, job: SandboxJob) -> SandboxResult:
        if not job.job_id:
            job.job_id = str(uuid.uuid4())[:8]

        cmd = ["docker", "run", "--rm"]

        # Resource limits
        cmd.extend(["--memory", f"{job.memory_mb}m"])
        cmd.extend(["--cpus", str(job.cpu_limit)])

        # Network
        if not job.network_enabled:
            cmd.extend(["--network", "none"])

        # Mount
        if job.mount_dir:
            cmd.extend(["-v", f"{job.mount_dir}:/workspace", "-w", "/workspace"])

        # Env vars
        for key, val in job.env_vars.items():
            cmd.extend(["-e", f"{key}={val}"])

        # Image + command
        cmd.append(job.image)
        cmd.extend(job.command)

        logger.info("Sandbox job %s: %s", job.job_id, " ".join(cmd))

        import time
        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=job.timeout_sec,
            )
            duration = time.time() - start

            return SandboxResult(
                job_id=job.job_id,
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout[-10000:],
                stderr=result.stderr[-5000:],
                duration_sec=round(duration, 2),
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                job_id=job.job_id,
                success=False,
                exit_code=-1,
                stderr=f"Timeout after {job.timeout_sec}s",
                duration_sec=job.timeout_sec,
            )
        except FileNotFoundError:
            return SandboxResult(
                job_id=job.job_id,
                success=False,
                exit_code=-1,
                stderr="Docker not found. Install Docker to use sandbox.",
            )
        except Exception as e:
            return SandboxResult(
                job_id=job.job_id, success=False, exit_code=-1, stderr=str(e),
            )
