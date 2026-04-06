"""Docker runner: low-level Docker container execution wrapper."""
from __future__ import annotations

import logging
import subprocess
import time

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DockerResult(BaseModel):
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
    timed_out: bool = False


class DockerRunner:
    """Wraps subprocess docker run with resource limits."""

    def __init__(self, docker_bin: str = "docker") -> None:
        self._bin = docker_bin

    def run(
        self,
        image: str,
        command: list[str],
        volumes: dict[str, str] | None = None,
        limits: dict[str, str | float] | None = None,
        timeout_sec: int = 600,
        network: bool = False,
        env: dict[str, str] | None = None,
    ) -> DockerResult:
        """Run a command in a Docker container."""
        cmd = [self._bin, "run", "--rm"]

        # Resource limits
        limits = limits or {}
        if "memory" in limits:
            cmd.extend(["--memory", str(limits["memory"])])
        if "cpus" in limits:
            cmd.extend(["--cpus", str(limits["cpus"])])

        if not network:
            cmd.extend(["--network", "none"])

        # Volumes
        for host_path, container_path in (volumes or {}).items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Env vars
        for key, val in (env or {}).items():
            cmd.extend(["-e", f"{key}={val}"])

        cmd.append(image)
        cmd.extend(command)

        logger.info("Docker run: %s", " ".join(cmd))
        start = time.time()

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            return DockerResult(
                exit_code=proc.returncode, stdout=proc.stdout[-10_000:],
                stderr=proc.stderr[-5_000:], duration_sec=round(time.time() - start, 2),
            )
        except subprocess.TimeoutExpired:
            return DockerResult(stderr=f"Timeout after {timeout_sec}s", duration_sec=timeout_sec, timed_out=True)
        except FileNotFoundError:
            return DockerResult(stderr="Docker binary not found")
        except Exception as e:
            return DockerResult(stderr=str(e))
