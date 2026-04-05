"""Test oracle: runs existing test suite to validate patches."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TestResult(BaseModel):
    passed: bool
    total: int = 0
    failures: int = 0
    errors: int = 0
    output: str = ""
    return_code: int = 0


class TestOracle:
    """Runs the existing test suite of a repository."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)

    def detect_test_command(self) -> list[str]:
        """Auto-detect the test runner command."""
        # Check for common test configurations
        if (self.repo_path / "pyproject.toml").exists():
            content = (self.repo_path / "pyproject.toml").read_text(errors="ignore")
            if "pytest" in content:
                return ["python", "-m", "pytest", "-v", "--tb=short"]

        if (self.repo_path / "pytest.ini").exists() or (self.repo_path / "setup.cfg").exists():
            return ["python", "-m", "pytest", "-v", "--tb=short"]

        if (self.repo_path / "Makefile").exists():
            content = (self.repo_path / "Makefile").read_text(errors="ignore")
            if "test:" in content:
                return ["make", "test"]

        if (self.repo_path / "package.json").exists():
            return ["npm", "test"]

        # Default: pytest
        return ["python", "-m", "pytest", "-v", "--tb=short"]

    def run_tests(self, command: list[str] | None = None, timeout: int = 300) -> TestResult:
        cmd = command or self.detect_test_command()
        logger.info("Running tests: %s in %s", " ".join(cmd), self.repo_path)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout + "\n" + result.stderr
            passed = result.returncode == 0

            # Parse pytest output for counts
            total, failures, errors = 0, 0, 0
            for line in output.splitlines():
                if "passed" in line or "failed" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "passed" and i > 0:
                            try:
                                total += int(parts[i - 1])
                            except ValueError:
                                pass
                        elif p == "failed" and i > 0:
                            try:
                                failures += int(parts[i - 1])
                            except ValueError:
                                pass
                        elif p == "error" and i > 0:
                            try:
                                errors += int(parts[i - 1])
                            except ValueError:
                                pass

            total = total + failures + errors

            return TestResult(
                passed=passed,
                total=total,
                failures=failures,
                errors=errors,
                output=output[-5000:],  # truncate
                return_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return TestResult(passed=False, output="Test timeout exceeded", return_code=-1)
        except Exception as e:
            return TestResult(passed=False, output=str(e), return_code=-1)
