from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from click.testing import CliRunner

from cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n"
        "  provider: ollama\n"
        "  model: test-model\n"
        "  host: http://localhost:11434\n"
        "search:\n"
        "  sources: [arxiv]\n"
        "  max_results_per_source: 5\n"
        "output:\n"
        "  base_dir: {out}\n"
        "journal:\n"
        "  base_dir: {journal}\n".format(
            out=str(tmp_path / "output"),
            journal=str(tmp_path / "journal"),
        )
    )
    return cfg


def test_cli_help(runner: CliRunner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "TOPIC" in result.output
    assert "--verbose" in result.output
    assert "--intervene" in result.output
    assert "--sources" in result.output


def test_cli_runs_agent(runner: CliRunner, config_file: Path, tmp_path: Path):
    output_dir = tmp_path / "output" / "test-topic"
    output_dir.mkdir(parents=True)

    with patch("cli.ResearchAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=output_dir)
        MockAgent.return_value = mock_instance

        result = runner.invoke(main, [
            "test topic",
            "--config", str(config_file),
            "--verbose",
        ])

    assert result.exit_code == 0
    assert "Done!" in result.output
    MockAgent.assert_called_once()
    mock_instance.run.assert_called_once_with("test topic")


def test_cli_missing_config(runner: CliRunner):
    with patch("cli.ResearchAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=Path("/tmp/out"))
        MockAgent.return_value = mock_instance

        result = runner.invoke(main, [
            "test topic",
            "--config", "/nonexistent/config.yaml",
        ])

    # Should still run with defaults
    assert result.exit_code == 0


def test_cli_sources_override(runner: CliRunner, config_file: Path, tmp_path: Path):
    output_dir = tmp_path / "output" / "test-topic"
    output_dir.mkdir(parents=True)

    with patch("cli.ResearchAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=output_dir)
        MockAgent.return_value = mock_instance

        result = runner.invoke(main, [
            "test topic",
            "--config", str(config_file),
            "--sources", "arxiv,github",
        ])

    assert result.exit_code == 0
    # Verify the agent was created with overridden sources
    call_kwargs = MockAgent.call_args
    agent_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert "arxiv" in agent_config.sources
    assert "github" in agent_config.sources
