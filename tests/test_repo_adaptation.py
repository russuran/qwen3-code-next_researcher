from __future__ import annotations

from repo_adaptation.repo_ingest import RepoIngest
from repo_adaptation.codebase_graph import CodebaseGraphBuilder
from repo_adaptation.git_versioning import GitVersioning
from repo_adaptation.test_oracle import TestOracle


# --- Repo Ingest ---

def test_ingest_current_project():
    ingest = RepoIngest()
    manifest = ingest.ingest(".")
    assert manifest.total_files > 0
    assert "python" in manifest.languages
    assert any("requirements.txt" in cf for cf in manifest.config_files)


# --- Codebase Graph ---

def test_codebase_graph_builder():
    builder = CodebaseGraphBuilder()
    graph = builder.build("core")
    assert graph.files_analyzed > 0
    assert len(graph.entities) > 0

    # Should find key classes
    entity_names = [e.name for e in graph.entities]
    assert any("LLM" in n for n in entity_names)
    assert any("ToolRegistry" in n for n in entity_names)


def test_codebase_graph_edges():
    builder = CodebaseGraphBuilder()
    graph = builder.build("core")
    assert len(graph.edges) > 0
    edge_types = {e["type"] for e in graph.edges}
    assert "calls" in edge_types or "imports" in edge_types


# --- Git Versioning ---

def test_git_current_branch():
    git = GitVersioning(".")
    branch = git.current_branch()
    assert isinstance(branch, str)


def test_git_list_branches():
    git = GitVersioning(".")
    branches = git.list_branches()
    assert isinstance(branches, list)


def test_git_log():
    git = GitVersioning(".")
    log = git.get_log(5)
    assert isinstance(log, list)


# --- Test Oracle ---

def test_oracle_detect_command():
    oracle = TestOracle(".")
    cmd = oracle.detect_test_command()
    assert "pytest" in " ".join(cmd)
