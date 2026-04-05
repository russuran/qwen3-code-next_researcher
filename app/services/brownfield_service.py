"""Brownfield service: repo adaptation pipeline via API."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.db.models import Run, Event
from app.db import session as db_session
from app.deps import build_llm
from app.settings import Settings
from repo_adaptation.repo_ingest import RepoIngest
from repo_adaptation.codebase_graph import CodebaseGraphBuilder
from repo_adaptation.architecture_mapper import ArchitectureMapper
from repo_adaptation.impact_analyzer import ImpactAnalyzer
from repo_adaptation.patch_editor import PatchEditor
from repo_adaptation.git_versioning import GitVersioning
from repo_adaptation.test_oracle import TestOracle
from repo_adaptation.pr_packager import PRPackager

logger = logging.getLogger(__name__)


async def execute_brownfield(
    run_id: UUID,
    repo_path: str,
    change_request: str,
    target_files: list[str],
    settings: Settings,
) -> None:
    """Full brownfield pipeline: ingest → graph → analyze → patch → test → PR."""
    if db_session.async_session_factory is None:
        logger.error("DB not initialized")
        return

    async with db_session.async_session_factory() as db:
        run = await db.get(Run, run_id)
        if not run:
            return

        try:
            run.status = "planning"
            run.started_at = datetime.now(timezone.utc)
            await db.commit()

            llm = build_llm(settings)

            # 1. Ingest repo
            await _log_event(db, run_id, "plan", "repo_ingest", f"Ingesting {repo_path}")
            ingest = RepoIngest()
            manifest = ingest.ingest(repo_path)
            await _log_event(db, run_id, "plan", "repo_manifest",
                           f"{manifest.total_files} files, {manifest.total_lines} lines")

            # 2. Build codebase graph
            await _log_event(db, run_id, "plan", "build_graph", "Building AST graph")
            builder = CodebaseGraphBuilder()
            graph = builder.build(repo_path)
            await _log_event(db, run_id, "plan", "graph_built",
                           f"{len(graph.entities)} entities, {len(graph.edges)} edges")

            # 3. Architecture mapping
            await _log_event(db, run_id, "plan", "architecture_map", "Mapping architecture")
            mapper = ArchitectureMapper()
            arch_map = mapper.map(graph)
            await _log_event(db, run_id, "plan", "architecture_mapped",
                           f"{len(arch_map.layers)} layers, patterns: {arch_map.patterns_detected}")

            # 4. Impact analysis
            for target in target_files:
                module_name = target.replace("/", ".").replace(".py", "")
                analyzer = ImpactAnalyzer(graph)
                impact = analyzer.analyze(module_name)
                await _log_event(db, run_id, "analyze", "impact_analysis", impact.summary)

            # 5. Create branch
            git = GitVersioning(repo_path)
            git.init_if_missing()
            branch_name = f"ai/task/{run_id}"
            try:
                git.create_branch(branch_name)
            except Exception:
                git.checkout(branch_name)
            await _log_event(db, run_id, "analyze", "branch_created", branch_name)

            # 6. Generate patches
            editor = PatchEditor(llm)
            for target_file in target_files:
                file_path = Path(repo_path) / target_file
                if not file_path.exists():
                    await _log_event(db, run_id, "analyze", "skip_file", f"Not found: {target_file}")
                    continue

                content = file_path.read_text(encoding="utf-8", errors="ignore")
                await _log_event(db, run_id, "analyze", "generate_patch", f"Patching: {target_file}")

                patch = await editor.generate_patch(target_file, content, change_request)
                editor.apply_patch(repo_path, patch)
                await _log_event(db, run_id, "analyze", "patch_applied", f"Applied to {target_file}")

            # 7. Commit changes
            sha = git.commit(f"ai: {change_request[:50]}", run_id=str(run_id))
            await _log_event(db, run_id, "analyze", "committed", f"SHA: {sha[:8]}")

            # 8. Run tests
            await _log_event(db, run_id, "synthesize", "run_tests", "Running test suite")
            oracle = TestOracle(repo_path)
            test_result = oracle.run_tests()
            await _log_event(db, run_id, "synthesize", "tests_complete",
                           f"Passed: {test_result.passed}, Total: {test_result.total}")

            # 9. Package PR
            packager = PRPackager(git)
            pr_pkg = packager.package(
                title=f"ai: {change_request[:60]}",
                description=change_request,
                base_branch="main",
                candidate_branch=branch_name,
                test_results=test_result.model_dump(),
            )
            output_dir = Path(f"output/brownfield-{str(run_id)[:8]}")
            packager.save(pr_pkg, output_dir)
            await _log_event(db, run_id, "synthesize", "pr_packaged",
                           f"{len(pr_pkg.files_changed)} files, validation={pr_pkg.validation_status}")

            run.status = "completed"
            run.output_dir = str(output_dir)
            run.finished_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("Brownfield run %s failed: %s", run_id, e, exc_info=True)
            run.status = "failed"
            run.error = str(e)
            run.finished_at = datetime.now(timezone.utc)

        run.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def _log_event(db, run_id: UUID, phase: str, action: str, summary: str) -> None:
    event = Event(run_id=run_id, phase=phase, action=action, result_summary=summary)
    db.add(event)
    await db.commit()
