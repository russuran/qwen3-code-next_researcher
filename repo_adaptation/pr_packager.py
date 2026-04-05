"""PR packager: creates pull-request-ready artifacts."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from repo_adaptation.git_versioning import GitVersioning

logger = logging.getLogger(__name__)


class PRPackage(BaseModel):
    title: str
    description: str
    base_branch: str
    candidate_branch: str
    diff: str = ""
    files_changed: list[str] = []
    test_results: dict = {}
    validation_status: str = "pending"  # pending | passed | failed


class PRPackager:
    def __init__(self, git: GitVersioning) -> None:
        self.git = git

    def package(
        self,
        title: str,
        description: str,
        base_branch: str = "main",
        candidate_branch: str | None = None,
        test_results: dict | None = None,
    ) -> PRPackage:
        if candidate_branch is None:
            candidate_branch = self.git.current_branch()

        diff = self.git.diff(base=base_branch, head=candidate_branch)

        files_changed = []
        for line in diff.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    files_changed.append(parts[3].lstrip("b/"))

        validation = "pending"
        if test_results:
            validation = "passed" if test_results.get("passed", False) else "failed"

        pkg = PRPackage(
            title=title,
            description=description,
            base_branch=base_branch,
            candidate_branch=candidate_branch,
            diff=diff,
            files_changed=files_changed,
            test_results=test_results or {},
            validation_status=validation,
        )

        logger.info("PR package: %s (%d files, validation=%s)",
                     title, len(files_changed), validation)
        return pkg

    def save(self, pkg: PRPackage, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "pr_metadata.json").write_text(
            pkg.model_dump_json(indent=2, exclude={"diff"}), encoding="utf-8"
        )
        (out / "changes.diff").write_text(pkg.diff, encoding="utf-8")

        # PR description markdown
        md = f"# {pkg.title}\n\n{pkg.description}\n\n"
        md += f"## Changes\n\n"
        for f in pkg.files_changed:
            md += f"- `{f}`\n"
        md += f"\n## Validation: {pkg.validation_status}\n"
        if pkg.test_results:
            md += f"\n```json\n{json.dumps(pkg.test_results, indent=2)}\n```\n"

        (out / "pr_description.md").write_text(md, encoding="utf-8")

        logger.info("Saved PR package to %s", out)
        return out
