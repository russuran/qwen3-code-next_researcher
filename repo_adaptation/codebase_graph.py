"""Codebase graph: AST-based analysis of Python code structure."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CodeEntity(BaseModel):
    name: str
    entity_type: str  # module, class, function, method
    file_path: str
    line_start: int = 0
    line_end: int = 0
    docstring: str = ""
    imports: list[str] = []
    calls: list[str] = []
    bases: list[str] = []  # for classes: parent classes


class CodebaseGraph(BaseModel):
    entities: list[CodeEntity] = []
    edges: list[dict[str, str]] = []  # {from, to, type}
    files_analyzed: int = 0
    errors: list[str] = []

    def get_entity(self, name: str) -> CodeEntity | None:
        for e in self.entities:
            if e.name == name:
                return e
        return None

    def get_callers(self, name: str) -> list[CodeEntity]:
        return [e for e in self.entities if name in e.calls]

    def get_imports_of(self, module: str) -> list[CodeEntity]:
        return [e for e in self.entities if any(module in imp for imp in e.imports)]


class CodebaseGraphBuilder:
    """Builds a codebase graph from Python source files using AST."""

    def build(self, repo_path: str | Path) -> CodebaseGraph:
        repo_path = Path(repo_path)
        graph = CodebaseGraph()

        for py_file in repo_path.rglob("*.py"):
            rel = str(py_file.relative_to(repo_path))
            if any(part.startswith(".") or part in ("node_modules", "__pycache__", "venv", ".venv")
                   for part in py_file.parts):
                continue

            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source, filename=rel)
                self._extract_entities(tree, rel, graph)
                graph.files_analyzed += 1
            except SyntaxError as e:
                graph.errors.append(f"{rel}: {e}")
            except Exception as e:
                graph.errors.append(f"{rel}: {e}")

        # Build edges from calls and imports
        entity_names = {e.name for e in graph.entities}
        for entity in graph.entities:
            for call in entity.calls:
                if call in entity_names:
                    graph.edges.append({"from": entity.name, "to": call, "type": "calls"})
            for imp in entity.imports:
                mod = imp.split(".")[-1]
                if mod in entity_names:
                    graph.edges.append({"from": entity.name, "to": mod, "type": "imports"})
            for base in entity.bases:
                if base in entity_names:
                    graph.edges.append({"from": entity.name, "to": base, "type": "inherits"})

        logger.info("Graph built: %d entities, %d edges from %d files",
                     len(graph.entities), len(graph.edges), graph.files_analyzed)
        return graph

    def _extract_entities(self, tree: ast.AST, file_path: str, graph: CodebaseGraph) -> None:
        module_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_imports.append(node.module)

        # Module-level entity
        module_name = file_path.replace("/", ".").replace(".py", "")
        graph.entities.append(CodeEntity(
            name=module_name,
            entity_type="module",
            file_path=file_path,
            imports=module_imports,
        ))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                calls = self._extract_calls(node)
                bases = [self._name_from_node(b) for b in node.bases if self._name_from_node(b)]
                graph.entities.append(CodeEntity(
                    name=node.name,
                    entity_type="class",
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node) or "",
                    calls=calls,
                    bases=bases,
                    imports=module_imports,
                ))

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_calls = self._extract_calls(item)
                        graph.entities.append(CodeEntity(
                            name=f"{node.name}.{item.name}",
                            entity_type="method",
                            file_path=file_path,
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            docstring=ast.get_docstring(item) or "",
                            calls=method_calls,
                        ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = self._extract_calls(node)
                graph.entities.append(CodeEntity(
                    name=node.name,
                    entity_type="function",
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node) or "",
                    calls=calls,
                    imports=module_imports,
                ))

    def _extract_calls(self, node: ast.AST) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self._name_from_node(child.func)
                if name:
                    calls.append(name)
        return list(set(calls))

    @staticmethod
    def _name_from_node(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""
