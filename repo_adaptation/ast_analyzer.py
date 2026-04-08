"""AST analysis via libcst: structural understanding of Python code.

Provides:
- Function/class/method extraction with signatures and docstrings
- Import graph (who imports what)
- Call graph (who calls what)
- Safe code transformations that preserve formatting and comments
- Scope analysis for targeted patching
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
from libcst import matchers as m

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    name: str
    file_path: str
    line_start: int
    line_end: int
    params: list[str]
    return_annotation: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: str = ""
    calls: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    file_path: str
    line_start: int
    line_end: int
    bases: list[str]
    docstring: str = ""
    methods: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    module: str
    names: list[str]  # imported names, or ["*"]
    is_from: bool = False
    file_path: str = ""


@dataclass
class ASTAnalysis:
    """Full analysis of a Python file or repo."""
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    files_analyzed: int = 0
    errors: list[str] = field(default_factory=list)

    def get_function(self, name: str) -> FunctionInfo | None:
        for f in self.functions:
            if f.name == name or f"{f.class_name}.{f.name}" == name:
                return f
        return None

    def get_class(self, name: str) -> ClassInfo | None:
        for c in self.classes:
            if c.name == name:
                return c
        return None

    def get_callers_of(self, func_name: str) -> list[FunctionInfo]:
        return [f for f in self.functions if func_name in f.calls]

    def get_imports_in(self, file_path: str) -> list[ImportInfo]:
        return [i for i in self.imports if i.file_path == file_path]

    def summary(self) -> dict:
        return {
            "files": self.files_analyzed,
            "functions": len(self.functions),
            "classes": len(self.classes),
            "imports": len(self.imports),
            "errors": len(self.errors),
        }


class _Visitor(cst.CSTVisitor):
    """Extracts functions, classes, imports, and calls from a CST."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self._current_class: str = ""
        self._current_calls: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        name = node.name.value
        bases = []
        for arg in node.bases:
            if isinstance(arg.value, cst.Name):
                bases.append(arg.value.value)
            elif isinstance(arg.value, cst.Attribute):
                bases.append(self._attr_name(arg.value))

        docstring = self._get_docstring(node.body)
        decorators = [self._decorator_name(d) for d in node.decorators]
        pos = self._pos(node)

        self.classes.append(ClassInfo(
            name=name,
            file_path=self.file_path,
            line_start=pos[0],
            line_end=pos[1],
            bases=bases,
            docstring=docstring,
            decorators=decorators,
        ))
        self._current_class = name
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        # Collect method names for the class
        class_name = node.name.value
        for cls in self.classes:
            if cls.name == class_name:
                cls.methods = [
                    f.name for f in self.functions
                    if f.class_name == class_name
                ]
        self._current_class = ""

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name = node.name.value
        params = self._extract_params(node.params)
        ret = ""
        if node.returns:
            ret = self._annotation_str(node.returns.annotation)

        docstring = self._get_docstring(node.body)
        decorators = [self._decorator_name(d) for d in node.decorators]
        is_async = isinstance(node.asynchronous, cst.Asynchronous) if node.asynchronous else False
        pos = self._pos(node)

        self._current_calls = []
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        name = node.name.value
        params = self._extract_params(node.params)
        ret = ""
        if node.returns:
            ret = self._annotation_str(node.returns.annotation)

        docstring = self._get_docstring(node.body)
        decorators = [self._decorator_name(d) for d in node.decorators]
        is_async = isinstance(node.asynchronous, cst.Asynchronous) if node.asynchronous else False
        pos = self._pos(node)

        self.functions.append(FunctionInfo(
            name=name,
            file_path=self.file_path,
            line_start=pos[0],
            line_end=pos[1],
            params=params,
            return_annotation=ret,
            docstring=docstring,
            decorators=decorators,
            is_async=is_async,
            is_method=bool(self._current_class),
            class_name=self._current_class,
            calls=list(set(self._current_calls)),
        ))
        self._current_calls = []

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Name):
            self._current_calls.append(node.func.value)
        elif isinstance(node.func, cst.Attribute):
            self._current_calls.append(node.func.attr.value)

    def visit_Import(self, node: cst.Import) -> None:
        if isinstance(node.names, cst.ImportStar):
            self.imports.append(ImportInfo(module="*", names=["*"], file_path=self.file_path))
        elif isinstance(node.names, (list, tuple)):
            for alias in node.names:
                if isinstance(alias, cst.ImportAlias):
                    name = self._import_alias_name(alias)
                    self.imports.append(ImportInfo(
                        module=name, names=[name], file_path=self.file_path,
                    ))

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = ""
        if node.module:
            module = self._attr_name(node.module) if isinstance(node.module, cst.Attribute) else node.module.value
        names = []
        if isinstance(node.names, cst.ImportStar):
            names = ["*"]
        elif isinstance(node.names, (list, tuple)):
            for alias in node.names:
                if isinstance(alias, cst.ImportAlias):
                    names.append(self._import_alias_name(alias))
        self.imports.append(ImportInfo(
            module=module, names=names, is_from=True, file_path=self.file_path,
        ))

    # ---- helpers ----

    def _extract_params(self, params: cst.Parameters) -> list[str]:
        result = []
        for p in params.params:
            name = p.name.value
            if p.annotation:
                name += f": {self._annotation_str(p.annotation.annotation)}"
            result.append(name)
        return result

    def _annotation_str(self, node: cst.BaseExpression) -> str:
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return self._attr_name(node)
        elif isinstance(node, cst.Subscript):
            return self._subscript_str(node)
        return ""

    def _subscript_str(self, node: cst.Subscript) -> str:
        base = self._annotation_str(node.value)
        slices = []
        for s in node.slice:
            if isinstance(s, cst.SubscriptElement) and isinstance(s.slice, cst.Index):
                slices.append(self._annotation_str(s.slice.value))
        return f"{base}[{', '.join(slices)}]" if slices else base

    @staticmethod
    def _attr_name(node) -> str:
        parts = []
        while isinstance(node, cst.Attribute):
            parts.append(node.attr.value)
            node = node.value
        if isinstance(node, cst.Name):
            parts.append(node.value)
        return ".".join(reversed(parts))

    @staticmethod
    def _decorator_name(dec: cst.Decorator) -> str:
        if isinstance(dec.decorator, cst.Name):
            return dec.decorator.value
        elif isinstance(dec.decorator, cst.Attribute):
            return _Visitor._attr_name(dec.decorator)
        elif isinstance(dec.decorator, cst.Call):
            if isinstance(dec.decorator.func, cst.Name):
                return dec.decorator.func.value
            elif isinstance(dec.decorator.func, cst.Attribute):
                return _Visitor._attr_name(dec.decorator.func)
        return ""

    @staticmethod
    def _import_alias_name(alias: cst.ImportAlias) -> str:
        if isinstance(alias.name, cst.Attribute):
            return _Visitor._attr_name(alias.name)
        elif isinstance(alias.name, cst.Name):
            return alias.name.value
        return ""

    @staticmethod
    def _get_docstring(body: cst.BaseSuite) -> str:
        if isinstance(body, cst.IndentedBlock):
            stmts = body.body
            if stmts and isinstance(stmts[0], cst.SimpleStatementLine):
                expr = stmts[0].body[0] if stmts[0].body else None
                if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString)):
                    raw = expr.value.evaluated_value if hasattr(expr.value, 'evaluated_value') else ""
                    return raw or ""
        return ""

    @staticmethod
    def _pos(node) -> tuple[int, int]:
        try:
            wrapper = cst.metadata.MetadataWrapper(cst.parse_module(""))
        except Exception:
            pass
        # libcst doesn't track line numbers directly on nodes, use a rough approach
        # via the code generator
        return (0, 0)


class ASTAnalyzer:
    """Analyze Python files/repos using libcst for structural understanding."""

    @staticmethod
    def _walk_tree(tree: cst.Module, visitor: _Visitor) -> None:
        """Walk a CST using MetadataWrapper for proper traversal."""
        try:
            wrapper = cst.metadata.MetadataWrapper(tree)
            wrapper.visit(visitor)
        except Exception:
            # Fallback: manual walk via transformer trick
            tree.visit(visitor)

    def analyze_file(self, file_path: str | Path) -> ASTAnalysis:
        file_path = Path(file_path)
        analysis = ASTAnalysis()

        try:
            source = file_path.read_text(errors="ignore")
            tree = cst.parse_module(source)
            visitor = _Visitor(str(file_path))
            self._walk_tree(tree, visitor)
            analysis.functions = visitor.functions
            analysis.classes = visitor.classes
            analysis.imports = visitor.imports
            analysis.files_analyzed = 1
        except Exception as e:
            analysis.errors.append(f"{file_path}: {e}")

        return analysis

    def analyze_repo(self, repo_path: str | Path) -> ASTAnalysis:
        repo_path = Path(repo_path)
        analysis = ASTAnalysis()

        _skip = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".tox"}
        for py_file in repo_path.rglob("*.py"):
            if any(part in _skip for part in py_file.parts):
                continue

            try:
                source = py_file.read_text(errors="ignore")
                tree = cst.parse_module(source)
                rel = str(py_file.relative_to(repo_path))
                visitor = _Visitor(rel)
                self._walk_tree(tree, visitor)
                analysis.functions.extend(visitor.functions)
                analysis.classes.extend(visitor.classes)
                analysis.imports.extend(visitor.imports)
                analysis.files_analyzed += 1
            except Exception as e:
                analysis.errors.append(f"{py_file.name}: {e}")

        logger.info(
            "AST analysis: %d files, %d functions, %d classes, %d imports, %d errors",
            analysis.files_analyzed, len(analysis.functions),
            len(analysis.classes), len(analysis.imports), len(analysis.errors),
        )
        return analysis


# ---------------------------------------------------------------------------
# Code transformer: targeted, structure-preserving edits
# ---------------------------------------------------------------------------

class AddArgumentTransformer(cst.CSTTransformer):
    """Add a keyword argument to a specific function/class instantiation."""

    def __init__(self, target_func: str, arg_name: str, arg_value: str) -> None:
        self.target_func = target_func
        self.arg_name = arg_name
        self.arg_value = arg_value
        self.changed = False

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.Call:
        func_name = ""
        if isinstance(updated.func, cst.Name):
            func_name = updated.func.value
        elif isinstance(updated.func, cst.Attribute):
            func_name = updated.func.attr.value

        if func_name != self.target_func:
            return updated

        # Check if arg already exists
        for arg in updated.args:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == self.arg_name:
                return updated  # already present

        new_arg = cst.Arg(
            keyword=cst.Name(self.arg_name),
            value=cst.parse_expression(self.arg_value),
            equal=cst.AssignEqual(
                whitespace_before=cst.SimpleWhitespace(""),
                whitespace_after=cst.SimpleWhitespace(""),
            ),
        )
        self.changed = True
        return updated.with_changes(args=[*updated.args, new_arg])


class ReplaceValueTransformer(cst.CSTTransformer):
    """Replace the value of a specific keyword argument in a function call."""

    def __init__(self, target_func: str, arg_name: str, new_value: str) -> None:
        self.target_func = target_func
        self.arg_name = arg_name
        self.new_value = new_value
        self.changed = False

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.Call:
        func_name = ""
        if isinstance(updated.func, cst.Name):
            func_name = updated.func.value
        elif isinstance(updated.func, cst.Attribute):
            func_name = updated.func.attr.value

        if func_name != self.target_func:
            return updated

        new_args = []
        for arg in updated.args:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == self.arg_name:
                new_args.append(arg.with_changes(value=cst.parse_expression(self.new_value)))
                self.changed = True
            else:
                new_args.append(arg)
        return updated.with_changes(args=new_args)


def apply_transform(source: str, transformer: cst.CSTTransformer) -> str:
    """Parse source, apply transformer, return modified source preserving formatting."""
    tree = cst.parse_module(source)
    modified = tree.visit(transformer)
    return modified.code
