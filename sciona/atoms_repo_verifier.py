"""Static verifier for atoms repositories across supported package layouts.

The verifier focuses on actionable completeness failures in generated atom
modules: undefined witness symbols, missing state-model imports, unresolved
implementation classes, missing common aliases (``np``), and syntax/import
smoke failures when explicitly requested.
"""

from __future__ import annotations

import ast
import builtins
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_BUILTIN_NAMES = set(dir(builtins))
_COMMON_ALIAS_HINTS = {
    "np": "Add `import numpy as np`.",
    "pd": "Add `import pandas as pd`.",
    "nx": "Add `import networkx as nx`.",
    "jnp": "Add `import jax.numpy as jnp`.",
    "hk": "Add `import haiku as hk`.",
    "torch": "Add `import torch`.",
    "jax": "Add `import jax`.",
}


@dataclass(frozen=True)
class VerificationIssue:
    path: str
    line: int
    column: int
    code: str
    symbol: str
    message: str
    hint: str
    context: str


@dataclass(frozen=True)
class VerificationReport:
    repo_root: str
    package: str
    issues: list[VerificationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_json(self) -> str:
        return json.dumps(
            {
                "repo_root": self.repo_root,
                "package": self.package,
                "ok": self.ok,
                "issue_count": len(self.issues),
                "issues": [asdict(issue) for issue in self.issues],
            },
            indent=2,
        )

    def to_text(self) -> str:
        if not self.issues:
            return f"OK: no atom completeness issues found under {self.package}"
        lines = [
            f"Found {len(self.issues)} atom completeness issue(s) under {self.package}:"
        ]
        for issue in self.issues:
            location = f"{issue.path}:{issue.line}:{issue.column}"
            lines.append(f"- {location} [{issue.code}] {issue.message}")
            if issue.symbol:
                lines.append(f"  symbol: {issue.symbol}")
            if issue.context:
                lines.append(f"  context: {issue.context}")
            if issue.hint:
                lines.append(f"  hint: {issue.hint}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _ModuleInfo:
    path: Path
    source: str
    tree: ast.AST | None
    top_level_defs: set[str]
    import_error: str | None = None


def verify_atoms_repo(
    repo_root: str | Path,
    package: str,
    *,
    import_smoke: bool = False,
    python_executable: str | None = None,
) -> VerificationReport:
    root = Path(repo_root).expanduser().resolve()
    package_root = root / Path(package.replace(".", "/"))
    if not package_root.exists():
        issue = VerificationIssue(
            path=str(package_root),
            line=1,
            column=1,
            code="missing-package",
            symbol=package,
            message=f"Package root not found for '{package}'.",
            hint=f"Ensure '{package}' exists under {root}.",
            context="",
        )
        return VerificationReport(str(root), package, [issue])

    modules = _load_modules(package_root)
    symbol_index = _build_symbol_index(modules)
    issues: list[VerificationIssue] = []

    for info in modules.values():
        if info.tree is None:
            assert info.import_error is not None
            issues.append(
                VerificationIssue(
                    path=str(info.path),
                    line=1,
                    column=1,
                    code="syntax-error",
                    symbol="",
                    message="Module does not parse.",
                    hint="Fix the Python syntax before re-running verification.",
                    context=info.import_error,
                )
            )
            continue
        visitor = _UndefinedNameVisitor(info, modules, symbol_index)
        visitor.visit(info.tree)
        issues.extend(visitor.issues)

    if import_smoke:
        issues.extend(
            _run_import_smoke(
                root,
                package,
                python_executable=python_executable or sys.executable,
            )
        )

    issues.sort(key=lambda issue: (issue.path, issue.line, issue.column, issue.code))
    return VerificationReport(str(root), package, issues)


def _load_modules(package_root: Path) -> dict[Path, _ModuleInfo]:
    modules: dict[Path, _ModuleInfo] = {}
    for path in sorted(package_root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            top_defs = _collect_scope_names(tree.body, include_params=False)
            modules[path] = _ModuleInfo(path=path, source=source, tree=tree, top_level_defs=top_defs)
        except SyntaxError as exc:
            modules[path] = _ModuleInfo(
                path=path,
                source=path.read_text(encoding="utf-8", errors="replace"),
                tree=None,
                top_level_defs=set(),
                import_error=str(exc),
            )
    return modules


def _build_symbol_index(modules: dict[Path, _ModuleInfo]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path, info in modules.items():
        for name in info.top_level_defs:
            index.setdefault(name, []).append(path)
    return index


def _collect_scope_names(stmts: list[ast.stmt], *, include_params: bool) -> set[str]:
    names: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split(".")[0])
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
            for target in _statement_targets(stmt):
                names.update(_target_names(target))
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                names.update(_collect_scope_names(stmt.body, include_params=include_params))
                names.update(_collect_scope_names(stmt.orelse, include_params=include_params))
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                names.update(_collect_scope_names(stmt.body, include_params=include_params))
            continue
        if isinstance(stmt, ast.Try):
            names.update(_collect_scope_names(stmt.body, include_params=include_params))
            names.update(_collect_scope_names(stmt.orelse, include_params=include_params))
            names.update(_collect_scope_names(stmt.finalbody, include_params=include_params))
            for handler in stmt.handlers:
                if handler.name:
                    names.add(handler.name)
                names.update(_collect_scope_names(handler.body, include_params=include_params))
            continue
        if isinstance(stmt, (ast.If, ast.While)):
            names.update(_collect_scope_names(stmt.body, include_params=include_params))
            names.update(_collect_scope_names(stmt.orelse, include_params=include_params))
            continue
        if isinstance(stmt, ast.Match):
            names.update(_collect_scope_names(stmt.body, include_params=include_params))
            for case in stmt.cases:
                names.update(_collect_scope_names(case.body, include_params=include_params))
            continue
    return names


def _statement_targets(stmt: ast.stmt) -> list[ast.expr]:
    if isinstance(stmt, ast.Assign):
        return list(stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return [stmt.target]
    if isinstance(stmt, ast.AugAssign):
        return [stmt.target]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.target]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in stmt.items if item.optional_vars is not None]
    return []


def _target_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names


class _UndefinedNameVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: _ModuleInfo,
        modules: dict[Path, _ModuleInfo],
        symbol_index: dict[str, list[Path]],
    ) -> None:
        self.module = module
        self.modules = modules
        self.symbol_index = symbol_index
        self.issues: list[VerificationIssue] = []
        self.scope_stack: list[set[str]] = [set(module.top_level_defs) | _BUILTIN_NAMES]

    def visit_Name(self, node: ast.Name) -> Any:
        if not isinstance(node.ctx, ast.Load):
            return None
        if node.id in {"__name__", "__file__", "__package__", "__doc__", "__annotations__"}:
            return None
        if any(node.id in scope for scope in reversed(self.scope_stack)):
            return None
        self.issues.append(
            VerificationIssue(
                path=str(self.module.path),
                line=node.lineno,
                column=node.col_offset + 1,
                code="undefined-name",
                symbol=node.id,
                message=f"Unresolved symbol '{node.id}'.",
                hint=_suggest_fix(node.id, self.module.path, self.modules, self.symbol_index),
                context=_line_context(self.module.source, node.lineno),
            )
        )
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function_like(node)
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function_like(node)
        return None

    def _visit_function_like(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        local_names = _collect_function_locals(node)
        self.scope_stack.append(self.scope_stack[-1] | local_names)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        local_names = {arg.arg for arg in node.args.args}
        local_names.update(arg.arg for arg in node.args.kwonlyargs)
        if node.args.vararg is not None:
            local_names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            local_names.add(node.args.kwarg.arg)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.scope_stack.append(self.scope_stack[-1] | local_names)
        self.visit(node.body)
        self.scope_stack.pop()
        return None

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self._visit_comprehension(node)
        return None

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self._visit_comprehension(node)
        return None

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self._visit_comprehension(node)
        return None

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self._visit_comprehension(node)
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        class_locals = _collect_scope_names(node.body, include_params=False)
        self.scope_stack.append(self.scope_stack[-1] | class_locals)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()
        return None

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        # Push a comprehension scope that grows as each generator adds its
        # target variable(s).  This mirrors Python semantics: the iter of the
        # first generator is evaluated in the enclosing scope, but each
        # target is visible to subsequent ifs, iters, and the final element.
        self.scope_stack.append(set(self.scope_stack[-1]))
        for generator in node.generators:
            self.visit(generator.iter)
            self.scope_stack[-1].update(_target_names(generator.target))
            for if_clause in generator.ifs:
                self.visit(if_clause)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scope_stack.pop()


def _collect_function_locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {arg.arg for arg in node.args.args}
    names.update(arg.arg for arg in node.args.kwonlyargs)
    names.update(arg.arg for arg in getattr(node.args, "posonlyargs", []))
    if node.args.vararg is not None:
        names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.add(node.args.kwarg.arg)
    names.update(_collect_scope_names(node.body, include_params=False))
    return names


def _suggest_fix(
    symbol: str,
    current_path: Path,
    modules: dict[Path, _ModuleInfo],
    symbol_index: dict[str, list[Path]],
) -> str:
    if symbol in _COMMON_ALIAS_HINTS:
        return _COMMON_ALIAS_HINTS[symbol]

    sibling_matches = []
    for path in symbol_index.get(symbol, []):
        if path == current_path:
            continue
        if path.parent == current_path.parent:
            sibling_matches.append(path)
    if sibling_matches:
        target = sibling_matches[0]
        module_name = target.stem
        return f"Import `{symbol}` from sibling module `{module_name}`."

    if symbol.startswith("witness_"):
        candidate = current_path.with_name(f"{current_path.stem}_witnesses.py")
        if candidate in modules:
            return f"Add `from .{candidate.stem} import {symbol}`."

    if symbol.endswith("State"):
        state_candidates = [
            current_path.with_name(f"{current_path.stem}_state.py"),
            current_path.with_name("state_models.py"),
        ]
        for candidate in state_candidates:
            if candidate in modules:
                return f"Add `from .{candidate.stem} import {symbol}`."

    return "Define the symbol or import it explicitly in this module."


def _line_context(source: str, lineno: int) -> str:
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _run_import_smoke(
    repo_root: Path,
    package: str,
    *,
    python_executable: str,
) -> list[VerificationIssue]:
    code = (
        "import importlib, json, pkgutil, sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"pkg = importlib.import_module({package!r})\n"
        "issues = []\n"
        "for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + '.'):\n"
        "    try:\n"
        "        importlib.import_module(modname)\n"
        "    except Exception as exc:\n"
        "        issues.append({'module': modname, 'error': repr(exc)})\n"
        "print(json.dumps(issues))\n"
    )
    proc = subprocess.run(
        [python_executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [
            VerificationIssue(
                path=str(repo_root),
                line=1,
                column=1,
                code="import-smoke-failed",
                symbol=package,
                message="Import smoke verification could not start.",
                hint="Check the Python executable and repo path passed to the verifier.",
                context=(proc.stderr or proc.stdout).strip(),
            )
        ]
    try:
        data = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return [
            VerificationIssue(
                path=str(repo_root),
                line=1,
                column=1,
                code="import-smoke-malformed",
                symbol=package,
                message="Import smoke verification returned malformed output.",
                hint="Inspect the verifier subprocess output.",
                context=(proc.stdout or proc.stderr).strip(),
            )
        ]
    issues: list[VerificationIssue] = []
    for item in data:
        issues.append(
            VerificationIssue(
                path=str(repo_root / Path(item["module"].replace(".", "/")).with_suffix(".py")),
                line=1,
                column=1,
                code="import-error",
                symbol=item["module"],
                message=f"Module import failed for '{item['module']}'.",
                hint="Import the missing dependency explicitly or make the module self-contained.",
                context=item["error"],
            )
        )
    return issues
