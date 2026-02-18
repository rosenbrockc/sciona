"""Phase 1: Deterministic AST extraction of a Python class's data flow.

No LLM calls.  Uses ``ast.NodeVisitor`` to parse the target class and
build a ``RawDataFlowGraph`` capturing method facts, ``self.*`` reads/writes,
config-gated branches, and the init preprocessing chain.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from ageom.architect.models import DependencyEdge
from ageom.ingester.models import (
    AttributeAccess,
    ConfigBranch,
    MethodFact,
    RawDataFlowGraph,
)


# ---------------------------------------------------------------------------
# AST visitors
# ---------------------------------------------------------------------------

# Common config container attribute names.
_CONFIG_CONTAINER_NAMES = frozenset({
    "options", "config", "params", "settings", "opts", "cfg", "hparams",
})

# Deep-learning base classes treated as opaque boundaries.
_OPAQUE_BASE_CLASSES: frozenset[str] = frozenset({
    "nn.Module", "Module",
    "hk.Module",
    "tf.keras.Model", "tf.keras.layers.Layer",
    "keras.Model", "keras.layers.Layer",
    "flax.linen.Module",
})


class _SelfAccessVisitor(ast.NodeVisitor):
    """Walk a method body and collect ``self.*`` reads, writes and config branches."""

    def __init__(self, method_name: str, config_attr_names: frozenset[str]) -> None:
        self.method_name = method_name
        self.config_attr_names = config_attr_names
        self.reads: list[AttributeAccess] = []
        self.writes: list[AttributeAccess] = []
        self.calls: list[str] = []
        self.config_branches: list[ConfigBranch] = []

    # --- self.X = ... (Store context) ---

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_self_write(target, node.lineno)
        # Also check the value side for reads
        self._walk_for_reads(node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_self_write(node.target, node.lineno)
        self._walk_for_reads(node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target:
            self._check_self_write(node.target, node.lineno)
        if node.value:
            self._walk_for_reads(node.value)
        self.generic_visit(node)

    # --- self.X on the right-hand side (Load context) ---

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self._check_self_read(node, node.lineno)
        self.generic_visit(node)

    # --- self.method() calls ---

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "self":
                self.calls.append(node.func.attr)
        self.generic_visit(node)

    # --- if self.options.X branches ---

    def visit_If(self, node: ast.If) -> None:
        branch = self._check_config_branch(node)
        if branch is not None:
            self.config_branches.append(branch)
        self.generic_visit(node)

    # --- Helpers ---

    def _check_self_write(self, target: ast.expr, lineno: int) -> None:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "self":
                is_config = target.attr in self.config_attr_names
                self.writes.append(AttributeAccess(
                    attr_name=target.attr,
                    access_type="write",
                    method_name=self.method_name,
                    line_number=lineno,
                    is_config=is_config,
                ))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_self_write(elt, lineno)

    def _check_self_read(self, node: ast.Attribute, lineno: int) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            is_config = node.attr in self.config_attr_names
            self.reads.append(AttributeAccess(
                attr_name=node.attr,
                access_type="read",
                method_name=self.method_name,
                line_number=lineno,
                is_config=is_config,
            ))

    def _walk_for_reads(self, node: ast.expr) -> None:
        """Walk an expression subtree collecting self.X reads."""
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                self._check_self_read(child, getattr(child, "lineno", 0))

    def _check_config_branch(self, node: ast.If) -> ConfigBranch | None:
        """Detect ``if self.options.X`` or ``if self.config.X`` pattern."""
        test = node.test
        # Handle ``if self.options.X:``
        if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Attribute):
            inner = test.value
            if isinstance(inner.value, ast.Name) and inner.value.id == "self":
                if inner.attr in self.config_attr_names:
                    # Collect reads/writes inside the branch body
                    branch_reads: list[str] = []
                    branch_writes: list[str] = []
                    for stmt in node.body:
                        for child in ast.walk(stmt):
                            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                                if child.value.id == "self":
                                    if isinstance(child.ctx, ast.Store):
                                        branch_writes.append(child.attr)
                                    elif isinstance(child.ctx, ast.Load):
                                        branch_reads.append(child.attr)
                    end_line = node.body[-1].end_lineno or node.body[-1].lineno
                    return ConfigBranch(
                        config_attr=test.attr,
                        method=self.method_name,
                        lines=(node.lineno, end_line),
                        reads=branch_reads,
                        writes=branch_writes,
                    )
        return None


# ---------------------------------------------------------------------------
# Method-level extraction
# ---------------------------------------------------------------------------


def _detect_config_attr_names(cls_node: ast.ClassDef) -> frozenset[str]:
    """Heuristic: identify config containers assigned in ``__init__``.

    Looks for ``self.options = ...``, ``self.config = ...``, etc.
    """
    found: set[str] = set()
    for node in ast.walk(cls_node):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and target.attr in _CONFIG_CONTAINER_NAMES
                        ):
                            found.add(target.attr)
    return frozenset(found)


def _extract_method_fact(
    method_node: ast.FunctionDef,
    config_attr_names: frozenset[str],
    source_lines: list[str],
) -> MethodFact:
    """Build a ``MethodFact`` from a single method AST node."""
    # Parameters (skip 'self')
    params = [
        arg.arg for arg in method_node.args.args if arg.arg != "self"
    ]

    # Return type annotation
    return_type = ""
    if method_node.returns:
        return_type = ast.unparse(method_node.returns)

    # Docstring
    docstring = ast.get_docstring(method_node) or ""

    # Source code
    start = method_node.lineno - 1
    end = method_node.end_lineno or method_node.lineno
    method_source = "\n".join(source_lines[start:end])

    # Walk the body for self.* accesses
    visitor = _SelfAccessVisitor(method_node.name, config_attr_names)
    for stmt in method_node.body:
        visitor.visit(stmt)

    return MethodFact(
        name=method_node.name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        reads=sorted({a.attr_name for a in visitor.reads}),
        writes=sorted({a.attr_name for a in visitor.writes}),
        calls=sorted(set(visitor.calls)),
        config_branches=visitor.config_branches,
        source_code=method_source,
    )


# ---------------------------------------------------------------------------
# Cross-window and init chain analysis
# ---------------------------------------------------------------------------


def _compute_cross_window_attrs(methods: list[MethodFact]) -> list[str]:
    """Attributes read but not written in non-``__init__`` methods.

    These are cross-window state candidates: they were set in a previous
    invocation and read in the current one.
    """
    init_writes: set[str] = set()
    non_init_reads: set[str] = set()
    non_init_writes: set[str] = set()

    for mf in methods:
        if mf.name == "__init__":
            init_writes.update(mf.writes)
        else:
            non_init_reads.update(mf.reads)
            non_init_writes.update(mf.writes)

    # Attrs read in non-init that are also written in non-init = cross-window
    # (they persist across calls)
    cross_window = non_init_reads & non_init_writes
    return sorted(cross_window)


def _compute_init_chain(init_method: MethodFact | None) -> list[str]:
    """Trace sequential ``self.X = ...`` assignments in ``__init__``.

    Returns the ordered list of attribute names written, representing
    the preprocessing chain order.
    """
    if init_method is None:
        return []
    return list(init_method.writes)


# ---------------------------------------------------------------------------
# Opaque boundary detection
# ---------------------------------------------------------------------------


def _detect_opaque_bases(cls_node: ast.ClassDef) -> list[str]:
    """Return matched DL base class names, or empty list if transparent."""
    matched: list[str] = []
    for base in cls_node.bases:
        unparsed = ast.unparse(base)
        if unparsed in _OPAQUE_BASE_CLASSES:
            matched.append(unparsed)
    return matched


def _extract_opaque_boundary_fact(
    cls_node: ast.ClassDef,
    source_lines: list[str],
) -> MethodFact:
    """Extract a single MethodFact from the entry method of an opaque class.

    Priority: ``forward`` > ``__call__`` > ``call`` > first non-``__init__``.
    """
    priority = ["forward", "__call__", "call"]
    methods: dict[str, ast.FunctionDef] = {}
    first_non_init: ast.FunctionDef | None = None

    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef):
            methods[item.name] = item
            if first_non_init is None and item.name != "__init__":
                first_non_init = item

    entry: ast.FunctionDef | None = None
    for name in priority:
        if name in methods:
            entry = methods[name]
            break
    if entry is None:
        entry = first_non_init
    if entry is None:
        # Fallback: use __init__ if nothing else exists
        entry = methods.get("__init__")
    if entry is None:
        return MethodFact(name="unknown", is_opaque=True)

    params = [arg.arg for arg in entry.args.args if arg.arg != "self"]
    return_type = ast.unparse(entry.returns) if entry.returns else ""
    docstring = ast.get_docstring(entry) or ""

    return MethodFact(
        name=entry.name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        is_opaque=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def extract_data_flow(source_path: str, class_name: str) -> RawDataFlowGraph:
    """Parse a Python file and extract the data-flow graph for a class.

    Args:
        source_path: Path to the ``.py`` file.
        class_name: Name of the class to extract.

    Returns:
        A ``RawDataFlowGraph`` with methods, attributes, config branches,
        init chain, and cross-window state candidates.

    Raises:
        FileNotFoundError: If *source_path* does not exist.
        ValueError: If *class_name* is not found in the file.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    source_code = path.read_text()
    source_lines = source_code.splitlines()

    tree = ast.parse(source_code, filename=source_path)

    # Find the target class
    cls_node: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cls_node = node
            break

    if cls_node is None:
        raise ValueError(f"Class '{class_name}' not found in {source_path}")

    # Check for opaque DL base classes — early return
    opaque_bases = _detect_opaque_bases(cls_node)
    if opaque_bases:
        boundary_fact = _extract_opaque_boundary_fact(cls_node, source_lines)
        return RawDataFlowGraph(
            class_name=class_name,
            source_code=source_code,
            methods=[boundary_fact],
            all_attributes={},
            is_opaque=True,
            opaque_base_classes=opaque_bases,
        )

    # Detect config container names
    config_attr_names = _detect_config_attr_names(cls_node)

    # Extract method facts
    methods: list[MethodFact] = []
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef):
            mf = _extract_method_fact(item, config_attr_names, source_lines)
            methods.append(mf)

    # Build all_attributes index: attr -> list of access types
    all_attributes: dict[str, list[str]] = {}
    for mf in methods:
        for attr in mf.reads:
            all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
        for attr in mf.writes:
            all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

    # Collect all config branches
    config_branches = [
        cb for mf in methods for cb in mf.config_branches
    ]

    # Init chain
    init_method = next((mf for mf in methods if mf.name == "__init__"), None)
    init_chain = _compute_init_chain(init_method)

    # Cross-window attrs
    cross_window_attrs = _compute_cross_window_attrs(methods)

    # Internal call graph
    internal_call_graph: dict[str, list[str]] = {}
    method_names = {mf.name for mf in methods}
    for mf in methods:
        internal_calls = [c for c in mf.calls if c in method_names]
        if internal_calls:
            internal_call_graph[mf.name] = internal_calls

    return RawDataFlowGraph(
        class_name=class_name,
        source_code=source_code,
        methods=methods,
        all_attributes=all_attributes,
        config_branches=config_branches,
        init_chain=init_chain,
        cross_window_attrs=cross_window_attrs,
        internal_call_graph=internal_call_graph,
    )


# ---------------------------------------------------------------------------
# Procedural / script-level SSA edge inference
# ---------------------------------------------------------------------------


@dataclass
class _CallSite:
    """A single ``y = func(x)`` call found at module level."""

    func_name: str
    targets: list[str]
    args: list[str]
    lineno: int


class _ProceduralBlockVisitor(ast.NodeVisitor):
    """Walk module-level statements and collect call sites + variable producers."""

    def __init__(self, known_functions: set[str]) -> None:
        self.known_functions = known_functions
        self.call_sites: list[_CallSite] = []
        self.var_producers: dict[str, _CallSite] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        if not isinstance(node.value, ast.Call):
            self.generic_visit(node)
            return

        func_name = self._resolve_func_name(node.value.func)
        if func_name is None or func_name not in self.known_functions:
            self.generic_visit(node)
            return

        # Collect LHS targets (handle tuple unpacking)
        targets = self._collect_targets(node.targets)

        # Collect variable names used as arguments
        args = self._collect_arg_names(node.value)

        cs = _CallSite(
            func_name=func_name,
            targets=targets,
            args=args,
            lineno=node.lineno,
        )
        self.call_sites.append(cs)

        # Record each target variable as produced by this call site
        for t in targets:
            self.var_producers[t] = cs

        self.generic_visit(node)

    # --- helpers ---

    @staticmethod
    def _resolve_func_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @staticmethod
    def _collect_targets(targets: list[ast.expr]) -> list[str]:
        names: list[str] = []
        for t in targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        return names

    @staticmethod
    def _collect_arg_names(call_node: ast.Call) -> list[str]:
        names: list[str] = []
        for arg in call_node.args:
            for child in ast.walk(arg):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    names.append(child.id)
        for kw in call_node.keywords:
            for child in ast.walk(kw.value):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    names.append(child.id)
        return names


def _infer_ssa_edges(
    call_sites: list[_CallSite],
    var_producers: dict[str, _CallSite],
) -> list[DependencyEdge]:
    """Build dependency edges from SSA variable tracking.

    For each call site, if an argument variable was produced by a prior
    call site, create an edge from the producer to the consumer.
    """
    seen: set[tuple[str, str, str]] = set()
    edges: list[DependencyEdge] = []

    for cs in call_sites:
        for arg_var in cs.args:
            producer = var_producers.get(arg_var)
            if producer is None or producer is cs:
                continue
            key = (producer.func_name, cs.func_name, arg_var)
            if key in seen:
                continue
            seen.add(key)
            edges.append(DependencyEdge(
                source_id=producer.func_name,
                target_id=cs.func_name,
                output_name=arg_var,
                input_name=arg_var,
                source_type="Any",
                target_type="Any",
            ))

    return edges


def _extract_function_fact(
    func_node: ast.FunctionDef,
    source_lines: list[str],
) -> MethodFact:
    """Build a ``MethodFact`` from a top-level function (no ``self.*`` tracking)."""
    params = [arg.arg for arg in func_node.args.args]

    return_type = ""
    if func_node.returns:
        return_type = ast.unparse(func_node.returns)

    docstring = ast.get_docstring(func_node) or ""

    start = func_node.lineno - 1
    end = func_node.end_lineno or func_node.lineno
    source_code = "\n".join(source_lines[start:end])

    return MethodFact(
        name=func_node.name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        source_code=source_code,
    )


async def extract_procedural_data_flow(
    source_path: str,
    pipeline_name: str | None = None,
    entry_block: str | None = None,
) -> RawDataFlowGraph:
    """Parse a procedural Python file and extract SSA-based data flow.

    Args:
        source_path: Path to the ``.py`` file.
        pipeline_name: Display name for the pipeline (defaults to file stem).
        entry_block: Unused, reserved for future notebook cell selection.

    Returns:
        A ``RawDataFlowGraph`` with function facts and inferred SSA edges.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    source_code = path.read_text()
    source_lines = source_code.splitlines()
    tree = ast.parse(source_code, filename=source_path)

    name = pipeline_name or path.stem

    # Collect top-level function definitions
    known_functions: set[str] = set()
    func_nodes: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            known_functions.add(node.name)
            func_nodes.append(node)

    # Build MethodFact for each function
    methods = [_extract_function_fact(fn, source_lines) for fn in func_nodes]

    # Run procedural block visitor over the entire module body
    visitor = _ProceduralBlockVisitor(known_functions)
    for stmt in tree.body:
        if not isinstance(stmt, ast.FunctionDef):
            visitor.visit(stmt)

    # Infer SSA edges
    inferred_edges = _infer_ssa_edges(visitor.call_sites, visitor.var_producers)

    # Build all_attributes from call sites
    all_attributes: dict[str, list[str]] = {}
    for cs in visitor.call_sites:
        for t in cs.targets:
            all_attributes.setdefault(t, []).append(f"write:{cs.func_name}")
        for a in cs.args:
            all_attributes.setdefault(a, []).append(f"read:{cs.func_name}")

    return RawDataFlowGraph(
        class_name=name,
        source_code=source_code,
        methods=methods,
        all_attributes=all_attributes,
        inferred_edges=inferred_edges,
    )
