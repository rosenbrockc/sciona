"""Phase 1: Deterministic AST extraction of a Python class's data flow.

No LLM calls.  Uses ``ast.NodeVisitor`` to parse the target class and
build a ``RawDataFlowGraph`` capturing method facts, ``self.*`` reads/writes,
config-gated branches, and the init preprocessing chain.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sciona.architect.models import DependencyEdge
from sciona.ingester.models import (
    AttributeAccess,
    AttributeSemanticFact,
    CallFact,
    ConfigBranch,
    FactProvenance,
    MethodFact,
    ParameterFact,
    RawDataFlowGraph,
    ReturnFact,
    SourceSpan,
    UnknownFact,
)

# ---------------------------------------------------------------------------
# AST visitors
# ---------------------------------------------------------------------------

# Common config container attribute names.
_CONFIG_CONTAINER_NAMES = frozenset(
    {
        "options",
        "config",
        "params",
        "settings",
        "opts",
        "cfg",
        "hparams",
    }
)

# Deep-learning base classes treated as opaque boundaries.
_OPAQUE_BASE_CLASSES: frozenset[str] = frozenset(
    {
        "nn.Module",
        "Module",
        "torch.nn.Module",
        "hk.Module",
        "tf.keras.Model",
        "tf.keras.layers.Layer",
        "keras.Model",
        "keras.layers.Layer",
        "flax.linen.Module",
    }
)


class _SelfAccessVisitor(ast.NodeVisitor):
    """Walk a method body and collect ``self.*`` reads, writes and config branches."""

    def __init__(
        self,
        method_name: str,
        config_attr_names: frozenset[str],
        source_path: str,
    ) -> None:
        self.method_name = method_name
        self.config_attr_names = config_attr_names
        self.source_path = source_path
        self.reads: list[AttributeAccess] = []
        self.writes: list[AttributeAccess] = []
        self.calls: list[str] = []
        self.call_facts: list[CallFact] = []
        self.config_branches: list[ConfigBranch] = []
        self.unknown_facts: list[UnknownFact] = []

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
        callee_expression = _unparse_or_empty(node.func)
        resolved_target = ""
        if isinstance(node.func, ast.Attribute):
            # 1. self.method() calls
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                self.calls.append(node.func.attr)
                resolved_target = node.func.attr

            # 2. self.attr.mutate() calls
            # Common mutating methods
            MUTATING_METHODS = {
                "append",
                "extend",
                "update",
                "pop",
                "insert",
                "remove",
                "clear",
                "sort",
            }
            if node.func.attr in MUTATING_METHODS:
                if isinstance(node.func.value, ast.Attribute):
                    inner = node.func.value
                    if isinstance(inner.value, ast.Name) and inner.value.id == "self":
                        self.writes.append(
                            AttributeAccess(
                                attr_name=inner.attr,
                                access_type="write",
                                method_name=self.method_name,
                                line_number=node.lineno,
                                is_config=False,
                            )
                        )
        elif isinstance(node.func, ast.Name) and node.func.id in {
            "getattr",
            "setattr",
            "hasattr",
        }:
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                self.unknown_facts.append(
                    UnknownFact(
                        reason=f"dynamic_{node.func.id}",
                        detail=_unparse_or_empty(node),
                        provenance=_make_provenance(
                            self.source_path,
                            node,
                            rule_id=f"python_ast.{node.func.id}",
                        ),
                    )
                )

        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            self.unknown_facts.append(
                UnknownFact(
                    reason="variadic_forwarding",
                    detail=_unparse_or_empty(node),
                    provenance=_make_provenance(
                        self.source_path,
                        node,
                        rule_id="python_ast.variadic_forwarding",
                    ),
                )
            )

        self.call_facts.append(
            CallFact(
                callee_expression=callee_expression,
                resolved_target=resolved_target,
                args=[_unparse_or_empty(arg) for arg in node.args],
                keywords=[
                    f"{kw.arg}={_unparse_or_empty(kw.value)}"
                    if kw.arg is not None
                    else f"**{_unparse_or_empty(kw.value)}"
                    for kw in node.keywords
                ],
                provenance=_make_provenance(
                    self.source_path,
                    node,
                    rule_id="python_ast.call",
                    evidence=callee_expression,
                ),
            )
        )
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
                self.writes.append(
                    AttributeAccess(
                        attr_name=target.attr,
                        access_type="write",
                        method_name=self.method_name,
                        line_number=lineno,
                        is_config=is_config,
                    )
                )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_self_write(elt, lineno)

    def _check_self_read(self, node: ast.Attribute, lineno: int) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            is_config = node.attr in self.config_attr_names
            self.reads.append(
                AttributeAccess(
                    attr_name=node.attr,
                    access_type="read",
                    method_name=self.method_name,
                    line_number=lineno,
                    is_config=is_config,
                )
            )

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
                            if isinstance(child, ast.Attribute) and isinstance(
                                child.value, ast.Name
                            ):
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


@dataclass
class _MethodExtractionResult:
    fact: MethodFact
    reads: list[AttributeAccess]
    writes: list[AttributeAccess]
    config_origin_attrs: set[str]


def _unparse_or_empty(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _source_span(source_path: str, node: ast.AST | None) -> SourceSpan:
    if node is None:
        return SourceSpan(file_path=source_path)
    line_start = int(getattr(node, "lineno", 0) or 0)
    line_end = int(getattr(node, "end_lineno", line_start) or line_start)
    col_start = int(getattr(node, "col_offset", 0) or 0)
    col_end = int(getattr(node, "end_col_offset", col_start) or col_start)
    return SourceSpan(
        file_path=source_path,
        line_start=line_start,
        line_end=line_end,
        col_start=col_start,
        col_end=col_end,
    )


def _make_provenance(
    source_path: str,
    node: ast.AST | None,
    *,
    rule_id: str,
    evidence: str = "",
) -> FactProvenance:
    return FactProvenance(
        rule_id=rule_id,
        span=_source_span(source_path, node),
        evidence=evidence,
    )


def _return_fact_from_node(source_path: str, node: ast.Return) -> ReturnFact:
    expr = node.value
    expression = _unparse_or_empty(expr)
    provenance = _make_provenance(
        source_path,
        node,
        rule_id="python_ast.return",
        evidence=expression,
    )
    if expr is None:
        return ReturnFact(kind="none", expression="", provenance=provenance)
    if isinstance(expr, ast.Constant):
        if expr.value is None:
            return ReturnFact(kind="none", expression="None", provenance=provenance)
        return ReturnFact(kind="constant", expression=expression, provenance=provenance)
    if isinstance(expr, ast.Name):
        if expr.id == "self":
            return ReturnFact(kind="self", expression="self", provenance=provenance)
        return ReturnFact(
            kind="parameter",
            expression=expression,
            provenance=provenance,
        )
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            return ReturnFact(
                kind="attribute",
                expression=expression,
                referenced_attrs=[expr.attr],
                provenance=provenance,
            )
        return ReturnFact(kind="unknown", expression=expression, provenance=provenance)
    if isinstance(expr, ast.Call):
        callee = _unparse_or_empty(expr.func)
        return ReturnFact(
            kind="call_result",
            expression=expression,
            referenced_callees=[callee] if callee else [],
            provenance=provenance,
        )
    if isinstance(expr, ast.Tuple):
        attrs: list[str] = []
        callees: list[str] = []
        for elt in expr.elts:
            if isinstance(elt, ast.Attribute) and isinstance(elt.value, ast.Name) and elt.value.id == "self":
                attrs.append(elt.attr)
            elif isinstance(elt, ast.Call):
                callee = _unparse_or_empty(elt.func)
                if callee:
                    callees.append(callee)
        return ReturnFact(
            kind="tuple",
            expression=expression,
            referenced_attrs=attrs,
            referenced_callees=callees,
            provenance=provenance,
        )
    return ReturnFact(kind="unknown", expression=expression, provenance=provenance)


def _extract_return_facts(
    method_node: ast.FunctionDef,
    source_path: str,
) -> tuple[list[ReturnFact], list[UnknownFact]]:
    return_facts: list[ReturnFact] = []
    unknowns: list[UnknownFact] = []
    for child in ast.walk(method_node):
        if isinstance(child, ast.Return):
            return_facts.append(_return_fact_from_node(source_path, child))
    kinds = {fact.kind for fact in return_facts}
    if len(kinds) > 1:
        unknowns.append(
            UnknownFact(
                reason="mixed_return_paths",
                detail=", ".join(sorted(kinds)),
                provenance=_make_provenance(
                    source_path,
                    method_node,
                    rule_id="python_ast.return_paths",
                ),
            )
        )
    return return_facts, unknowns


def _skip_bound_first_arg(method_node: ast.FunctionDef) -> bool:
    if not method_node.args.args:
        return False
    first = method_node.args.args[0].arg
    if first == "self":
        return True
    if first != "cls":
        return False
    decorators = {_unparse_or_empty(deco) for deco in method_node.decorator_list}
    return "classmethod" in decorators


def _build_signature(
    method_node: ast.FunctionDef,
    source_path: str,
) -> tuple[list[str], list[ParameterFact]]:
    args = method_node.args
    flat_params: list[str] = []
    signature: list[ParameterFact] = []
    skip_first = _skip_bound_first_arg(method_node)
    positional = list(args.posonlyargs) + list(args.args)
    positional_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for index, arg in enumerate(args.posonlyargs):
        if skip_first and index == 0:
            continue
        default = positional_defaults[index]
        flat_params.append(arg.arg)
        signature.append(
            ParameterFact(
                name=arg.arg,
                kind="positional_only",
                annotation=_unparse_or_empty(arg.annotation),
                default_expression=_unparse_or_empty(default),
                has_default=default is not None,
                provenance=_make_provenance(
                    source_path,
                    arg,
                    rule_id="python_ast.signature.positional_only",
                ),
            )
        )

    offset = len(args.posonlyargs)
    for index, arg in enumerate(args.args):
        if skip_first and index == 0 and offset == 0:
            continue
        default = positional_defaults[offset + index]
        flat_params.append(arg.arg)
        signature.append(
            ParameterFact(
                name=arg.arg,
                kind="positional_or_keyword",
                annotation=_unparse_or_empty(arg.annotation),
                default_expression=_unparse_or_empty(default),
                has_default=default is not None,
                provenance=_make_provenance(
                    source_path,
                    arg,
                    rule_id="python_ast.signature.positional_or_keyword",
                ),
            )
        )

    if args.vararg is not None:
        flat_params.append(args.vararg.arg)
        signature.append(
            ParameterFact(
                name=args.vararg.arg,
                kind="vararg",
                annotation=_unparse_or_empty(args.vararg.annotation),
                provenance=_make_provenance(
                    source_path,
                    args.vararg,
                    rule_id="python_ast.signature.vararg",
                ),
            )
        )

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        flat_params.append(arg.arg)
        signature.append(
            ParameterFact(
                name=arg.arg,
                kind="keyword_only",
                annotation=_unparse_or_empty(arg.annotation),
                default_expression=_unparse_or_empty(default),
                has_default=default is not None,
                provenance=_make_provenance(
                    source_path,
                    arg,
                    rule_id="python_ast.signature.keyword_only",
                ),
            )
        )

    if args.kwarg is not None:
        flat_params.append(args.kwarg.arg)
        signature.append(
            ParameterFact(
                name=args.kwarg.arg,
                kind="kwarg",
                annotation=_unparse_or_empty(args.kwarg.annotation),
                provenance=_make_provenance(
                    source_path,
                    args.kwarg,
                    rule_id="python_ast.signature.kwarg",
                ),
            )
        )

    return flat_params, signature


def _config_origin_attrs(
    method_node: ast.FunctionDef,
    params: list[str],
    config_attr_names: frozenset[str],
) -> set[str]:
    if method_node.name != "__init__":
        return set()
    param_names = set(params)
    config_origin: set[str] = set()
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue
            if target.attr in config_attr_names:
                config_origin.add(target.attr)
            elif isinstance(value, ast.Name) and value.id in param_names:
                config_origin.add(target.attr)
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in param_names
            ):
                config_origin.add(target.attr)
    return config_origin


def _classify_method_role(
    method_name: str,
    *,
    reads: list[str],
    writes: list[str],
    return_facts: list[ReturnFact],
) -> str:
    lower = method_name.lower()
    if method_name == "__init__":
        return "constructor"
    if method_name in {"__sklearn_tags__", "get_metadata_routing"}:
        return "query_or_metadata"
    if lower.startswith("_"):
        return "helper"
    if lower.startswith("set_") and writes:
        return "config_setter"
    if any(token in lower for token in ("fit", "partial_fit", "update", "train", "learn")) and writes:
        return "fit_or_update"
    if any(token in lower for token in ("predict", "transform", "infer", "decode", "encode")):
        return "predict_or_transform"
    if "score" in lower or "evaluate" in lower:
        return "score_or_evaluate"
    if (
        lower.startswith("get_")
        or lower.startswith("is_")
        or lower.startswith("has_")
        or (return_facts and not writes)
    ):
        return "query_or_metadata"
    if reads or writes:
        return "unknown"
    return "helper"


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
    source_path: str,
) -> _MethodExtractionResult:
    """Build a ``MethodFact`` from a single method AST node."""
    params, signature = _build_signature(method_node, source_path)

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
    visitor = _SelfAccessVisitor(method_node.name, config_attr_names, source_path)
    for stmt in method_node.body:
        visitor.visit(stmt)
    return_facts, return_unknowns = _extract_return_facts(method_node, source_path)

    # Extract decorators
    decorators = []
    for deco in method_node.decorator_list:
        try:
            decorators.append("@" + ast.unparse(deco))
        except Exception:
            pass

    reads = sorted({a.attr_name for a in visitor.reads})
    writes = sorted({a.attr_name for a in visitor.writes})
    config_origin_attrs = _config_origin_attrs(method_node, params, config_attr_names)
    role = _classify_method_role(
        method_node.name,
        reads=reads,
        writes=writes,
        return_facts=return_facts,
    )
    fact = MethodFact(
        name=method_node.name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        decorators=decorators,
        reads=reads,
        writes=writes,
        calls=sorted(set(visitor.calls)),
        config_branches=visitor.config_branches,
        source_code=method_source,
        signature=signature,
        return_facts=return_facts,
        call_facts=visitor.call_facts,
        unknown_facts=visitor.unknown_facts + return_unknowns,
        semantic_role=role,
        config_attributes=sorted(
            {access.attr_name for access in visitor.reads + visitor.writes if access.is_config}
            | config_origin_attrs
        ),
        provenance=[
            _make_provenance(
                source_path,
                method_node,
                rule_id="python_ast.method",
                evidence=method_node.name,
            )
        ],
    )
    return _MethodExtractionResult(
        fact=fact,
        reads=visitor.reads,
        writes=visitor.writes,
        config_origin_attrs=config_origin_attrs,
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


def _compute_fitted_attrs(methods: list[MethodFact]) -> set[str]:
    attr_writers: dict[str, set[str]] = defaultdict(set)
    attr_readers: dict[str, set[str]] = defaultdict(set)
    roles = {mf.name: mf.semantic_role for mf in methods}
    for mf in methods:
        for attr in mf.writes:
            attr_writers[attr].add(mf.name)
        for attr in mf.reads:
            attr_readers[attr].add(mf.name)

    fitted: set[str] = set()
    reader_roles = {"predict_or_transform", "score_or_evaluate", "query_or_metadata"}
    for attr, writers in attr_writers.items():
        non_init_writers = {name for name in writers if name != "__init__"}
        if not non_init_writers:
            continue
        if any(roles.get(name) == "fit_or_update" for name in non_init_writers):
            fitted.add(attr)
            continue
        if attr not in attr_readers:
            continue
        if any(roles.get(name) in reader_roles for name in attr_readers[attr]):
            fitted.add(attr)
    return fitted


def _build_attribute_facts(
    methods: list[MethodFact],
    method_results: list[_MethodExtractionResult],
    config_origin_attrs: set[str],
    fitted_attrs: set[str],
    source_path: str,
) -> list[AttributeSemanticFact]:
    facts: dict[str, dict[str, Any]] = {}
    role_by_method = {mf.name: mf.semantic_role for mf in methods}

    def _bucket(attr_name: str) -> dict[str, Any]:
        bucket = facts.get(attr_name)
        if bucket is None:
            bucket = {
                "read_methods": set(),
                "write_methods": set(),
                "provenances": [],
                "first_seen": "",
            }
            facts[attr_name] = bucket
        return bucket

    for result in method_results:
        for access in result.reads:
            bucket = _bucket(access.attr_name)
            bucket["read_methods"].add(access.method_name)
            bucket["provenances"].append(
                FactProvenance(
                    rule_id=f"python_ast.attr_{access.access_type}",
                    span=SourceSpan(
                        file_path=source_path,
                        line_start=access.line_number,
                        line_end=access.line_number,
                    ),
                    evidence=f"{access.method_name}:{access.attr_name}@{access.line_number}",
                )
            )
            if not bucket["first_seen"]:
                bucket["first_seen"] = access.method_name
        for access in result.writes:
            bucket = _bucket(access.attr_name)
            bucket["write_methods"].add(access.method_name)
            bucket["provenances"].append(
                FactProvenance(
                    rule_id=f"python_ast.attr_{access.access_type}",
                    span=SourceSpan(
                        file_path=source_path,
                        line_start=access.line_number,
                        line_end=access.line_number,
                    ),
                    evidence=f"{access.method_name}:{access.attr_name}@{access.line_number}",
                )
            )
            if not bucket["first_seen"]:
                bucket["first_seen"] = access.method_name

    for method in methods:
        for attr_name in method.reads:
            bucket = _bucket(attr_name)
            bucket["read_methods"].add(method.name)
            if not bucket["first_seen"]:
                bucket["first_seen"] = method.name
        for attr_name in method.writes:
            bucket = _bucket(attr_name)
            bucket["write_methods"].add(method.name)
            if not bucket["first_seen"]:
                bucket["first_seen"] = method.name

    attribute_facts: list[AttributeSemanticFact] = []
    for attr_name in sorted(facts):
        bucket = facts[attr_name]
        read_methods = sorted(bucket["read_methods"])
        write_methods = sorted(bucket["write_methods"])
        is_config = attr_name in config_origin_attrs or any(
            attr_name in method.config_attributes for method in methods
        )
        is_fitted = attr_name in fitted_attrs
        is_derived = bool(write_methods) and not is_config and not is_fitted
        query_methods = {
            name
            for name in set(read_methods) | set(write_methods)
            if role_by_method.get(name) == "query_or_metadata"
        }
        is_query_only = bool(query_methods) and set(write_methods) <= query_methods
        attribute_facts.append(
            AttributeSemanticFact(
                attr_name=attr_name,
                first_seen_in=bucket["first_seen"],
                read_methods=read_methods,
                write_methods=write_methods,
                is_config=is_config,
                is_fitted=is_fitted,
                is_derived=is_derived,
                is_query_only=is_query_only,
                provenances=bucket["provenances"],
            )
        )
    return attribute_facts


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
    method_results: list[_MethodExtractionResult] = []
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef):
            result = _extract_method_fact(
                item,
                config_attr_names,
                source_lines,
                source_path,
            )
            method_results.append(result)
    methods = [result.fact for result in method_results]
    config_origin_attrs = {
        attr for result in method_results for attr in result.config_origin_attrs
    }

    # Build all_attributes index: attr -> list of access types
    all_attributes: dict[str, list[str]] = {}
    for mf in methods:
        for attr in mf.reads:
            all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
        for attr in mf.writes:
            all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

    # Collect all config branches
    config_branches = [cb for mf in methods for cb in mf.config_branches]

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

    # Recursive state mutation bubbling
    # If method A calls B, then A inherits B reads and writes
    changed = True
    while changed:
        changed = False
        for mf in methods:
            if mf.name in internal_call_graph:
                for callee_name in internal_call_graph[mf.name]:
                    callee = next((m for m in methods if m.name == callee_name), None)
                    if callee:
                        # Inherit reads
                        new_reads = set(mf.reads) | set(callee.reads)
                        if len(new_reads) > len(mf.reads):
                            mf.reads = sorted(new_reads)
                            changed = True
                        # Inherit writes
                        new_writes = set(mf.writes) | set(callee.writes)
                        if len(new_writes) > len(mf.writes):
                            mf.writes = sorted(new_writes)
                            changed = True

    fitted_attrs = _compute_fitted_attrs(methods)
    attribute_facts = _build_attribute_facts(
        methods,
        method_results,
        config_origin_attrs,
        fitted_attrs,
        source_path,
    )
    for mf in methods:
        mf.fitted_attributes = sorted(
            {
                attr
                for attr in (*mf.reads, *mf.writes)
                if attr in fitted_attrs
            }
        )
    semantic_unknowns = [
        unknown
        for mf in methods
        for unknown in mf.unknown_facts
    ]

    return RawDataFlowGraph(
        class_name=class_name,
        source_code=source_code,
        methods=methods,
        all_attributes=all_attributes,
        config_branches=config_branches,
        init_chain=init_chain,
        cross_window_attrs=cross_window_attrs,
        internal_call_graph=internal_call_graph,
        attribute_facts=attribute_facts,
        config_attributes=sorted(config_origin_attrs),
        fitted_attributes=sorted(fitted_attrs),
        derived_attributes=sorted(
            fact.attr_name for fact in attribute_facts if fact.is_derived
        ),
        semantic_unknowns=semantic_unknowns,
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
    """Walk module-level statements and collect call sites + inferred edges."""

    def __init__(self, known_functions: set[str]) -> None:
        self.known_functions = known_functions
        self.call_sites: list[_CallSite] = []
        self.var_producers: dict[str, _CallSite | str] = {}  # var_name -> producer
        self.inferred_edges: list[DependencyEdge] = []
        self.external_tools: list[MethodFact] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            self._handle_call(node.value, node.targets, node.lineno)
        else:
            # Simple assignment: track variable lineage
            targets = self._collect_targets(node.targets)
            for t in targets:
                if isinstance(node.value, ast.Name):
                    self.var_producers[t] = self.var_producers.get(
                        node.value.id, "INTERNAL_VAR"
                    )
                else:
                    self.var_producers[t] = "INTERNAL_VAR"
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self._handle_call(node.value, [], node.lineno)
        self.generic_visit(node)

    def _handle_call(
        self, call_node: ast.Call, targets_nodes: list[ast.expr], lineno: int
    ) -> None:
        func_name = self._resolve_func_name(call_node.func)
        full_func_name = self._resolve_full_func_name(call_node.func)

        is_known = func_name in self.known_functions
        is_external = full_func_name in {
            "subprocess.run",
            "subprocess.call",
            "subprocess.check_output",
            "os.system",
        }

        if not (is_known or is_external):
            # For unknown calls, still mark targets as INTERNAL_VAR
            targets = self._collect_targets(targets_nodes)
            for t in targets:
                self.var_producers[t] = "INTERNAL_VAR"
            return

        # 1. CONSUME: Arguments read prior to the call
        args = self._collect_arg_names(call_node)
        for arg_var in args:
            producer = self.var_producers.get(arg_var)
            if producer and producer != "INTERNAL_VAR":
                # Determine source_id (string)
                source_id = (
                    producer.func_name
                    if hasattr(producer, "func_name")
                    else str(producer)
                )

                edge = DependencyEdge(
                    source_id=source_id,
                    target_id=func_name if is_known else full_func_name,
                    output_name=arg_var,
                    input_name=arg_var,
                    source_type="Any",
                    target_type="Any",
                )
                if edge not in self.inferred_edges:
                    self.inferred_edges.append(edge)

        # 2. PRODUCE: Targets assigned by the call
        targets = self._collect_targets(targets_nodes)

        cs = None
        if is_known:
            cs = _CallSite(
                func_name=func_name,
                targets=targets,
                args=args,
                lineno=lineno,
            )
            self.call_sites.append(cs)

        for t in targets:
            if cs:
                self.var_producers[t] = cs
            else:
                self.var_producers[t] = (
                    full_func_name if is_external else "INTERNAL_VAR"
                )

        if is_external:
            # Wrap as ExternalToolAtom
            self.external_tools.append(
                MethodFact(
                    name=full_func_name,
                    params=args,
                    docstring=f"External tool call: {full_func_name}",
                    source_code=ast.unparse(call_node),
                    is_external=True,
                )
            )

    @staticmethod
    def _resolve_full_func_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            val = _ProceduralBlockVisitor._resolve_full_func_name(node.value)
            if val:
                return f"{val}.{node.attr}"
            return node.attr
        return None

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
            edges.append(
                DependencyEdge(
                    source_id=producer.func_name,
                    target_id=cs.func_name,
                    output_name=arg_var,
                    input_name=arg_var,
                    source_type="Any",
                    target_type="Any",
                )
            )

    return edges


def _extract_function_fact(
    func_node: ast.FunctionDef,
    source_lines: list[str],
    source_path: str,
) -> MethodFact:
    """Build a ``MethodFact`` from a top-level function (no ``self.*`` tracking)."""
    params, signature = _build_signature(func_node, source_path)

    return_type = ""
    if func_node.returns:
        return_type = ast.unparse(func_node.returns)

    docstring = ast.get_docstring(func_node) or ""

    start = func_node.lineno - 1
    end = func_node.end_lineno or func_node.lineno
    source_code = "\n".join(source_lines[start:end])

    # Extract decorators
    decorators = []
    for deco in func_node.decorator_list:
        try:
            decorators.append("@" + ast.unparse(deco))
        except Exception:
            pass
    return_facts, unknown_facts = _extract_return_facts(func_node, source_path)

    return MethodFact(
        name=func_node.name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        source_code=source_code,
        decorators=decorators,
        signature=signature,
        return_facts=return_facts,
        unknown_facts=unknown_facts,
        semantic_role=_classify_method_role(
            func_node.name,
            reads=[],
            writes=[],
            return_facts=return_facts,
        ),
        provenance=[
            _make_provenance(
                source_path,
                func_node,
                rule_id="python_ast.function",
                evidence=func_node.name,
            )
        ],
    )


async def extract_function_data_flow(
    source_path: str, function_name: str
) -> RawDataFlowGraph:
    """Parse a Python file and extract the data-flow graph for a named function.

    Finds the target top-level function, collects all file-level functions it
    calls (direct + transitive), and builds a ``RawDataFlowGraph`` with the
    function's internal call tree — analogous to what ``extract_data_flow``
    does for a class's methods.

    Args:
        source_path: Path to the ``.py`` file.
        function_name: Name of the top-level function to extract.

    Returns:
        A ``RawDataFlowGraph`` with function facts and inferred SSA edges.

    Raises:
        FileNotFoundError: If *source_path* does not exist.
        ValueError: If *function_name* is not found as a top-level function.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    source_code = path.read_text()
    source_lines = source_code.splitlines()
    tree = ast.parse(source_code, filename=source_path)

    # Collect all top-level function definitions
    all_func_nodes: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            all_func_nodes[node.name] = node

    if function_name not in all_func_nodes:
        raise ValueError(
            f"Function '{function_name}' not found as a top-level function in {source_path}"
        )

    # Build a file-level call graph: for each function, which other
    # top-level functions does it call?
    file_call_graph: dict[str, set[str]] = {}
    all_func_names = set(all_func_nodes.keys())
    for fname, fnode in all_func_nodes.items():
        called: set[str] = set()
        for child in ast.walk(fnode):
            if isinstance(child, ast.Call):
                callee = None
                if isinstance(child.func, ast.Name):
                    callee = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    callee = child.func.attr
                if callee and callee in all_func_names and callee != fname:
                    called.add(callee)
        file_call_graph[fname] = called

    # Collect transitive closure of functions called from the target
    reachable: set[str] = set()
    frontier = {function_name}
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        frontier |= file_call_graph.get(current, set()) - reachable

    # Build MethodFact for each reachable function
    methods: list[MethodFact] = []
    for fname in sorted(reachable):
        fnode = all_func_nodes.get(fname)
        if fnode is not None:
            methods.append(_extract_function_fact(fnode, source_lines, source_path))

    # Internal call graph (only edges between reachable functions)
    internal_call_graph: dict[str, list[str]] = {}
    for fname in reachable:
        callees = sorted(file_call_graph.get(fname, set()) & reachable)
        if callees:
            internal_call_graph[fname] = callees

    # Infer SSA edges by running _ProceduralBlockVisitor over the target
    # function body (treating its statements like module-level statements)
    known_functions = reachable
    visitor = _ProceduralBlockVisitor(known_functions)
    target_node = all_func_nodes[function_name]
    for stmt in target_node.body:
        visitor.visit(stmt)
    inferred_edges = visitor.inferred_edges

    return RawDataFlowGraph(
        class_name=function_name,
        source_code=source_code,
        methods=methods,
        all_attributes={},
        internal_call_graph=internal_call_graph,
        inferred_edges=inferred_edges,
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
    methods = [
        _extract_function_fact(fn, source_lines, source_path) for fn in func_nodes
    ]

    # Run procedural block visitor over the entire module body
    visitor = _ProceduralBlockVisitor(known_functions)
    for stmt in tree.body:
        if not isinstance(stmt, ast.FunctionDef):
            visitor.visit(stmt)

    # SSA edges were inferred during the visit
    inferred_edges = visitor.inferred_edges

    # Build all_attributes from call sites
    all_attributes: dict[str, list[str]] = {}
    for cs in visitor.call_sites:
        for t in cs.targets:
            all_attributes.setdefault(t, []).append(f"write:{cs.func_name}")
        for a in cs.args:
            all_attributes.setdefault(a, []).append(f"read:{cs.func_name}")

    # Merge detected external tool calls as atoms
    methods.extend(visitor.external_tools)

    return RawDataFlowGraph(
        class_name=name,
        source_code=source_code,
        methods=methods,
        all_attributes=all_attributes,
        inferred_edges=inferred_edges,
    )
