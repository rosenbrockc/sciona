"""Tree-sitter based extractor for C++ and Julia source files.

Parses foreign-language classes/structs via tree-sitter and produces the
same ``RawDataFlowGraph`` schema as the Python AST extractor, enabling
downstream phases (chunker, emitter) to work unchanged.
"""

from __future__ import annotations

from pathlib import Path
import re

from tree_sitter import Node as TSNode
from tree_sitter_language_pack import get_parser

from sciona.architect.models import DependencyEdge
from sciona.ingester.base_extractor import SourceLanguage
from sciona.ingester.models import MethodFact, OracleEdge, RawDataFlowGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_text(node: TSNode) -> str:
    """Decode a tree-sitter node's text to a Python string."""
    return node.text.decode("utf-8") if node.text else ""


def _find_children(node: TSNode, type_name: str) -> list[TSNode]:
    """Return direct children matching *type_name*."""
    return [c for c in node.children if c.type == type_name]


def _find_descendants(node: TSNode, type_name: str) -> list[TSNode]:
    """Return all descendants (BFS) matching *type_name*."""
    result: list[TSNode] = []
    stack = list(node.children)
    while stack:
        n = stack.pop(0)
        if n.type == type_name:
            result.append(n)
        stack.extend(n.children)
    return result


# ---------------------------------------------------------------------------
# Function-level extraction helpers (used by extract_function)
# ---------------------------------------------------------------------------


def _extract_callees_from_body(node: TSNode) -> set[str]:
    """Extract all function call names from a tree-sitter subtree."""
    callees: set[str] = set()
    for call_node in _find_descendants(node, "call_expression"):
        if not call_node.children:
            continue
        func_expr = call_node.children[0]
        if func_expr.type == "identifier":
            callees.add(_node_text(func_expr))
        elif func_expr.type in ("field_expression", "field_identifier"):
            ids = [c for c in func_expr.children
                   if c.type in ("identifier", "field_identifier")]
            if ids:
                callees.add(_node_text(ids[-1]))
        elif func_expr.type == "scoped_identifier":
            ids = [c for c in func_expr.children if c.type == "identifier"]
            if ids:
                callees.add(_node_text(ids[-1]))
    return callees


def _collect_all_function_nodes(
    root: TSNode, language: "SourceLanguage",
) -> dict[str, TSNode]:
    """Collect all function definition nodes in a file, keyed by name.

    For Rust includes functions inside impl blocks.  For Julia, if multiple
    dispatch produces duplicate names the last definition wins.
    """
    from .base_extractor import SourceLanguage  # avoid circular at module level

    func_map: dict[str, TSNode] = {}

    if language == SourceLanguage.RUST:
        for fn_node in _find_descendants(root, "function_item"):
            for child in fn_node.children:
                if child.type == "identifier":
                    func_map[_node_text(child)] = fn_node
                    break

    elif language == SourceLanguage.JULIA:
        for fn_node in _find_descendants(root, "function_definition"):
            name = _julia_function_name(fn_node)
            if name:
                func_map[name] = fn_node

    elif language == SourceLanguage.HASKELL:
        for fn_node in _find_descendants(root, "function"):
            is_def = False
            name = ""
            for child in fn_node.children:
                if child.type == "variable":
                    name = _node_text(child)
                if child.type in ("exp", "body", "match"):
                    is_def = True
            if is_def and name:
                func_map[name] = fn_node

    else:  # C++
        # Include both function_definition (with body) and declaration
        # (header-only forward declarations with function_declarator).
        for node_type in ("function_definition", "declaration"):
            for fn_node in _find_descendants(root, node_type):
                for child in fn_node.children:
                    if child.type == "function_declarator":
                        for fcc in child.children:
                            if fcc.type in ("identifier", "field_identifier"):
                                name = _node_text(fcc)
                                # Prefer definitions over declarations
                                if name not in func_map or node_type == "function_definition":
                                    func_map[name] = fn_node
                                break
                        break

    return func_map


def _julia_function_name(fn_node: TSNode) -> str:
    """Extract function name from a Julia function_definition node."""
    for child in fn_node.children:
        if child.type != "signature":
            continue
        for sc in child.children:
            if sc.type == "call_expression":
                return _julia_call_sig_name(sc)
            elif sc.type == "typed_expression":
                for tc in sc.children:
                    if tc.type == "call_expression":
                        return _julia_call_sig_name(tc)
            elif sc.type == "where_expression":
                for wc in sc.children:
                    if wc.type == "call_expression":
                        return _julia_call_sig_name(wc)
    return ""


def _julia_call_sig_name(call_node: TSNode) -> str:
    """Extract the function name from a Julia call_expression signature."""
    for child in call_node.children:
        if child.type == "identifier":
            return _node_text(child)
        elif child.type == "field_expression":
            ids = [c for c in child.children if c.type == "identifier"]
            if ids:
                return _node_text(ids[-1])
    return ""


def _build_file_call_graph_ts(
    func_nodes: dict[str, TSNode],
) -> dict[str, set[str]]:
    """Build a file-level call graph from tree-sitter function nodes."""
    all_names = set(func_nodes.keys())
    graph: dict[str, set[str]] = {}
    for fname, fnode in func_nodes.items():
        callees = _extract_callees_from_body(fnode)
        graph[fname] = (callees & all_names) - {fname}
    return graph


def _transitive_closure(start: str, graph: dict[str, set[str]]) -> set[str]:
    """Compute transitive closure of reachable names from *start*."""
    reachable: set[str] = set()
    frontier = {start}
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        frontier |= graph.get(current, set()) - reachable
    return reachable


# Tree-sitter queries for probabilistic/oracle interfaces.
_RUST_TRAIT_BOUNDS_QUERY = """
(impl_item
  (type_parameters
    (type_parameter
      (type_identifier) @rust.typevar
      (trait_bounds
        [
          (type_identifier) @rust.trait_bound
          (scoped_type_identifier) @rust.trait_bound
        ]))) @rust.impl_type_bound)

(impl_item
  (where_clause
    (where_predicate
      (type_identifier) @rust.where_typevar
      (trait_bounds
        [
          (type_identifier) @rust.trait_bound
          (scoped_type_identifier) @rust.trait_bound
        ]))) @rust.impl_where_bound)

(function_item
  (type_parameters
    (type_parameter
      (type_identifier) @rust.fn_typevar
      (trait_bounds
        [
          (type_identifier) @rust.trait_bound
          (scoped_type_identifier) @rust.trait_bound
        ]))) @rust.fn_type_bound)

(function_item
  (where_clause
    (where_predicate
      (type_identifier) @rust.fn_where_typevar
      (trait_bounds
        [
          (type_identifier) @rust.trait_bound
          (scoped_type_identifier) @rust.trait_bound
        ]))) @rust.fn_where_bound)
"""

_RUST_ORACLE_CALL_QUERY = """
(call_expression
  (field_expression) @rust.oracle_function
  (arguments) @rust.oracle_args
) @rust.oracle_call
"""

_JULIA_TYPED_DISPATCH_QUERY = """
((function_definition
   (signature
     (call_expression
       [
         (identifier) @julia.dispatch_name
         (field_expression
           (identifier) @julia.dispatch_module
           (identifier) @julia.dispatch_name
         )
       ]
       (argument_list
         (typed_expression) @julia.typed_param
       )
     )
   )
 ) @julia.dispatch_fn)

((function_definition
   (signature
     (where_expression
       (call_expression
         [
           (identifier) @julia.dispatch_name
           (field_expression
             (identifier) @julia.dispatch_module
             (identifier) @julia.dispatch_name
           )
         ]
         (argument_list
           (typed_expression) @julia.typed_param
         )
       )
       (curly_expression
         (binary_expression
           (identifier) @julia.typevar
           (operator)
           [
             (identifier) @julia.type_bound
             (field_expression) @julia.type_bound
           ]
         )
       )
     )
   )
 ) @julia.dispatch_fn)
"""

_JULIA_ORACLE_CALL_QUERY = """
(call_expression
  (field_expression
    (identifier) @julia.oracle_obj
    (identifier) @julia.oracle_method
  )
  (argument_list) @julia.oracle_args
) @julia.oracle_call

(call_expression
  (identifier) @julia.oracle_method
  (argument_list
    (identifier) @julia.oracle_obj
  )
) @julia.oracle_call
"""

_ORACLE_CALL_METHOD_HINTS: frozenset[str] = frozenset(
    {
        "log_prob",
        "logprob",
        "logdensity",
        "log_density",
        "log_likelihood",
        "likelihood",
        "evaluate_likelihood",
        "score",
        "gradient",
        "grad",
        "value_and_grad",
        "value_and_gradient",
    }
)


def _node_key(node: TSNode) -> tuple[int, int, int, int]:
    """Stable tuple key for a tree-sitter node."""
    return (
        node.start_point[0],
        node.start_point[1],
        node.end_point[0],
        node.end_point[1],
    )


def _normalize_trait_name(trait_name: str) -> str:
    """Normalize Rust trait bounds like ``foo::Bar`` to ``Bar``."""
    return trait_name.split("::")[-1].strip()


def _normalize_type_name(type_name: str) -> str:
    """Normalize Julia type expressions to a base identifier."""
    base = type_name.strip()
    if "{" in base:
        base = base.split("{", 1)[0]
    if "." in base:
        base = base.split(".")[-1]
    return base


def _extract_identifiers_from_args(arg_node: TSNode) -> list[str]:
    """Extract identifier tokens from a call argument list node."""
    names: list[str] = []
    for ident in _find_descendants(arg_node, "identifier"):
        name = _node_text(ident).strip()
        if name and name not in names:
            names.append(name)
    return names


def _oracle_outputs_for_method(method_name: str) -> list[str]:
    """Infer oracle output channels from method names."""
    name = method_name.lower()
    outputs: list[str] = []
    if any(
        tok in name
        for tok in ("log_prob", "logprob", "logdensity", "log_density", "score")
    ):
        outputs.append("log_prob")
    if any(tok in name for tok in ("likelihood",)):
        outputs.append("likelihood")
    if any(tok in name for tok in ("gradient", "grad")):
        outputs.append("gradient")
    if not outputs:
        outputs.append("oracle_value")
    return outputs


def _add_edge_if_new(
    edges: list[DependencyEdge],
    seen: set[tuple[str, str, str, str, str, str]],
    edge: DependencyEdge,
) -> None:
    key = (
        edge.source_id,
        edge.target_id,
        edge.output_name,
        edge.input_name,
        edge.source_type,
        edge.target_type,
    )
    if key not in seen:
        seen.add(key)
        edges.append(edge)


# ---------------------------------------------------------------------------
# C++ class visitor
# ---------------------------------------------------------------------------


class _CppClassVisitor:
    """Extract class metadata from a ``class_specifier`` tree-sitter node."""

    def __init__(self, class_node: TSNode, source_code: str) -> None:
        self.class_node = class_node
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.class_name = ""
        self.base_classes: list[str] = []
        self.fields: list[tuple[str, str]] = []  # (name, type)
        self.methods: list[MethodFact] = []
        self.known_fields: set[str] = set()

    def visit(self) -> None:
        """Walk the class_specifier and extract all metadata."""
        # Class name
        for child in self.class_node.children:
            if child.type == "type_identifier":
                self.class_name = _node_text(child)
                break

        # Base classes
        for child in self.class_node.children:
            if child.type == "base_class_clause":
                for bc_child in child.children:
                    if bc_child.type == "type_identifier":
                        self.base_classes.append(_node_text(bc_child))

        # Field declaration list
        field_list = None
        for child in self.class_node.children:
            if child.type == "field_declaration_list":
                field_list = child
                break

        if field_list is None:
            return

        # First pass: collect fields
        current_access = "private"  # C++ default
        for child in field_list.children:
            if child.type == "access_specifier":
                current_access = _node_text(child).rstrip(":")
            elif child.type == "field_declaration":
                self._visit_field_declaration(child, current_access)

        self.known_fields = {name for name, _ in self.fields}

        # Second pass: collect methods (need known_fields for bare member detection)
        current_access = "private"
        for child in field_list.children:
            if child.type == "access_specifier":
                current_access = _node_text(child).rstrip(":")
            elif child.type == "function_definition":
                self._visit_method(child, current_access)

    def _visit_field_declaration(self, node: TSNode, access: str) -> None:
        """Extract a member variable declaration."""
        type_parts: list[str] = []
        field_name = ""
        for child in node.children:
            if child.type == "field_identifier":
                field_name = _node_text(child)
            elif child.type in (
                "primitive_type",
                "type_identifier",
                "qualified_identifier",
                "sized_type_specifier",
                "template_type",
            ):
                type_parts.append(_node_text(child))
        if field_name:
            type_str = " ".join(type_parts) if type_parts else "auto"
            self.fields.append((field_name, type_str))

    def _visit_method(self, node: TSNode, access: str) -> None:
        """Extract a method definition and track field accesses in its body."""
        method_name = ""
        params: list[str] = []
        return_type = ""

        # Find the function declarator for name + params
        for child in node.children:
            if child.type == "function_declarator":
                for fc in child.children:
                    if fc.type in ("field_identifier", "identifier"):
                        method_name = _node_text(fc)
                    elif fc.type == "parameter_list":
                        params = self._extract_params(fc)
            elif child.type in (
                "primitive_type",
                "type_identifier",
                "qualified_identifier",
                "sized_type_specifier",
                "template_type",
            ):
                return_type = _node_text(child)

        # Source code
        start_line = node.start_point[0]
        end_line = node.end_point[0] + 1
        method_source = "\n".join(self.source_lines[start_line:end_line])

        # Walk body for field accesses
        reads: set[str] = set()
        writes: set[str] = set()
        calls: set[str] = set()

        body = None
        for child in node.children:
            if child.type == "compound_statement":
                body = child
                break

        # Also check field_initializer_list (constructor)
        init_list = None
        for child in node.children:
            if child.type == "field_initializer_list":
                init_list = child
                break

        if init_list:
            for init_node in _find_children(init_list, "field_initializer"):
                for ic in init_node.children:
                    if ic.type == "field_identifier":
                        writes.add(_node_text(ic))

        if body:
            self._walk_body_accesses(body, reads, writes, calls)

        is_constructor = method_name == self.class_name

        self.methods.append(
            MethodFact(
                name="__init__" if is_constructor else method_name,
                params=params,
                return_type=return_type,
                reads=sorted(reads),
                writes=sorted(writes),
                calls=sorted(calls),
                source_code=method_source,
            )
        )

    def _extract_params(self, param_list: TSNode) -> list[str]:
        """Extract parameter names from a parameter_list node."""
        params: list[str] = []
        for child in param_list.children:
            if child.type == "parameter_declaration":
                for pc in child.children:
                    if pc.type == "identifier":
                        params.append(_node_text(pc))
        return params

    def _walk_body_accesses(
        self,
        node: TSNode,
        reads: set[str],
        writes: set[str],
        calls: set[str],
    ) -> None:
        """Recursively walk a method body for this->field accesses and bare member accesses."""
        # Check for assignment targets (writes)
        if node.type in ("assignment_expression",):
            lhs = node.children[0] if node.children else None
            if lhs is not None:
                field = self._extract_member_field(lhs)
                if field:
                    writes.add(field)
                # Walk RHS for reads
                for child in node.children[1:]:
                    self._walk_body_accesses(child, reads, writes, calls)
                return

        # Check for field_expression reads: this->field or bare member
        if node.type == "field_expression":
            field = self._extract_member_field(node)
            if field:
                reads.add(field)
            # Still recurse into children for nested expressions
            for child in node.children:
                self._walk_body_accesses(child, reads, writes, calls)
            return

        # Check for bare member access used as identifier in expressions
        if node.type == "identifier":
            name = _node_text(node)
            if name in self.known_fields and name != "this":
                # Check parent context to determine read vs write
                # (writes are handled at assignment_expression level)
                reads.add(name)
            return

        # Check for this->method() calls
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            if func_node and func_node.type == "field_expression":
                field = self._extract_member_field(func_node)
                if field:
                    calls.add(field)
                    # Don't add to reads — it's a call, not a field read
                    for child in node.children[1:]:
                        self._walk_body_accesses(child, reads, writes, calls)
                    return

        # Recurse
        for child in node.children:
            self._walk_body_accesses(child, reads, writes, calls)

    def _extract_member_field(self, node: TSNode) -> str | None:
        """Extract the field name from this->field or a bare known field."""
        if node.type == "field_expression":
            children = node.children
            if len(children) >= 3:
                obj = children[0]
                field_node = children[-1]
                if field_node.type == "field_identifier":
                    field_name = _node_text(field_node)
                    # this->field
                    if obj.type == "this":
                        return field_name
                    # Could also be obj.field for known patterns
            return None
        if node.type == "identifier":
            name = _node_text(node)
            if name in self.known_fields:
                return name
        return None


# ---------------------------------------------------------------------------
# Julia struct/function visitors
# ---------------------------------------------------------------------------


# Known Julia types that form Cartesian product components in Bayesian
# samplers (e.g., AdvancedHMC.jl composes Metric × Integrator × TrajectorySampler).
_JULIA_CARTESIAN_PRODUCT_TYPES: frozenset[str] = frozenset(
    {
        "Metric",
        "AbstractMetric",
        "DenseEuclideanMetric",
        "DiagEuclideanMetric",
        "UnitEuclideanMetric",
        "Integrator",
        "AbstractIntegrator",
        "Leapfrog",
        "JitteredLeapfrog",
        "TemperedLeapfrog",
        "TrajectorySampler",
        "AbstractTrajectorySampler",
        "MultinomialTS",
        "SliceTS",
        "AbstractAdaptor",
        "StanHMCAdaptor",
        "NesterovDualAveraging",
        "Hamiltonian",
        "AbstractHamiltonian",
        "AbstractMCMCKernel",
        "HMCKernel",
    }
)


class _JuliaStructVisitor:
    """Extract struct metadata from a ``struct_definition`` tree-sitter node."""

    def __init__(self, struct_node: TSNode) -> None:
        self.struct_node = struct_node
        self.struct_name = ""
        self.type_params: list[str] = []
        self.fields: list[tuple[str, str]] = []  # (name, type)
        self.cartesian_product_fields: list[list[str]] = []

    def visit(self) -> None:
        for child in self.struct_node.children:
            if child.type == "type_head":
                self._visit_type_head(child)
            elif child.type == "identifier":
                # Simple struct without type params
                if not self.struct_name:
                    self.struct_name = _node_text(child)
            elif child.type == "typed_expression":
                self._visit_field(child)

        # After collecting all fields, detect Cartesian product groups
        self._detect_cartesian_products()

    def _visit_type_head(self, node: TSNode) -> None:
        for child in node.children:
            if child.type == "parametrized_type_expression":
                for pc in child.children:
                    if pc.type == "identifier":
                        self.struct_name = _node_text(pc)
                    elif pc.type == "curly_expression":
                        for ce in pc.children:
                            if ce.type == "identifier":
                                self.type_params.append(_node_text(ce))
            elif child.type == "identifier":
                if not self.struct_name:
                    self.struct_name = _node_text(child)

    def _visit_field(self, node: TSNode) -> None:
        """Extract a typed field like ``dur::T``."""
        parts = [c for c in node.children if c.type == "identifier"]
        if len(parts) >= 2:
            field_name = _node_text(parts[0])
            field_type = _node_text(parts[1])
            self.fields.append((field_name, field_type))
        elif len(parts) == 1:
            self.fields.append((_node_text(parts[0]), "Any"))

    def _detect_cartesian_products(self) -> None:
        """Detect groups of fields whose types are known Cartesian product components.

        For example, an AdvancedHMC.jl sampler struct with fields
        ``metric::M``, ``integrator::I``, ``ts::S`` where M/I/S are bounded
        by abstract types in _JULIA_CARTESIAN_PRODUCT_TYPES — these should
        be emitted as distinct swappable subgraph inputs.
        """
        # A field is a product component if its type (or the type param it
        # binds to) matches a known abstract Bayesian sampler component.
        # We check both the concrete type annotation and the type parameter.
        product_group: list[str] = []
        for field_name, field_type in self.fields:
            if field_type in _JULIA_CARTESIAN_PRODUCT_TYPES:
                product_group.append(field_name)
            elif field_type in self.type_params:
                # Type parameter — could be bounded by a product type.
                # Heuristic: field name hints (metric, integrator, sampler, etc.)
                hint_lower = field_name.lower()
                if any(
                    kw in hint_lower
                    for kw in (
                        "metric",
                        "integrator",
                        "sampler",
                        "adaptor",
                        "hamiltonian",
                        "kernel",
                        "trajectory",
                    )
                ):
                    product_group.append(field_name)
        if len(product_group) >= 2:
            self.cartesian_product_fields.append(product_group)


class _HaskellFunctionVisitor:
    """Extract metadata from a Haskell function definition."""

    def __init__(self, func_node: TSNode, source_code: str):
        self.func_node = func_node
        self.source_code = source_code
        self.func_name = ""
        self.params: list[str] = []
        self.return_type = ""
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.is_oracle = False

    def visit(self):
        """Parse Haskell AST node to extract properties."""
        params_set = set()
        for child in self.func_node.children:
            if child.type == "variable":
                self.func_name = _node_text(child)
            elif child.type in ("exp", "body"):
                self._visit_expression(child)
            elif child.type in ("match", "patterns"):
                for pat in _find_descendants(child, "variable"):
                    params_set.add(_node_text(pat))
        
        self.params = sorted(list(params_set))

        # Check for @oracle / @register_atom comments in source segment OR preceding siblings
        src = _node_text(self.func_node)
        
        # Look at preceding siblings for comments
        curr = self.func_node.prev_sibling
        comment_text = ""
        while curr and curr.type in ("comment", "signature"):
            comment_text += _node_text(curr)
            curr = curr.prev_sibling

        if "@oracle" in src or "@oracle" in comment_text or "@register_atom" in src or "@register_atom" in comment_text:
            self.is_oracle = True

    def _visit_expression(self, node: TSNode):
        """Heuristically find reads/writes from expression subtree."""
        # Very simple: all variables in body are potential reads
        for var in _find_descendants(node, "variable"):
            name = _node_text(var)
            if name not in self.params:
                self.reads.add(name)



class _JuliaFunctionVisitor:
    """Extract function metadata and associate methods with structs."""

    def __init__(
        self,
        func_node: TSNode,
        source_code: str,
        struct_names: set[str],
        struct_fields: dict[str, set[str]],
    ) -> None:
        self.func_node = func_node
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.struct_names = struct_names
        self.struct_fields = struct_fields

        self.func_name = ""
        self.params: list[str] = []
        self.param_types: dict[str, str] = {}
        self.return_type = ""
        self.associated_struct: str | None = None
        self.self_param_name: str | None = None
        self.reads: set[str] = set()
        self.writes: set[str] = set()

    def visit(self) -> None:
        # Extract signature
        for child in self.func_node.children:
            if child.type == "signature":
                self._visit_signature(child)

        # Walk body for field accesses
        for child in self.func_node.children:
            if child.type not in ("function", "signature", "end"):
                self._walk_body(child)

    def _visit_signature(self, sig_node: TSNode) -> None:
        """Parse function signature for name, params, return type."""
        # The signature can be:
        #   call_expression (no return type) or
        #   typed_expression wrapping call_expression (with return type)
        #   where_expression wrapping call_expression with type bounds
        for child in sig_node.children:
            if child.type == "call_expression":
                self._visit_call_sig(child)
            elif child.type == "typed_expression":
                self._visit_typed_sig(child)
            elif child.type == "where_expression":
                for wc in child.children:
                    if wc.type == "call_expression":
                        self._visit_call_sig(wc)

    def _visit_typed_sig(self, node: TSNode) -> None:
        """Handle ``func_name(params)::ReturnType``."""
        children = [c for c in node.children if c.type != "::"]
        for child in children:
            if child.type == "call_expression":
                self._visit_call_sig(child)
            elif child.type == "identifier":
                self.return_type = _node_text(child)

    def _visit_call_sig(self, node: TSNode) -> None:
        """Handle ``func_name(params)``."""
        for child in node.children:
            if child.type == "identifier":
                self.func_name = _node_text(child)
            elif child.type == "field_expression":
                id_children = [c for c in child.children if c.type == "identifier"]
                if id_children:
                    # Module-qualified dispatch: AbstractMCMC.step -> step
                    self.func_name = _node_text(id_children[-1])
            elif child.type == "argument_list":
                self._visit_arg_list(child)

    def _visit_arg_list(self, node: TSNode) -> None:
        """Extract parameters and detect struct association."""
        first_param = True
        for child in node.children:
            if child.type == "typed_expression":
                typed_text = _node_text(child)
                if "::" in typed_text:
                    lhs, rhs = typed_text.split("::", 1)
                    param_name = lhs.strip()
                    param_type = rhs.strip()
                else:
                    parts = [c for c in child.children if c.type == "identifier"]
                    if parts:
                        param_name = _node_text(parts[0])
                        param_type = _node_text(parts[1]) if len(parts) >= 2 else ""
                    else:
                        param_name = ""
                        param_type = ""

                if param_name:
                    self.params.append(param_name)
                    if param_type:
                        self.param_types[param_name] = param_type
                    normalized_type = _normalize_type_name(param_type)
                    # Check if first param is typed as a known struct
                    if first_param and normalized_type in self.struct_names:
                        self.associated_struct = normalized_type
                        self.self_param_name = param_name
                first_param = False
            elif child.type == "identifier":
                self.params.append(_node_text(child))
                first_param = False

    def _walk_body(self, node: TSNode) -> None:
        """Walk function body tracking field accesses on the struct parameter."""
        if node.type == "field_expression":
            self._check_field_access(node)

        for child in node.children:
            self._walk_body(child)

    def _check_field_access(self, node: TSNode) -> None:
        """Check if a field_expression accesses the struct parameter's fields."""
        if self.self_param_name is None:
            return

        children = [c for c in node.children if c.type not in (".",)]
        if len(children) >= 2:
            obj_node = children[0]
            field_node = children[1]
            if (
                obj_node.type == "identifier"
                and _node_text(obj_node) == self.self_param_name
                and field_node.type == "identifier"
            ):
                field_name = _node_text(field_node)
                # Check if this is an assignment target
                parent = node.parent
                if parent and parent.type in (
                    "assignment_expression",
                    "assignment",
                ):
                    # LHS of assignment → write
                    lhs = parent.children[0] if parent.children else None
                    if lhs == node:
                        self.writes.add(field_name)
                        return
                self.reads.add(field_name)


# ---------------------------------------------------------------------------
# C++ procedural extraction
# ---------------------------------------------------------------------------

# Known kthohr/mcmc algorithm entry points that accept a kernel function pointer.
# Pattern: mcmc::algo(initial_vals, kernel_func, ...) or algo(init, kernel_func, settings)
_MCMC_ALGO_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "rwmh",
        "mala",
        "hmc",
        "rmhmc",
        "aees",
        "de",
        "nuts",
        "hmc_int",
        "mala_int",
        # Qualified names: mcmc::algo
        "mcmc::rwmh",
        "mcmc::mala",
        "mcmc::hmc",
        "mcmc::rmhmc",
        "mcmc::aees",
        "mcmc::de",
        "mcmc::nuts",
    }
)


def _scan_cpp_oracle_edges(root: TSNode, source_code: str) -> list[OracleEdge]:
    """Scan C++ source for kthohr/mcmc-style functional API calls.

    Looks for patterns like:
        mcmc::hmc(initial_vals, log_target, settings)
        algo(init_vals, kernel_func_ptr, data, settings)

    The second argument (kernel function pointer) is extracted as an
    explicit oracle dependency.
    """
    oracle_edges: list[OracleEdge] = []
    call_nodes = _find_descendants(root, "call_expression")

    for call_node in call_nodes:
        children = call_node.children
        if len(children) < 2:
            continue

        func_node = children[0]
        func_name = _node_text(func_node)

        # Normalize: strip mcmc:: prefix for matching
        normalized = func_name.lower().replace(" ", "")
        # Check if this is a known MCMC algorithm entry point
        is_mcmc_call = (
            normalized in _MCMC_ALGO_IDENTIFIERS or func_name in _MCMC_ALGO_IDENTIFIERS
        )
        if not is_mcmc_call:
            continue

        # Find the argument list
        arg_list = None
        for child in children:
            if child.type == "argument_list":
                arg_list = child
                break

        if arg_list is None:
            continue

        # Extract arguments (skip commas and parens)
        args = [c for c in arg_list.children if c.type not in (",", "(", ")")]

        # The kernel function pointer is typically the 2nd argument
        if len(args) >= 2:
            oracle_ref = _node_text(args[1])
            oracle_edges.append(
                OracleEdge(
                    caller=func_name,
                    oracle_ref=oracle_ref,
                    call_site=_node_text(call_node),
                )
            )

    return oracle_edges


def _extract_cpp_functions(root: TSNode, source_code: str) -> list[MethodFact]:
    """Extract top-level function definitions from C++ source."""
    source_lines = source_code.splitlines()
    methods: list[MethodFact] = []

    for child in root.children:
        if child.type == "function_definition":
            func_name = ""
            params: list[str] = []
            return_type = ""

            for fc in child.children:
                if fc.type == "function_declarator":
                    for fcc in fc.children:
                        if fcc.type in ("identifier", "field_identifier"):
                            func_name = _node_text(fcc)
                        elif fcc.type == "parameter_list":
                            for pc in fcc.children:
                                if pc.type == "parameter_declaration":
                                    for pcc in pc.children:
                                        if pcc.type == "identifier":
                                            params.append(_node_text(pcc))
                elif fc.type in (
                    "primitive_type",
                    "type_identifier",
                    "qualified_identifier",
                ):
                    return_type = _node_text(fc)

            start = child.start_point[0]
            end = child.end_point[0] + 1
            source = "\n".join(source_lines[start:end])

            if func_name:
                methods.append(
                    MethodFact(
                        name=func_name,
                        params=params,
                        return_type=return_type,
                        source_code=source,
                    )
                )

    return methods


# ---------------------------------------------------------------------------
# Julia: Bijectors.jl detection
# ---------------------------------------------------------------------------


def _scan_julia_bijectors(source_code: str) -> bool:
    """Scan Julia source for Bijectors.jl usages indicating constrained variables.

    When constrained variables are present, the model requires a
    log-determinant Jacobian correction for correct posterior inference.
    """
    # Direct Bijectors.jl API calls and types
    bijector_patterns = [
        r"\bBijectors\b",
        r"\bbijector\s*\(",
        r"\binverse\s*\(\s*\w*[Bb]ij",
        r"\blogabsdetjac\b",
        r"\bwith_logabsdet_jacobian\b",
        r"\bTransformed\s*\(",
        r"\btransformed\s*\(",
        # Common bijector constructors
        r"\b(?:Logit|Log|Exp|Softplus|OrderedBijector|Stacked)\s*\(",
    ]
    for pat in bijector_patterns:
        if re.search(pat, source_code):
            return True
    return False


# ---------------------------------------------------------------------------
# TreeSitterExtractor — public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rust visitors
# ---------------------------------------------------------------------------


class _RustStructVisitor:
    def __init__(self, node: TSNode) -> None:
        self.node = node
        self.struct_name = ""
        self.fields: list[tuple[str, str]] = []  # (name, type_str)
        self.static_shape: dict[str, str] = {}  # e.g. {"N": "6", "M": "3"}

    def visit(self) -> None:
        for child in self.node.children:
            if child.type == "type_identifier":
                self.struct_name = _node_text(child)
            elif child.type == "field_declaration_list":
                for field in _find_children(child, "field_declaration"):
                    fname = ""
                    ftype = ""
                    for fc in field.children:
                        if fc.type == "field_identifier":
                            fname = _node_text(fc)
                        elif fc.type in (
                            "type_identifier",
                            "primitive_type",
                            "generic_type",
                        ):
                            ftype = _node_text(fc)
                    if fname:
                        self.fields.append((fname, ftype or "Any"))
                        # Extract nalgebra static matrix dimensions
                        # e.g. SMatrix<f64, N, M>, SVector<f64, N>
                        if ftype:
                            self._scan_static_shape(ftype)

    def _scan_static_shape(self, type_str: str) -> None:
        """Extract compile-time dimension literals from nalgebra types."""

        # Match SMatrix<scalar, R, C>, SVector<scalar, N>, etc.
        for m in re.finditer(
            r"\b(?:SMatrix|SVector|OMatrix|OVector)\s*<\s*\w+\s*,"
            r"\s*(\w+)\s*(?:,\s*(\w+)\s*)?(?:,\s*\w+\s*)*>",
            type_str,
        ):
            dim1 = m.group(1)
            dim2 = m.group(2)
            # Only store if they look like numeric literals or named constants
            if dim1 and dim1 not in ("f32", "f64", "i32", "i64"):
                self.static_shape.setdefault(dim1, dim1)
            if dim2 and dim2 not in ("f32", "f64", "i32", "i64"):
                self.static_shape.setdefault(dim2, dim2)


# Trait bounds that indicate oracle (stateless log-density/gradient) implementations
_ORACLE_TRAIT_BOUNDS: frozenset[str] = frozenset(
    {
        "BatchedGradientTarget",
        "GradientTarget",
        "LogDensityTarget",
        "LogDensity",
        "DiffFn",
    }
)

# Trait bounds that indicate conjugate/analytical update implementations
_CONJUGATE_TRAIT_BOUNDS: frozenset[str] = frozenset(
    {
        "Estimator",
        "ConjugateUpdate",
        "ConjugatePrior",
        "AnalyticalPosterior",
        "SufficientStatistic",
    }
)


class _RustFunctionVisitor:
    def __init__(self, node: TSNode, source_code: str) -> None:
        self.node = node
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.func_name = ""
        self.params: list[str] = []
        self.return_type = ""
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.is_oracle = False
        self.is_conjugate = False

    def visit(self) -> None:
        for child in self.node.children:
            if child.type == "identifier":
                self.func_name = _node_text(child)
            elif child.type == "parameters":
                for param in _find_children(child, "parameter"):
                    for pc in param.children:
                        if pc.type == "identifier":
                            self.params.append(_node_text(pc))
            elif child.type == "type_identifier":  # Return type
                self.return_type = _node_text(child)

    def detect_trait_bounds(self, impl_node: TSNode) -> None:
        """Scan the parent impl block for trait bounds on the impl or where clause."""
        impl_text = _node_text(impl_node)
        for trait in _ORACLE_TRAIT_BOUNDS:
            if trait in impl_text:
                self.is_oracle = True
                break
        for trait in _CONJUGATE_TRAIT_BOUNDS:
            if trait in impl_text:
                self.is_conjugate = True
                break

    def get_fact(self) -> MethodFact:
        start = self.node.start_point[0]
        end = self.node.end_point[0] + 1
        return MethodFact(
            name=self.func_name,
            params=self.params,
            return_type=self.return_type,
            reads=sorted(self.reads),
            writes=sorted(self.writes),
            source_code="\n".join(self.source_lines[start:end]),
            is_oracle=self.is_oracle,
            is_conjugate=self.is_conjugate,
        )


class TreeSitterExtractor:
    """Extract data-flow graphs from C++ or Julia source via tree-sitter."""

    def __init__(self, language: SourceLanguage) -> None:
        if language == SourceLanguage.PYTHON:
            raise ValueError("Use PythonASTExtractor for Python sources")
        self.language = language
        if language == SourceLanguage.CPP:
            lang_key = "cpp"
        elif language == SourceLanguage.JULIA:
            lang_key = "julia"
        elif language == SourceLanguage.HASKELL:
            lang_key = "haskell"
        else:
            lang_key = "rust"
        self._parser = get_parser(lang_key)
        self._rust_trait_query = None
        self._rust_oracle_call_query = None
        self._julia_dispatch_query = None
        self._julia_oracle_call_query = None

        if language == SourceLanguage.RUST:
            self._rust_trait_query = self._compile_query(_RUST_TRAIT_BOUNDS_QUERY)
            self._rust_oracle_call_query = self._compile_query(_RUST_ORACLE_CALL_QUERY)
        elif language == SourceLanguage.JULIA:
            self._julia_dispatch_query = self._compile_query(
                _JULIA_TYPED_DISPATCH_QUERY
            )
            self._julia_oracle_call_query = self._compile_query(
                _JULIA_ORACLE_CALL_QUERY
            )

    def _compile_query(self, query_src: str):
        """Compile a tree-sitter query. Returns None if unsupported."""
        try:
            return self._parser.language.query(query_src)
        except Exception:
            return None

    @staticmethod
    def _query_captures(root: TSNode, query) -> dict[str, list[TSNode]]:
        """Execute query and return capture-name -> nodes mapping."""
        if query is None:
            return {}
        try:
            # tree-sitter >= 0.25: Query no longer has .captures();
            # use QueryCursor instead.
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            return cursor.captures(root)
        except Exception:
            # Fallback for older tree-sitter versions
            try:
                return query.captures(root)
            except Exception:
                return {}

    def _rust_trait_bounds_for_node(self, node: TSNode) -> set[str]:
        """Extract trait-bound names from a Rust impl/function node."""
        captures = self._query_captures(node, self._rust_trait_query)
        trait_nodes = captures.get("rust.trait_bound", [])
        return {_normalize_trait_name(_node_text(tn)) for tn in trait_nodes}

    def _build_rust_oracle_subgraph(
        self,
        fn_node: TSNode,
        caller_name: str,
        oracle_traits: set[str],
    ) -> tuple[list[OracleEdge], list[DependencyEdge], set[str], set[str]]:
        """Build oracle subgraph dependencies for trait-bound Rust calls."""
        captures = self._query_captures(fn_node, self._rust_oracle_call_query)
        call_nodes = captures.get("rust.oracle_call", [])

        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()
        seen_oracle_edges: set[tuple[str, str]] = set()
        method_state_vars: set[str] = set()
        method_outputs: set[str] = set()

        trait_label = sorted(oracle_traits)[0] if oracle_traits else "OracleInterface"

        for call_node in call_nodes:
            func_expr = call_node.children[0] if call_node.children else None
            if func_expr is None or func_expr.type != "field_expression":
                continue

            method_nodes = [
                c for c in func_expr.children if c.type == "field_identifier"
            ]
            if not method_nodes:
                continue
            oracle_method = _node_text(method_nodes[-1]).strip()
            if oracle_method.lower() not in _ORACLE_CALL_METHOD_HINTS:
                continue

            obj_node = func_expr.children[0] if func_expr.children else None
            if obj_node is None:
                continue
            oracle_obj = _node_text(obj_node).strip()
            if not oracle_obj:
                continue

            oracle_anchor = (
                re.sub(r"[^A-Za-z0-9_]+", "_", oracle_obj).strip("_") or "oracle"
            )
            oracle_node_id = (
                f"oracle_subgraph::{trait_label}::{caller_name}::{oracle_anchor}"
            )

            edge_key = (caller_name, oracle_node_id)
            if edge_key not in seen_oracle_edges:
                seen_oracle_edges.add(edge_key)
                oracle_edges.append(
                    OracleEdge(
                        caller=caller_name,
                        oracle_ref=oracle_node_id,
                        call_site=_node_text(call_node),
                    )
                )

            args_node = next(
                (c for c in call_node.children if c.type == "arguments"), None
            )
            state_vars: list[str] = []
            if args_node is not None:
                obj_tokens = {
                    tok for tok in re.split(r"[^A-Za-z0-9_]+", oracle_obj) if tok
                }
                for ident in _extract_identifiers_from_args(args_node):
                    if ident in obj_tokens:
                        continue
                    if ident == oracle_method:
                        continue
                    state_vars.append(ident)

            for state_var in state_vars:
                method_state_vars.add(state_var)
                _add_edge_if_new(
                    inferred_edges,
                    seen_inferred,
                    DependencyEdge(
                        source_id=f"state:{state_var}",
                        target_id=oracle_node_id,
                        output_name=state_var,
                        input_name=state_var,
                        source_type="StateVar",
                        target_type="Any",
                    ),
                )

            for out_name in _oracle_outputs_for_method(oracle_method):
                method_outputs.add(out_name)
                out_type = "ndarray" if out_name == "gradient" else "AbstractScalar"
                _add_edge_if_new(
                    inferred_edges,
                    seen_inferred,
                    DependencyEdge(
                        source_id=oracle_node_id,
                        target_id=caller_name,
                        output_name=out_name,
                        input_name=out_name,
                        source_type=out_type,
                        target_type=out_type,
                    ),
                )

        return oracle_edges, inferred_edges, method_state_vars, method_outputs

    def _julia_dispatch_function_keys(
        self, root: TSNode
    ) -> set[tuple[int, int, int, int]]:
        """Return function-definition nodes using typed or where-dispatched signatures."""
        captures = self._query_captures(root, self._julia_dispatch_query)
        fn_nodes = captures.get("julia.dispatch_fn", [])
        return {_node_key(node) for node in fn_nodes}

    def _is_julia_oracle_interface(
        self,
        source_node: TSNode,
        fv: _JuliaFunctionVisitor,
        dispatch_fn_keys: set[tuple[int, int, int, int]],
    ) -> bool:
        """Determine whether a Julia function acts as a probabilistic oracle interface."""
        if _node_key(source_node) not in dispatch_fn_keys:
            return False

        type_hints = [t.lower() for t in fv.param_types.values()]
        bound_hints = _node_text(source_node).lower()
        if any(
            "abstractmcmc" in th
            or "abstract" in th
            or "density" in th
            or "logdensityproblems" in th
            for th in type_hints
        ):
            return True
        if "where" in bound_hints and "<:" in bound_hints and "abstract" in bound_hints:
            return True
        return fv.func_name.lower() in {"step", "logdensity", "log_prob", "gradient"}

    def _build_julia_oracle_subgraph(
        self,
        fn_node: TSNode,
        caller_name: str,
    ) -> tuple[list[OracleEdge], list[DependencyEdge], set[str], set[str]]:
        """Build oracle subgraph dependencies for Julia type-dispatched interfaces."""
        captures = self._query_captures(fn_node, self._julia_oracle_call_query)
        call_nodes = captures.get("julia.oracle_call", [])

        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()
        seen_oracle_edges: set[tuple[str, str]] = set()
        method_state_vars: set[str] = set()
        method_outputs: set[str] = set()

        for call_node in call_nodes:
            children = [c for c in call_node.children if c.type not in ("(", ")", ",")]
            if len(children) < 2:
                continue

            func_expr = children[0]
            args_node = next((c for c in children if c.type == "argument_list"), None)
            if args_node is None:
                continue

            oracle_obj = ""
            oracle_method = ""

            if func_expr.type == "field_expression":
                id_parts = [c for c in func_expr.children if c.type == "identifier"]
                if len(id_parts) >= 2:
                    oracle_obj = _node_text(id_parts[0]).strip()
                    oracle_method = _node_text(id_parts[-1]).strip()
            elif func_expr.type == "identifier":
                oracle_method = _node_text(func_expr).strip()
                arg_idents = _extract_identifiers_from_args(args_node)
                if arg_idents:
                    oracle_obj = arg_idents[0]

            if not oracle_obj or not oracle_method:
                continue
            if oracle_method.lower() not in _ORACLE_CALL_METHOD_HINTS:
                continue

            oracle_anchor = (
                re.sub(r"[^A-Za-z0-9_]+", "_", oracle_obj).strip("_") or "oracle"
            )
            oracle_node_id = (
                f"oracle_subgraph::JuliaDispatch::{caller_name}::{oracle_anchor}"
            )

            edge_key = (caller_name, oracle_node_id)
            if edge_key not in seen_oracle_edges:
                seen_oracle_edges.add(edge_key)
                oracle_edges.append(
                    OracleEdge(
                        caller=caller_name,
                        oracle_ref=oracle_node_id,
                        call_site=_node_text(call_node),
                    )
                )

            for ident in _extract_identifiers_from_args(args_node):
                if ident in {oracle_obj, oracle_method}:
                    continue
                method_state_vars.add(ident)
                _add_edge_if_new(
                    inferred_edges,
                    seen_inferred,
                    DependencyEdge(
                        source_id=f"state:{ident}",
                        target_id=oracle_node_id,
                        output_name=ident,
                        input_name=ident,
                        source_type="StateVar",
                        target_type="Any",
                    ),
                )

            for out_name in _oracle_outputs_for_method(oracle_method):
                method_outputs.add(out_name)
                out_type = "ndarray" if out_name == "gradient" else "AbstractScalar"
                _add_edge_if_new(
                    inferred_edges,
                    seen_inferred,
                    DependencyEdge(
                        source_id=oracle_node_id,
                        target_id=caller_name,
                        output_name=out_name,
                        input_name=out_name,
                        source_type=out_type,
                        target_type=out_type,
                    ),
                )

        return oracle_edges, inferred_edges, method_state_vars, method_outputs

    async def extract_function(
        self, source_path: str, function_name: str
    ) -> RawDataFlowGraph:
        """Extract a function-level data-flow graph for non-Python languages.

        Mirrors ``extract_function_data_flow()`` from extractor.py:
        collect all functions, build a call graph, compute the transitive
        closure from *function_name*, and return a :class:`RawDataFlowGraph`.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        source_code = path.read_text()
        source_lines = source_code.splitlines()
        tree = self._parser.parse(source_code.encode("utf-8"))
        root = tree.root_node

        # 1. Collect all function definitions in the file
        func_nodes = _collect_all_function_nodes(root, self.language)

        if function_name not in func_nodes:
            raise ValueError(
                f"Function '{function_name}' not found in {source_path}"
            )

        # 2. Build file-level call graph
        file_call_graph = _build_file_call_graph_ts(func_nodes)

        # 3. Compute transitive closure from target
        reachable = _transitive_closure(function_name, file_call_graph)

        # 4. Build MethodFact for each reachable function
        methods: list[MethodFact] = []
        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()

        for fname in sorted(reachable):
            fn_node = func_nodes.get(fname)
            if fn_node is None:
                continue
            mf, sub_oracle, sub_inferred = self._build_method_fact_for_function(
                fn_node, source_code, source_lines, fname
            )
            methods.append(mf)
            oracle_edges.extend(sub_oracle)
            for edge in sub_inferred:
                _add_edge_if_new(inferred_edges, seen_inferred, edge)

        # 5. Build internal_call_graph (edges between reachable functions)
        internal_call_graph: dict[str, list[str]] = {}
        for fname in reachable:
            callees = sorted(file_call_graph.get(fname, set()) & reachable)
            if callees:
                internal_call_graph[fname] = callees

        # 6. Build all_attributes from method reads/writes
        all_attributes: dict[str, list[str]] = {}
        for mf in methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        # Language-specific metadata
        requires_logdet_jacobian = False
        if self.language == SourceLanguage.JULIA:
            requires_logdet_jacobian = _scan_julia_bijectors(source_code)

        return RawDataFlowGraph(
            class_name=function_name,
            source_code=source_code,
            methods=methods,
            all_attributes=all_attributes,
            internal_call_graph=internal_call_graph,
            inferred_edges=inferred_edges,
            oracle_edges=oracle_edges,
            source_language=self.language.value,
            requires_logdet_jacobian=requires_logdet_jacobian,
        )

    def _extract_haskell_functions_procedural(
        self, root: TSNode, source_code: str
    ) -> tuple[list[MethodFact], list[OracleEdge], list[DependencyEdge]]:
        """Extract all functions from a Haskell module."""
        methods: list[MethodFact] = []
        source_lines = source_code.splitlines()

        for fn_node in _find_descendants(root, "function"):
            is_def = False
            for child in fn_node.children:
                if child.type in ("exp", "body", "match"):
                    is_def = True
            if not is_def:
                continue

            fv = _HaskellFunctionVisitor(fn_node, source_code)
            fv.visit()
            if not fv.func_name:
                continue

            start = fn_node.start_point[0]
            end = fn_node.end_point[0] + 1
            source_segment = "\n".join(source_lines[start:end])

            callees = sorted(_extract_callees_from_body(fn_node))

            methods.append(
                MethodFact(
                    name=fv.func_name,
                    params=fv.params,
                    return_type=fv.return_type,
                    reads=sorted(fv.reads),
                    writes=sorted(fv.writes),
                    calls=callees,
                    source_code=source_segment,
                    is_oracle=fv.is_oracle,
                )
            )

        return methods, [], []

    def _extract_haskell_class(
        self, root: TSNode, source_code: str, class_name: str
    ) -> RawDataFlowGraph:
        """Haskell has no classes; placeholder for protocol parity."""
        return RawDataFlowGraph(
            class_name=class_name,
            source_code=source_code,
            methods=[],
            all_attributes={},
            source_language=self.language.value,
        )

    # ------------------------------------------------------------------
    # Private: build a MethodFact for a single function node
    # ------------------------------------------------------------------

    def _build_method_fact_for_function(
        self,
        fn_node: TSNode,
        source_code: str,
        source_lines: list[str],
        func_name: str,
    ) -> tuple[MethodFact, list[OracleEdge], list[DependencyEdge]]:
        """Build a MethodFact for a single function, with oracle detection."""
        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []

        start = fn_node.start_point[0]
        end = fn_node.end_point[0] + 1
        source = "\n".join(source_lines[start:end])

        # Track calls to other functions in the file
        callees = sorted(_extract_callees_from_body(fn_node))

        if self.language == SourceLanguage.RUST:
            fv = _RustFunctionVisitor(fn_node, source_code)
            fv.visit()

            parent = fn_node.parent
            if parent and parent.type == "impl_item":
                trait_bounds = self._rust_trait_bounds_for_node(parent)
            else:
                trait_bounds = self._rust_trait_bounds_for_node(fn_node)

            oracle_traits = {t for t in trait_bounds if t in _ORACLE_TRAIT_BOUNDS}
            conjugate_traits = {
                t for t in trait_bounds if t in _CONJUGATE_TRAIT_BOUNDS
            }
            if oracle_traits:
                fv.is_oracle = True
            if conjugate_traits:
                fv.is_conjugate = True

            fact = fv.get_fact()
            fact = fact.model_copy(update={"calls": callees})

            if fact.is_oracle:
                sub_o, sub_i, sv, oo = self._build_rust_oracle_subgraph(
                    fn_node, fact.name, oracle_traits
                )
                oracle_edges.extend(sub_o)
                inferred_edges.extend(sub_i)
                fact = fact.model_copy(
                    update={
                        "reads": sorted(set(fact.reads) | sv),
                        "writes": sorted(set(fact.writes) | oo),
                    }
                )

            return fact, oracle_edges, inferred_edges

        elif self.language == SourceLanguage.JULIA:
            struct_names: set[str] = set()
            fv = _JuliaFunctionVisitor(fn_node, source_code, struct_names, {})
            fv.visit()

            dispatch_fn_keys = self._julia_dispatch_function_keys(
                fn_node.parent or fn_node
            )
            is_oracle = self._is_julia_oracle_interface(
                fn_node, fv, dispatch_fn_keys
            )

            state_vars: set[str] = set()
            oracle_outputs: set[str] = set()
            if is_oracle:
                sub_o, sub_i, state_vars, oracle_outputs = (
                    self._build_julia_oracle_subgraph(fn_node, fv.func_name)
                )
                oracle_edges.extend(sub_o)
                inferred_edges.extend(sub_i)

            fact = MethodFact(
                name=fv.func_name or func_name,
                params=fv.params,
                return_type=fv.return_type,
                reads=sorted(set(fv.reads) | state_vars),
                writes=sorted(set(fv.writes) | oracle_outputs),
                calls=callees,
                source_code=source,
                is_oracle=is_oracle,
            )
            return fact, oracle_edges, inferred_edges

        elif self.language == SourceLanguage.HASKELL:
            fv = _HaskellFunctionVisitor(fn_node, source_code)
            fv.visit()
            fact = MethodFact(
                name=fv.func_name or func_name,
                params=fv.params,
                return_type=fv.return_type,
                reads=sorted(fv.reads),
                writes=sorted(fv.writes),
                calls=callees,
                source_code=source,
                is_oracle=fv.is_oracle,
            )
            return fact, oracle_edges, inferred_edges

        else:  # C++
            extracted_name = ""
            params: list[str] = []
            return_type = ""

            for fc in fn_node.children:
                if fc.type == "function_declarator":
                    for fcc in fc.children:
                        if fcc.type in ("identifier", "field_identifier"):
                            extracted_name = _node_text(fcc)
                        elif fcc.type == "parameter_list":
                            for pc in fcc.children:
                                if pc.type == "parameter_declaration":
                                    for pcc in pc.children:
                                        if pcc.type == "identifier":
                                            params.append(_node_text(pcc))
                elif fc.type in (
                    "primitive_type",
                    "type_identifier",
                    "qualified_identifier",
                ):
                    return_type = _node_text(fc)

            oracle_edges = _scan_cpp_oracle_edges(fn_node, source_code)

            fact = MethodFact(
                name=extracted_name or func_name,
                params=params,
                return_type=return_type,
                calls=callees,
                source_code=source,
            )
            return fact, oracle_edges, inferred_edges

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        """Extract a class (C++) or struct (Julia) data-flow graph."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        source_code = path.read_text()
        tree = self._parser.parse(source_code.encode("utf-8"))
        root = tree.root_node

        if self.language == SourceLanguage.CPP:
            return self._extract_cpp_class(root, source_code, class_name)
        elif self.language == SourceLanguage.JULIA:
            return self._extract_julia_struct(root, source_code, class_name)
        elif self.language == SourceLanguage.HASKELL:
            return self._extract_haskell_class(root, source_code, class_name)
        else:
            return self._extract_rust_struct(root, source_code, class_name)

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        """Extract procedural/top-level functions."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        source_code = path.read_text()
        name = pipeline_name or path.stem
        tree = self._parser.parse(source_code.encode("utf-8"))
        root = tree.root_node

        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        requires_logdet_jacobian = False

        if self.language == SourceLanguage.CPP:
            methods = _extract_cpp_functions(root, source_code)
            oracle_edges = _scan_cpp_oracle_edges(root, source_code)
        elif self.language == SourceLanguage.JULIA:
            methods, oracle_edges, inferred_edges = (
                self._extract_julia_functions_procedural(root, source_code)
            )
            requires_logdet_jacobian = _scan_julia_bijectors(source_code)
        elif self.language == SourceLanguage.HASKELL:
            methods, oracle_edges, inferred_edges = (
                self._extract_haskell_functions_procedural(root, source_code)
            )
        else:
            methods, oracle_edges, inferred_edges = (
                self._extract_rust_functions_procedural(root, source_code)
            )

        # Build all_attributes from method reads/writes
        all_attributes: dict[str, list[str]] = {}
        for mf in methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        return RawDataFlowGraph(
            class_name=name,
            source_code=source_code,
            methods=methods,
            all_attributes=all_attributes,
            source_language=self.language.value,
            oracle_edges=oracle_edges,
            inferred_edges=inferred_edges,
            requires_logdet_jacobian=requires_logdet_jacobian,
        )

    def _extract_julia_functions_procedural(
        self,
        root: TSNode,
        source_code: str,
    ) -> tuple[list[MethodFact], list[OracleEdge], list[DependencyEdge]]:
        """Extract top-level Julia functions plus Oracle subgraph dependencies."""
        source_lines = source_code.splitlines()

        # Collect struct names for method association filtering
        struct_names: set[str] = set()
        for child in root.children:
            if child.type == "struct_definition":
                v = _JuliaStructVisitor(child)
                v.visit()
                if v.struct_name:
                    struct_names.add(v.struct_name)

        dispatch_fn_keys = self._julia_dispatch_function_keys(root)

        methods: list[MethodFact] = []
        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()

        for child in root.children:
            if child.type != "function_definition":
                continue

            fv = _JuliaFunctionVisitor(child, source_code, struct_names, {})
            fv.visit()

            # Only include free/procedural functions (exclude struct-associated methods)
            if fv.associated_struct is not None or not fv.func_name:
                continue

            is_oracle = self._is_julia_oracle_interface(child, fv, dispatch_fn_keys)
            state_vars: set[str] = set()
            oracle_outputs: set[str] = set()
            if is_oracle:
                (
                    sub_oracle_edges,
                    sub_inferred_edges,
                    state_vars,
                    oracle_outputs,
                ) = self._build_julia_oracle_subgraph(child, fv.func_name)
                oracle_edges.extend(sub_oracle_edges)
                for edge in sub_inferred_edges:
                    _add_edge_if_new(inferred_edges, seen_inferred, edge)

            start = child.start_point[0]
            end = child.end_point[0] + 1
            source = "\n".join(source_lines[start:end])
            methods.append(
                MethodFact(
                    name=fv.func_name,
                    params=fv.params,
                    return_type=fv.return_type,
                    reads=sorted(set(fv.reads) | state_vars),
                    writes=sorted(set(fv.writes) | oracle_outputs),
                    source_code=source,
                    is_oracle=is_oracle,
                )
            )

        return methods, oracle_edges, inferred_edges

    # --- C++ extraction ---

    def _extract_cpp_class(
        self, root: TSNode, source_code: str, class_name: str
    ) -> RawDataFlowGraph:
        """Find and extract a C++ class by name."""
        class_nodes = _find_descendants(root, "class_specifier")

        target: TSNode | None = None
        for cn in class_nodes:
            for child in cn.children:
                if child.type == "type_identifier" and _node_text(child) == class_name:
                    target = cn
                    break
            if target:
                break

        if target is None:
            raise ValueError(f"Class '{class_name}' not found in source")

        visitor = _CppClassVisitor(target, source_code)
        visitor.visit()

        # Build all_attributes index
        all_attributes: dict[str, list[str]] = {}
        for mf in visitor.methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        # Init chain from constructor
        init_method = next((m for m in visitor.methods if m.name == "__init__"), None)
        init_chain = list(init_method.writes) if init_method else []

        # Internal call graph
        method_names = {m.name for m in visitor.methods}
        internal_call_graph: dict[str, list[str]] = {}
        for mf in visitor.methods:
            internal_calls = [c for c in mf.calls if c in method_names]
            if internal_calls:
                internal_call_graph[mf.name] = internal_calls

        # Scan for kthohr/mcmc functional oracle edges
        oracle_edges = _scan_cpp_oracle_edges(target, source_code)

        return RawDataFlowGraph(
            class_name=class_name,
            source_code=source_code,
            methods=visitor.methods,
            all_attributes=all_attributes,
            init_chain=init_chain,
            internal_call_graph=internal_call_graph,
            source_language="cpp",
            oracle_edges=oracle_edges,
        )

    # --- Julia extraction ---

    def _extract_julia_struct(
        self, root: TSNode, source_code: str, struct_name: str
    ) -> RawDataFlowGraph:
        """Find and extract a Julia struct by name, associating methods."""
        # Collect all structs
        struct_visitors: list[_JuliaStructVisitor] = []
        for child in root.children:
            if child.type == "struct_definition":
                sv = _JuliaStructVisitor(child)
                sv.visit()
                struct_visitors.append(sv)

        # Find target struct
        target_sv: _JuliaStructVisitor | None = None
        for sv in struct_visitors:
            if sv.struct_name == struct_name:
                target_sv = sv
                break

        if target_sv is None:
            raise ValueError(f"Struct '{struct_name}' not found in source")

        struct_names = {sv.struct_name for sv in struct_visitors}
        struct_fields = {
            sv.struct_name: {f[0] for f in sv.fields} for sv in struct_visitors
        }
        dispatch_fn_keys = self._julia_dispatch_function_keys(root)

        # Find functions associated with this struct
        methods: list[MethodFact] = []
        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()
        source_lines = source_code.splitlines()
        for child in root.children:
            if child.type == "function_definition":
                fv = _JuliaFunctionVisitor(
                    child, source_code, struct_names, struct_fields
                )
                fv.visit()
                if fv.associated_struct == struct_name:
                    start = child.start_point[0]
                    end = child.end_point[0] + 1
                    source = "\n".join(source_lines[start:end])
                    # Params: skip the self-like first param
                    params = fv.params[1:] if fv.params else []
                    is_oracle = self._is_julia_oracle_interface(
                        child, fv, dispatch_fn_keys
                    )
                    sub_oracle_edges: list[OracleEdge] = []
                    sub_inferred_edges: list[DependencyEdge] = []
                    state_vars: set[str] = set()
                    oracle_outputs: set[str] = set()
                    if is_oracle:
                        (
                            sub_oracle_edges,
                            sub_inferred_edges,
                            state_vars,
                            oracle_outputs,
                        ) = self._build_julia_oracle_subgraph(child, fv.func_name)
                        oracle_edges.extend(sub_oracle_edges)
                        for edge in sub_inferred_edges:
                            _add_edge_if_new(inferred_edges, seen_inferred, edge)

                    methods.append(
                        MethodFact(
                            name=fv.func_name,
                            params=params,
                            return_type=fv.return_type,
                            reads=sorted(set(fv.reads) | state_vars),
                            writes=sorted(set(fv.writes) | oracle_outputs),
                            source_code=source,
                            is_oracle=is_oracle,
                        )
                    )

        # Build all_attributes
        all_attributes: dict[str, list[str]] = {}
        for mf in methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        # Detect Bijectors.jl usage (constrained variables requiring log-det Jacobian)
        requires_logdet_jacobian = _scan_julia_bijectors(source_code)

        return RawDataFlowGraph(
            class_name=struct_name,
            source_code=source_code,
            methods=methods,
            all_attributes=all_attributes,
            source_language="julia",
            cartesian_product_fields=target_sv.cartesian_product_fields,
            requires_logdet_jacobian=requires_logdet_jacobian,
            oracle_edges=oracle_edges,
            inferred_edges=inferred_edges,
        )

    def _extract_rust_struct(
        self, root: TSNode, source_code: str, name: str
    ) -> RawDataFlowGraph:
        struct_nodes = _find_descendants(root, "struct_item")
        target = None
        for sn in struct_nodes:
            v = _RustStructVisitor(sn)
            v.visit()
            if v.struct_name == name:
                target = v
                break

        if not target:
            raise ValueError(f"Rust struct {name} not found")

        methods: list[MethodFact] = []
        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()
        impl_nodes = _find_descendants(root, "impl_item")
        for im in impl_nodes:
            is_target = False
            for child in im.children:
                if child.type == "type_identifier" and _node_text(child) == name:
                    is_target = True
                    break
                if child.type == "generic_type":
                    for tid in _find_descendants(child, "type_identifier"):
                        if _node_text(tid) == name:
                            is_target = True
                            break
                    if is_target:
                        break

            if is_target:
                trait_bounds = self._rust_trait_bounds_for_node(im)
                oracle_traits = {t for t in trait_bounds if t in _ORACLE_TRAIT_BOUNDS}
                conjugate_traits = {
                    t for t in trait_bounds if t in _CONJUGATE_TRAIT_BOUNDS
                }
                for fn_node in _find_descendants(im, "function_item"):
                    fv = _RustFunctionVisitor(fn_node, source_code)
                    fv.visit()
                    # Detect trait bounds from the enclosing impl block
                    fv.detect_trait_bounds(im)
                    if oracle_traits:
                        fv.is_oracle = True
                    if conjugate_traits:
                        fv.is_conjugate = True
                    fact = fv.get_fact()

                    if fact.is_oracle:
                        (
                            sub_oracle_edges,
                            sub_inferred_edges,
                            state_vars,
                            oracle_outputs,
                        ) = self._build_rust_oracle_subgraph(
                            fn_node, fact.name, oracle_traits
                        )
                        oracle_edges.extend(sub_oracle_edges)
                        for edge in sub_inferred_edges:
                            _add_edge_if_new(inferred_edges, seen_inferred, edge)
                        fact = fact.model_copy(
                            update={
                                "reads": sorted(set(fact.reads) | state_vars),
                                "writes": sorted(set(fact.writes) | oracle_outputs),
                            }
                        )

                    methods.append(fact)

        all_attributes: dict[str, list[str]] = {}
        for mf in methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        return RawDataFlowGraph(
            class_name=name,
            source_code=source_code,
            methods=methods,
            all_attributes=all_attributes,
            source_language="rust",
            static_shape=target.static_shape,
            oracle_edges=oracle_edges,
            inferred_edges=inferred_edges,
        )

    def _extract_rust_functions_procedural(
        self, root: TSNode, source_code: str
    ) -> tuple[list[MethodFact], list[OracleEdge], list[DependencyEdge]]:
        func_nodes = _find_descendants(root, "function_item")
        methods: list[MethodFact] = []
        oracle_edges: list[OracleEdge] = []
        inferred_edges: list[DependencyEdge] = []
        seen_inferred: set[tuple[str, str, str, str, str, str]] = set()
        for fn in func_nodes:
            if fn.parent and fn.parent.type == "source_file":
                fv = _RustFunctionVisitor(fn, source_code)
                fv.visit()
                fn_traits = self._rust_trait_bounds_for_node(fn)
                oracle_traits = {t for t in fn_traits if t in _ORACLE_TRAIT_BOUNDS}
                conjugate_traits = {
                    t for t in fn_traits if t in _CONJUGATE_TRAIT_BOUNDS
                }
                if oracle_traits:
                    fv.is_oracle = True
                if conjugate_traits:
                    fv.is_conjugate = True
                fact = fv.get_fact()

                if fact.is_oracle:
                    (
                        sub_oracle_edges,
                        sub_inferred_edges,
                        state_vars,
                        oracle_outputs,
                    ) = self._build_rust_oracle_subgraph(fn, fact.name, oracle_traits)
                    oracle_edges.extend(sub_oracle_edges)
                    for edge in sub_inferred_edges:
                        _add_edge_if_new(inferred_edges, seen_inferred, edge)
                    fact = fact.model_copy(
                        update={
                            "reads": sorted(set(fact.reads) | state_vars),
                            "writes": sorted(set(fact.writes) | oracle_outputs),
                        }
                    )

                methods.append(fact)
        return methods, oracle_edges, inferred_edges
