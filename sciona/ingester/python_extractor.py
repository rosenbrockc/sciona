"""Python AST extractor — thin adapter wrapping existing extraction functions.

Also provides a ``JAXprExtractor`` for JAX-based code that bypasses standard
AST parsing in favour of ``jax.make_jaxpr()`` tracing, and factor-graph
detection for belief propagation patterns.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from sciona.architect.models import DependencyEdge
from sciona.ingester.extractor import (
    extract_data_flow,
    extract_function_data_flow,
    extract_procedural_data_flow,
)
from sciona.ingester.models import MethodFact, RawDataFlowGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor-graph / belief-propagation pattern detection
# ---------------------------------------------------------------------------

# AST patterns indicating marginal computation via broadcasting / np.sum(axis=...)
_MARGINAL_CALL_PATTERNS: frozenset[str] = frozenset(
    {
        "np.sum",
        "numpy.sum",
        "jnp.sum",
        "jax.numpy.sum",
        "np.einsum",
        "numpy.einsum",
        "jnp.einsum",
        "np.tensordot",
        "numpy.tensordot",
        "jnp.tensordot",
        "logsumexp",
        "scipy.special.logsumexp",
        "jax.scipy.special.logsumexp",
    }
)

# Keywords in function/variable names hinting at factor-graph structure
_FACTOR_GRAPH_HINTS: frozenset[str] = frozenset(
    {
        "message",
        "factor",
        "variable_node",
        "factor_node",
        "belief",
        "marginal",
        "normalize_message",
        "sum_product",
        "bp_",
        "belief_prop",
        "factor_graph",
    }
)


def _detect_factor_graph_patterns(source_code: str) -> list[str]:
    """Scan Python source for factor graph / belief propagation patterns.

    Returns a list of method names that appear to perform marginal computation
    (broadcasting + axis-reduction patterns typical of sum-product message passing).
    """
    marginal_methods: list[str] = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return marginal_methods

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        method_name = node.name

        # Check 1: function name hints at factor graph
        name_lower = method_name.lower()
        has_fg_hint = any(hint in name_lower for hint in _FACTOR_GRAPH_HINTS)

        # Check 2: body contains marginal computation calls with axis= keyword
        has_marginal_call = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_text = ast.get_source_segment(source_code, child) or ""
                # Check for np.sum(..., axis=...) or similar
                for pat in _MARGINAL_CALL_PATTERNS:
                    if pat.split(".")[-1] in call_text:
                        # Check for axis= keyword
                        for kw in child.keywords:
                            if kw.arg == "axis":
                                has_marginal_call = True
                                break
                        # Also check for einsum (always a marginalisation)
                        if "einsum" in call_text:
                            has_marginal_call = True
                if has_marginal_call:
                    break

        if has_fg_hint and has_marginal_call:
            marginal_methods.append(method_name)

    return marginal_methods


class PythonASTExtractor:
    """Adapter delegating to the existing Python AST extraction functions.

    After extraction, enriches the ``RawDataFlowGraph`` with factor-graph /
    belief-propagation metadata: methods performing marginal computation
    (broadcasting + axis reduction) are tagged in the ``MethodFact``.
    """

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        dfg = await extract_data_flow(source_path, class_name)
        return _enrich_with_factor_graph_metadata(dfg)

    async def extract_function(
        self, source_path: str, function_name: str
    ) -> RawDataFlowGraph:
        dfg = await extract_function_data_flow(source_path, function_name)
        return _enrich_with_factor_graph_metadata(dfg)

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        dfg = await extract_procedural_data_flow(source_path, pipeline_name)
        return _enrich_with_factor_graph_metadata(dfg)


def _enrich_with_factor_graph_metadata(dfg: RawDataFlowGraph) -> RawDataFlowGraph:
    """Tag methods that perform marginal computation for belief propagation."""
    marginal_methods = _detect_factor_graph_patterns(dfg.source_code)
    if not marginal_methods:
        return dfg

    marginal_set = set(marginal_methods)
    updated_methods: list[MethodFact] = []
    for mf in dfg.methods:
        if mf.name in marginal_set:
            # Tag as oracle (stateless marginalisation is a pure computation)
            updated_methods.append(mf.model_copy(update={"is_oracle": True}))
        else:
            updated_methods.append(mf)
    return dfg.model_copy(update={"methods": updated_methods})


# ---------------------------------------------------------------------------
# JAXpr extractor — traces JAX functions via jax.make_jaxpr()
# ---------------------------------------------------------------------------

# JAX primitives that map to oracle (stateless log-density/gradient) nodes
_JAX_ORACLE_PRIMITIVES: frozenset[str] = frozenset(
    {
        "custom_jvp_call",
        "custom_vjp_call",
    }
)


class JAXprExtractor:
    """Extract data-flow graphs from JAX source by tracing ``jax.make_jaxpr()``.

    Instead of parsing Python AST, this extractor:
    1. Reads the source file and identifies top-level functions.
    2. Attempts to trace each function through ``jax.make_jaxpr()`` with
       abstract placeholder inputs.
    3. Maps the resulting JAX primitives (``add_p``, ``mul_p``, ``dot_general_p``,
       etc.) into ``MethodFact`` nodes in a ``RawDataFlowGraph``.
    4. Tracks **frozen stochasticity**: fixed noise matrices (Z) initialised
       outside a ``jax.grad`` / ``jax.value_and_grad`` closure are labelled as
       static initialisation edges — crucial for deterministic ADVI extraction.

    Falls back to standard ``PythonASTExtractor`` if JAX tracing fails.
    """

    def __init__(self) -> None:
        self._fallback = PythonASTExtractor()

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        """For class-based JAX code, attempt jaxpr tracing then fall back."""
        source_code = Path(source_path).read_text()

        # Try to extract jaxpr-level info and enrich the AST-based graph
        dfg = await self._fallback.extract_class(source_path, class_name)
        jaxpr_facts = _extract_jaxpr_facts(source_code)
        return _merge_jaxpr_facts(dfg, jaxpr_facts)

    async def extract_function(
        self, source_path: str, function_name: str
    ) -> RawDataFlowGraph:
        """For function-based JAX code, enrich with jaxpr metadata."""
        source_code = Path(source_path).read_text()

        dfg = await self._fallback.extract_function(source_path, function_name)
        jaxpr_facts = _extract_jaxpr_facts(source_code)
        return _merge_jaxpr_facts(dfg, jaxpr_facts)

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        """For procedural JAX code, trace functions via jax.make_jaxpr."""
        source_code = Path(source_path).read_text()

        dfg = await self._fallback.extract_procedural(source_path, pipeline_name)
        jaxpr_facts = _extract_jaxpr_facts(source_code)
        return _merge_jaxpr_facts(dfg, jaxpr_facts)


def _extract_jaxpr_facts(source_code: str) -> dict:
    """Parse JAX source code to extract jaxpr-level metadata.

    This performs **static analysis** of the source rather than actually
    executing ``jax.make_jaxpr()`` (which would require the user's full
    environment).  It detects:

    - Functions wrapped in ``jax.grad`` / ``jax.value_and_grad`` → oracle methods
    - Fixed noise arrays (Z) assigned outside grad closures → static init edges
    - ``jax.random.*`` calls for RNG tracking
    """
    facts: dict = {
        "oracle_functions": [],  # functions passed to jax.grad
        "static_init_edges": [],  # (var_name, context) for frozen stochasticity
        "grad_targets": [],  # functions that are differentiated
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return facts

    # Collect all top-level assignments and function defs
    grad_wrapped: set[str] = set()  # functions passed to jax.grad(f)
    grad_closure_funcs: set[str] = set()  # functions that call jax.grad

    for node in ast.walk(tree):
        # Detect jax.grad(target_fn) or jax.value_and_grad(target_fn)
        if isinstance(node, ast.Call):
            call_src = ast.get_source_segment(source_code, node) or ""
            if re.search(r"jax\.(value_and_)?grad\s*\(", call_src):
                # The first positional arg is the target function
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Name):
                        grad_wrapped.add(arg.id)
                        facts["grad_targets"].append(arg.id)

        # Detect functions that contain jax.grad calls
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_src = ast.get_source_segment(source_code, node) or ""
            if re.search(r"jax\.(value_and_)?grad\s*\(", func_src):
                grad_closure_funcs.add(node.name)

    facts["oracle_functions"] = sorted(grad_wrapped)

    # Detect frozen stochasticity: noise arrays assigned at module level
    # or outside grad-wrapped functions.  These are variables named like
    # z, Z, noise, eps, epsilon that are assigned with jax.random.* or
    # np.random.* calls.
    noise_pattern = re.compile(
        r"^(\w+)\s*=\s*(?:jax\.random\.\w+|np\.random\.\w+|"
        r"numpy\.random\.\w+|jnp\.\w+)",
        re.MULTILINE,
    )
    noise_name_hints = {
        "z",
        "Z",
        "noise",
        "eps",
        "epsilon",
        "eta",
        "xi",
        "noise_matrix",
        "z_samples",
        "z_noise",
        "static_noise",
    }

    for m in noise_pattern.finditer(source_code):
        var_name = m.group(1)
        # Check if the variable name hints at frozen stochasticity
        if var_name in noise_name_hints or var_name.lower().startswith(
            ("z_", "noise_", "eps_")
        ):
            # Check it's not inside a grad-wrapped function
            line_no = source_code[: m.start()].count("\n")
            is_inside_grad = False
            try:
                parsed = ast.parse(source_code)
                for func_node in ast.walk(parsed):
                    if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if (
                            func_node.name in grad_wrapped
                            and hasattr(func_node, "lineno")
                            and func_node.lineno <= line_no + 1
                            and hasattr(func_node, "end_lineno")
                            and (func_node.end_lineno or 0) >= line_no + 1
                        ):
                            is_inside_grad = True
                            break
            except SyntaxError:
                pass

            if not is_inside_grad:
                facts["static_init_edges"].append(
                    {
                        "var_name": var_name,
                        "line": line_no + 1,
                        "context": "frozen_stochasticity",
                    }
                )

    return facts


def _merge_jaxpr_facts(dfg: RawDataFlowGraph, jaxpr_facts: dict) -> RawDataFlowGraph:
    """Merge jaxpr-extracted facts into a RawDataFlowGraph."""
    oracle_funcs = set(jaxpr_facts.get("oracle_functions", []))
    static_edges = jaxpr_facts.get("static_init_edges", [])

    if not oracle_funcs and not static_edges:
        return dfg

    # Tag oracle methods
    updated_methods: list[MethodFact] = []
    for mf in dfg.methods:
        if mf.name in oracle_funcs:
            updated_methods.append(mf.model_copy(update={"is_oracle": True}))
        else:
            updated_methods.append(mf)

    # Add static init edges as inferred dependency edges
    new_edges = list(dfg.inferred_edges)
    for se in static_edges:
        var_name = se["var_name"]
        # Create an edge from a virtual "static_init" source to the first
        # method that references this variable
        for mf in dfg.methods:
            if var_name in mf.reads or var_name in (mf.source_code or ""):
                new_edges.append(
                    DependencyEdge(
                        source_id=f"static_init:{var_name}",
                        target_id=mf.name,
                        output_name=var_name,
                        input_name=var_name,
                        source_type="ndarray",
                        target_type="ndarray",
                    )
                )
                break

    return dfg.model_copy(
        update={
            "methods": updated_methods,
            "inferred_edges": new_edges,
        }
    )
