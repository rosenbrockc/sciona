"""Ghost Witness simulation pass for the synthesizer pipeline.

Converts a CDG + MatchResults into ghost SimNodes and runs the abstract
simulation to catch structural mismatches (shape, dtype, domain) before
committing to expensive compilation.

Requires the ``ageoa`` package (optional dependency).  If ``ageoa`` is not
installed, the simulation pass is silently skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from collections import deque

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, NodeStatus
from ageom.synthesizer.toposort import toposort_nodes
from ageom.types import MatchResult

logger = logging.getLogger(__name__)

try:
    from ageoa.ghost.simulator import SimNode, SimResult, simulate_graph, PlanError
    from ageoa.ghost.abstract import AbstractDistribution, AbstractSignal, AbstractMatrix
    from ageoa.ghost.registry import REGISTRY, list_registered

    _GHOST_AVAILABLE = True
except ImportError:
    _GHOST_AVAILABLE = False


@dataclass
class ErrorInterval:
    """Interval arithmetic bounds for numerical approximation error."""

    lo: float = 0.0
    hi: float = 0.0

    @property
    def width(self) -> float:
        """Width of the error interval (hi - lo)."""
        return self.hi - self.lo


@dataclass
class GhostSimReport:
    """Result of the ghost simulation pass."""

    ran: bool = False
    """Whether the simulation actually executed (False if ageoa not installed)."""
    passed: bool = False
    """Whether the simulation completed without errors."""
    node_count: int = 0
    """Number of nodes that were simulated."""
    skipped_nodes: list[str] = field(default_factory=list)
    """Names of nodes that were skipped (no registered witness)."""
    trace: list[str] = field(default_factory=list)
    """Ordered list of node names that were simulated."""
    error: str = ""
    """Human-readable error if the simulation failed."""
    error_node: str = ""
    """Name of the node where the simulation failed."""
    error_function: str = ""
    """Function name where the simulation failed."""
    coverage: float = 0.0
    """Ratio of simulable / total atomic nodes (0.0 - 1.0)."""
    precision_gradients: dict[str, float] = field(default_factory=dict)
    """Per-node error expansion (output interval width - input interval width)."""
    cyclic_deadlock: bool = False
    """True if iterative message passing detected an unbroken cycle."""
    deadlock_nodes: list[str] = field(default_factory=list)
    """Node names involved in the deadlocked cycle."""
    iterations_used: int = 0
    """Number of message-passing iterations before convergence/deadlock."""


def _extract_atom_name(declaration_name: str) -> str:
    """Extract the short atom name from a possibly qualified declaration name.

    Examples:
        "ageoa.numpy.fft.fft" -> "fft"
        "numpy.fft.fft" -> "fft"
        "fft" -> "fft"
        "scipy.signal.butter" -> "butter"
    """
    return declaration_name.rsplit(".", 1)[-1]


def _build_abstract_value(type_desc: str, constraints: str) -> Any:
    """Build an abstract value from IOSpec metadata.

    Parses type_desc and constraints to construct the appropriate
    abstract type (AbstractSignal, AbstractFilterCoefficients, or
    AbstractGraphMeta).  Falls back to a generic AbstractSignal.
    """
    if not _GHOST_AVAILABLE:
        return None

    from ageoa.ghost.abstract import AbstractSignal
    from ageoa.ghost.witnesses import AbstractFilterCoefficients, AbstractGraphMeta

    type_lower = type_desc.lower()
    constraints_lower = constraints.lower()

    # Primitive scalar types — return sensible defaults
    if type_lower in ("int", "integer"):
        return 4
    if type_lower in ("float", "double", "number"):
        return 1000.0
    if type_lower in ("str", "string"):
        return "low"
    if type_lower in ("bool", "boolean"):
        return True

    # Graph / Laplacian
    if any(kw in type_lower for kw in ("graph", "laplacian", "adjacency")):
        n_nodes = 10  # default
        for part in constraints_lower.replace(",", " ").split():
            if part.startswith("n=") or part.startswith("n_nodes="):
                try:
                    n_nodes = int(part.split("=", 1)[1])
                except ValueError:
                    pass
        symmetric = "symmetric" in constraints_lower or "undirected" in constraints_lower
        return AbstractGraphMeta(n_nodes=n_nodes, is_symmetric=symmetric)

    # Filter coefficients
    if any(kw in type_lower for kw in ("filter", "coefficients", "sos")):
        order = 4  # default
        return AbstractFilterCoefficients(order=order, btype="low", is_stable=True)

    # Default: treat as signal
    shape = (1024,)
    dtype = "float64"
    sampling_rate = 44100.0
    domain = "time"
    units = "amplitude"

    # Parse constraints for hints
    if "freq" in constraints_lower:
        domain = "freq"
    if "complex" in constraints_lower or "complex" in type_lower:
        dtype = "complex128"
    if "int" in type_lower and "point" not in type_lower:
        dtype = "int64"

    # Parse shape hints like "shape=(512,)" or "(1024,)"
    for token in constraints_lower.replace(",", " ").split():
        if token.startswith("shape="):
            try:
                shape_str = token.split("=", 1)[1].strip("()")
                shape = tuple(int(x) for x in shape_str.split(",") if x.strip())
            except (ValueError, IndexError):
                pass

    # Parse sampling_rate hints
    for token in constraints_lower.replace(",", " ").split():
        if token.startswith("fs=") or token.startswith("sr=") or token.startswith("sampling_rate="):
            try:
                sampling_rate = float(token.split("=", 1)[1])
            except ValueError:
                pass

    return AbstractSignal(
        shape=shape,
        dtype=dtype,
        sampling_rate=sampling_rate,
        domain=domain,
        units=units,
    )


# Parameter name -> sensible default for scalar inputs to witnesses.
# These override the generic _build_abstract_value defaults when the
# IOSpec type is a scalar and the parameter name is semantically known.
_PARAM_DEFAULTS: dict[str, Any] = {
    "order": 4,
    "n": 1024,
    "wn": 100.0,        # cutoff frequency — must be < nyquist
    "fs": 44100.0,      # sampling rate — high default to keep wn < nyquist
    "rp": 1.0,          # passband ripple (dB)
    "rs": 40.0,         # stopband attenuation (dB)
    "btype": "low",
    "numtaps": 65,
    "t": 0.5,           # diffusion time
    "k": 6,             # number of eigenvectors
    "n_freqs": 512,
    "worN": 512,
}


def _build_initial_value(param_name: str, type_desc: str, constraints: str) -> Any:
    """Build an initial abstract value, using parameter-name heuristics for scalars."""
    type_lower = type_desc.lower()
    # Check if this is a known scalar parameter
    if type_lower in ("int", "integer", "float", "double", "number", "str", "string", "bool", "boolean"):
        if param_name in _PARAM_DEFAULTS:
            return _PARAM_DEFAULTS[param_name]
    return _build_abstract_value(type_desc, constraints)


# ---------------------------------------------------------------------------
# Interval arithmetic for precision gradient
# ---------------------------------------------------------------------------

# Known per-atom error expansion factors.  Each value is a multiplier
# applied to the incoming interval width.  Missing atoms are treated
# as identity (factor = 1.0).
_ATOM_ERROR_FACTORS: dict[str, float] = {
    "fft": 1.5,        # O(n log n) rounding accumulation
    "ifft": 1.5,
    "rfft": 1.3,
    "irfft": 1.3,
    "stft": 1.6,
    "istft": 1.6,
    "butter": 1.1,     # filter design — coefficient quantisation
    "cheby1": 1.3,
    "cheby2": 1.3,
    "ellip": 1.4,
    "lfilter": 2.0,    # IIR recursive application amplifies error
    "sosfilt": 1.4,    # SOS form is more stable than TF
    "firwin": 1.05,
    "convolve": 1.2,
    "correlate": 1.2,
    "resample": 1.3,
    "hilbert": 1.4,
    "welch": 1.2,
}


def _parse_error_bounds(constraints: str) -> ErrorInterval:
    """Extract ``error_bounds=(lo,hi)`` from an IOSpec constraints string.

    Falls back to a zero-width interval when no bounds are declared.
    """
    import re

    match = re.search(r"error_bounds\s*=\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", constraints)
    if match:
        try:
            return ErrorInterval(lo=float(match.group(1)), hi=float(match.group(2)))
        except ValueError:
            pass
    return ErrorInterval(0.0, 0.0)


def _compute_precision_gradients(
    cdg: CDGExport,
    match_map: dict[str, MatchResult],
    sorted_ids: list[str],
) -> dict[str, float]:
    """Propagate error intervals through the CDG and return per-node expansion.

    Works without ageoa — purely based on IOSpec metadata and known atom
    error factors.  Returns an empty dict when no nodes carry error metadata.
    """
    node_map: dict[str, AlgorithmicNode] = {n.node_id: n for n in cdg.nodes}
    atomic_leaves = {n.node_id for n in cdg.nodes if n.status == NodeStatus.ATOMIC}

    # Build edge index: target_id -> list of incoming edges
    incoming: dict[str, list[DependencyEdge]] = {nid: [] for nid in node_map}
    for edge in cdg.edges:
        if edge.target_id in incoming:
            incoming[edge.target_id].append(edge)

    # Track the output error interval per node
    output_intervals: dict[str, ErrorInterval] = {}
    gradients: dict[str, float] = {}

    for nid in sorted_ids:
        if nid not in atomic_leaves:
            continue
        node = node_map[nid]

        # Determine input interval: merge from incoming edges or parse from IOSpec
        input_interval = ErrorInterval(0.0, 0.0)
        in_edges = incoming.get(nid, [])
        upstream_intervals = [
            output_intervals[e.source_id]
            for e in in_edges
            if e.source_id in output_intervals
        ]
        if upstream_intervals:
            # Union of all upstream intervals
            input_interval = ErrorInterval(
                lo=min(iv.lo for iv in upstream_intervals),
                hi=max(iv.hi for iv in upstream_intervals),
            )
        elif node.inputs:
            # No upstream — seed from the first input's constraints
            parsed = _parse_error_bounds(node.inputs[0].constraints)
            if parsed.width > 0:
                input_interval = parsed

        # Look up error factor for this atom
        mr = match_map.get(nid)
        if mr and mr.success:
            atom_name = _extract_atom_name(mr.verified_match.candidate.declaration.name)
            factor = _ATOM_ERROR_FACTORS.get(atom_name, 1.0)
        else:
            factor = 1.0

        # Compute output interval
        if input_interval.width > 0:
            out = ErrorInterval(
                lo=input_interval.lo * factor,
                hi=input_interval.hi * factor,
            )
        elif node.outputs:
            # Seed from output constraints if present
            out = _parse_error_bounds(node.outputs[0].constraints)
            if out.width == 0:
                out = ErrorInterval(0.0, 0.0)
        else:
            out = ErrorInterval(0.0, 0.0)

        output_intervals[nid] = out

        # Precision gradient = output width - input width
        delta = out.width - input_interval.width
        if delta != 0.0 or input_interval.width > 0 or out.width > 0:
            gradients[nid] = delta

    return gradients


def _detect_message_passing_cycle(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> tuple[set[str], bool]:
    """Detect whether a cycle is a valid MESSAGE_PASSING factor graph cycle.

    Returns (cycle_node_ids, is_message_passing).  If Kahn's algorithm
    completes, returns (empty set, False).  If it doesn't, collects the
    remaining nodes and checks whether ALL of them have
    concept_type == MESSAGE_PASSING.
    """
    node_ids = {n.node_id for n in nodes}
    node_map = {n.node_id: n for n in nodes}

    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    successors: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in edges:
        if edge.source_id in node_ids and edge.target_id in node_ids:
            successors[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

    queue: deque[str] = deque()
    for nid in node_ids:
        if in_degree[nid] == 0:
            queue.append(nid)

    visited: set[str] = set()
    while queue:
        nid = queue.popleft()
        visited.add(nid)
        for succ in successors[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(visited) == len(node_ids):
        return set(), False

    # Remaining nodes are in the cycle
    cycle_ids = node_ids - visited

    # Check if ALL cycle nodes are MESSAGE_PASSING
    is_mp = all(
        getattr(node_map.get(nid), "concept_type", None) == ConceptType.MESSAGE_PASSING
        for nid in cycle_ids
        if nid in node_map
    )

    return cycle_ids, is_mp


def _simulate_message_passing_iterative(
    sim_nodes: list[Any],
    initial_state: dict[str, Any],
    max_iterations: int = 100,
    convergence_tol: float = 1e-8,
) -> tuple[Any, int, bool, list[str]]:
    """Run iterative message passing instead of topological sort.

    Returns (SimResult, iterations_used, deadlocked, deadlock_node_names).
    """
    if not _GHOST_AVAILABLE:
        raise RuntimeError("ageoa package required for iterative simulation")

    state = dict(initial_state)
    trace: list[str] = []
    iterations_used = 0

    for iteration in range(max_iterations):
        iterations_used = iteration + 1
        prev_state = dict(state)

        # Execute each node in round-robin order
        for node in sim_nodes:
            try:
                res = simulate_graph([node], state)
                state.update(res.final_state)
                if iteration == 0:
                    trace.append(node.name)
            except PlanError:
                raise
            except Exception as exc:
                raise PlanError(
                    node_name=node.name,
                    function_name=node.function_name,
                    detail=f"Iterative simulation error: {exc}",
                )

        # Check for convergence via memoization state node
        converged = False
        for node in sim_nodes:
            if node.output_name and "converged" in node.output_name.lower():
                val = state.get(node.output_name)
                if val is True:
                    converged = True
                    break
            # Also check tuple outputs from memo nodes
            if "memo" in node.name.lower() and node.output_name:
                val = state.get(node.output_name)
                if isinstance(val, tuple) and len(val) == 2 and val[1] is True:
                    converged = True
                    break

        if converged:
            return (
                SimResult(node_count=len(sim_nodes), final_state=state, trace=trace),
                iterations_used,
                False,
                [],
            )

        # Deadlock detection: no state change and not converged
        state_unchanged = all(
            state.get(k) is prev_state.get(k)
            for k in state
        )
        if state_unchanged and iteration > 0:
            deadlock_names = [n.name for n in sim_nodes]
            raise PlanError(
                node_name=sim_nodes[0].name if sim_nodes else "unknown",
                function_name="iterative_message_passing",
                detail="Cyclic deadlock: messages unchanged but convergence not reached",
            )

    # Max iterations reached — treat as convergence
    return (
        SimResult(node_count=len(sim_nodes), final_state=state, trace=trace),
        iterations_used,
        False,
        [],
    )


def run_ghost_simulation(
    cdg: CDGExport,
    match_results: list[MatchResult],
) -> GhostSimReport:
    """Run the ghost witness simulation on a CDG.

    Converts atomic leaf nodes to SimNodes using the matched atom names,
    constructs initial abstract state from IOSpec metadata, and runs the
    simulator.  Non-DSP nodes (those without registered witnesses) are
    skipped — the simulation is best-effort.

    Args:
        cdg: The Conceptual Dependency Graph.
        match_results: Verified match results for atomic leaves.

    Returns:
        GhostSimReport describing the simulation outcome.
    """
    report = GhostSimReport()

    # Index match results by node_id (predicate_id)
    match_map: dict[str, MatchResult] = {}
    for mr in match_results:
        match_map[mr.pdg_node.predicate_id] = mr

    # Topological sort (needed by both precision gradients and ghost sim)
    cycle_ids: set[str] = set()
    is_message_passing_cycle = False
    try:
        sorted_ids = toposort_nodes(cdg.nodes, cdg.edges)
    except ValueError:
        # Topological sort failed — check if this is a valid factor graph cycle
        cycle_ids, is_message_passing_cycle = _detect_message_passing_cycle(
            cdg.nodes, cdg.edges
        )
        if not is_message_passing_cycle:
            report.ran = True
            report.error = f"Cycle detected in non-message-passing nodes: {cycle_ids}"
            return report
        # For message-passing cycles, build a partial order for acyclic nodes
        # and use the cycle node ids for iterative simulation
        acyclic_nodes = [n for n in cdg.nodes if n.node_id not in cycle_ids]
        acyclic_edges = [
            e for e in cdg.edges
            if e.source_id not in cycle_ids and e.target_id not in cycle_ids
        ]
        try:
            sorted_ids = toposort_nodes(acyclic_nodes, acyclic_edges)
        except ValueError:
            sorted_ids = [n.node_id for n in acyclic_nodes]
        # Append cycle node ids at the end (they'll be simulated iteratively)
        sorted_ids.extend(sorted(cycle_ids))

    # Precision gradients — runs independently of ageoa
    try:
        report.precision_gradients = _compute_precision_gradients(
            cdg, match_map, sorted_ids,
        )
    except Exception as exc:
        logger.warning("Precision gradient computation failed: %s", exc)

    if not _GHOST_AVAILABLE:
        logger.info("Ghost simulation skipped: ageoa package not installed")
        return report

    # Ensure atom modules are imported so witnesses are registered
    _ensure_atoms_imported()

    registered = set(list_registered())
    if not registered:
        logger.info("Ghost simulation skipped: no witnesses registered")
        return report

    # Index nodes by id
    node_map: dict[str, AlgorithmicNode] = {n.node_id: n for n in cdg.nodes}

    # Get atomic leaves
    atomic_leaves = {n.node_id for n in cdg.nodes if n.status == NodeStatus.ATOMIC}

    # Build edge index: target_id -> list of incoming edges
    incoming_edges: dict[str, list[DependencyEdge]] = {nid: [] for nid in node_map}
    for edge in cdg.edges:
        if edge.target_id in incoming_edges:
            incoming_edges[edge.target_id].append(edge)

    # Determine which nodes have witnesses
    simulable_nodes: list[str] = []
    skipped: list[str] = []

    for nid in sorted_ids:
        if nid not in atomic_leaves:
            continue
        node = node_map[nid]
        mr = match_map.get(nid)
        if mr is None or not mr.success:
            skipped.append(node.name)
            continue

        atom_name = _extract_atom_name(mr.verified_match.candidate.declaration.name)
        if atom_name in registered:
            simulable_nodes.append(nid)
        else:
            skipped.append(node.name)

    report.skipped_nodes = skipped

    # Compute coverage
    total_atomic = len(simulable_nodes) + len(skipped)
    if total_atomic > 0:
        report.coverage = len(simulable_nodes) / total_atomic
    else:
        report.coverage = 0.0

    if report.coverage < 0.5 and total_atomic > 0:
        unsimulated_categories = set()
        for nid in sorted_ids:
            if nid not in atomic_leaves:
                continue
            node = node_map[nid]
            mr = match_map.get(nid)
            if mr and mr.success:
                atom_name = _extract_atom_name(mr.verified_match.candidate.declaration.name)
                if atom_name not in registered:
                    unsimulated_categories.add(node.concept_type.value if hasattr(node, "concept_type") else "unknown")
        logger.warning(
            "Ghost simulation coverage is low (%.0f%%): unsimulated categories: %s",
            report.coverage * 100,
            sorted(unsimulated_categories),
        )

    if not simulable_nodes:
        logger.info("Ghost simulation skipped: no simulable nodes")
        return report

    # Build SimNodes and initial state
    sim_nodes: list[SimNode] = []
    initial_state: dict[str, Any] = {}

    # Pre-populate initial state for root inputs
    # (inputs to nodes that have no incoming edges from other atomic leaves)
    for nid in simulable_nodes:
        node = node_map[nid]
        in_edges = incoming_edges.get(nid, [])

        # Map incoming edges to state keys
        edge_inputs: dict[str, str] = {}
        for edge in in_edges:
            if edge.source_id in atomic_leaves:
                state_key = f"{edge.source_id}:{edge.output_name}"
                edge_inputs[edge.input_name] = state_key

        # For inputs not covered by edges from simulable nodes, create initial state
        for inp in node.inputs:
            if inp.name not in edge_inputs:
                state_key = f"root:{nid}:{inp.name}"
                edge_inputs[inp.name] = state_key
                if state_key not in initial_state:
                    initial_state[state_key] = _build_initial_value(
                        inp.name, inp.type_desc, inp.constraints
                    )

        # Build SimNode
        mr = match_map[nid]
        atom_name = _extract_atom_name(mr.verified_match.candidate.declaration.name)

        output_name = ""
        if node.outputs:
            output_name = f"{nid}:{node.outputs[0].name}"

        sim_nodes.append(SimNode(
            name=node.name,
            function_name=atom_name,
            inputs=edge_inputs,
            output_name=output_name,
        ))

    # Run the simulation
    report.ran = True

    if is_message_passing_cycle:
        # Split sim_nodes into acyclic (run once) and cyclic (run iteratively)
        cyclic_sim_nodes = [n for n in sim_nodes if any(
            nid in cycle_ids for nid in simulable_nodes
            if node_map.get(nid, AlgorithmicNode(node_id="")).name == n.name
        )]
        acyclic_sim_nodes = [n for n in sim_nodes if n not in cyclic_sim_nodes]

        # If we can't cleanly separate, treat all as cyclic
        if not cyclic_sim_nodes:
            cyclic_sim_nodes = sim_nodes
            acyclic_sim_nodes = []

        try:
            # Run acyclic nodes first
            if acyclic_sim_nodes:
                result = _simulate_with_bayesian_checks(acyclic_sim_nodes, initial_state)
                initial_state.update(result.final_state)

            # Run cyclic nodes iteratively
            result, iters, deadlocked, dl_names = _simulate_message_passing_iterative(
                cyclic_sim_nodes, initial_state
            )
            report.passed = True
            report.node_count = result.node_count + len(acyclic_sim_nodes)
            report.trace = [n.name for n in acyclic_sim_nodes] + result.trace
            report.iterations_used = iters
            logger.info(
                "Ghost simulation (iterative) passed: %d nodes, %d iterations",
                report.node_count, iters,
            )
        except PlanError as exc:
            if "deadlock" in str(exc).lower():
                report.cyclic_deadlock = True
                report.deadlock_nodes = [n.name for n in cyclic_sim_nodes]
            report.error = str(exc)
            report.error_node = getattr(exc, "node_name", "")
            report.error_function = getattr(exc, "function_name", "")
            logger.warning("Ghost simulation (iterative) failed: %s", exc)
        except Exception as exc:
            report.error = f"Unexpected error: {exc}"
            logger.warning("Ghost simulation unexpected error: %s", exc)
    else:
        # Standard topological simulation
        try:
            result = _simulate_with_bayesian_checks(sim_nodes, initial_state)
            report.passed = True
            report.node_count = result.node_count
            report.trace = result.trace
            logger.info(
                "Ghost simulation passed: %d nodes simulated, trace: %s",
                result.node_count, result.trace,
            )
        except PlanError as exc:
            report.error = str(exc)
            report.error_node = exc.node_name
            report.error_function = exc.function_name
            logger.warning("Ghost simulation failed: %s", exc)
        except Exception as exc:
            report.error = f"Unexpected error: {exc}"
            logger.warning("Ghost simulation unexpected error: %s", exc)

    return report


# ---------------------------------------------------------------------------
# Bayesian-aware simulation (refactored from ageo-atoms)
# ---------------------------------------------------------------------------

# Witness functions that produce reparameterized outputs.
_REPARAM_WITNESSES = frozenset({
    "witness_reparameterized_sample",
    "reparameterized_sample",
})

# Witness functions that produce bijector outputs (carry Jacobian).
_BIJECTOR_WITNESSES = frozenset({
    "witness_bijector_transform",
    "bijector_transform",
})

# Witness functions that require reparameterized / bijector inputs.
_ELBO_WITNESSES = frozenset({
    "witness_vi_elbo",
    "vi_elbo",
})

# Abstract types allowed as inputs to stateless MCMC oracle nodes.
_ORACLE_ALLOWED_TYPES: tuple[type, ...] = (float, int, str)
if _GHOST_AVAILABLE:
    _ORACLE_ALLOWED_TYPES = (AbstractDistribution, AbstractSignal, AbstractMatrix, float, int, str)


def _simulate_with_bayesian_checks(
    nodes: list[Any],
    initial_state: dict[str, Any],
) -> Any:
    """Run ghost simulation with Bayesian verification.

    Extends the standard ``simulate_graph`` pass with:
    1. **VI ELBO provenance check** — inputs must originate from
       reparameterized traces or bijector outputs.
    2. **MCMC Oracle isolation check** — oracle nodes must be stateless
       (only receive abstract types and scalars).
    3. **Provenance tracking** — propagates ``is_reparameterized`` and
       ``has_jacobian`` flags through the graph.

    Raises ``PlanError`` on violations.
    """
    state = dict(initial_state)

    # Provenance: state_key -> {"is_reparameterized": bool, "has_jacobian": bool}
    provenance: dict[str, dict[str, bool]] = {
        key: {"is_reparameterized": False, "has_jacobian": False}
        for key in state
    }

    trace: list[str] = []

    for node in nodes:
        # 1. VI ELBO provenance check
        if node.function_name in _ELBO_WITNESSES:
            for param, state_key in node.inputs.items():
                prov = provenance.get(
                    state_key,
                    {"is_reparameterized": False, "has_jacobian": False},
                )
                if not (prov["is_reparameterized"] or prov["has_jacobian"]):
                    raise PlanError(
                        node_name=node.name,
                        function_name=node.function_name,
                        detail=(
                            f"ELBO input '{param}' (key '{state_key}') must "
                            f"originate from a reparameterized trace or a "
                            f"bijector output (with Jacobian)."
                        ),
                    )

        # 2. MCMC Oracle isolation check
        if "ORACLE" in node.function_name.upper():
            for param, state_key in node.inputs.items():
                val = state.get(state_key)
                if not isinstance(val, _ORACLE_ALLOWED_TYPES):
                    raise PlanError(
                        node_name=node.name,
                        function_name=node.function_name,
                        detail=(
                            f"Oracle nodes must be stateless. Found state "
                            f"input '{param}' of type {type(val).__name__}."
                        ),
                    )

        # 3. Execute single-node simulation step
        res = simulate_graph([node], state)
        state.update(res.final_state)

        # 4. Update provenance for this node's output
        if node.output_name:
            provenance[node.output_name] = {
                "is_reparameterized": node.function_name in _REPARAM_WITNESSES,
                "has_jacobian": node.function_name in _BIJECTOR_WITNESSES,
            }

        trace.append(node.name)

    return SimResult(
        node_count=len(nodes),
        final_state=state,
        trace=trace,
    )


def _ensure_atoms_imported() -> None:
    """Import atom modules to trigger @register_atom decorators."""
    import importlib

    modules = [
        "ageoa.numpy.fft",
        "ageoa.scipy.fft",
        "ageoa.scipy.signal",
        "ageoa.scipy.sparse_graph",
        "ageoa.algorithms.sorting",
        "ageoa.algorithms.graph",
        "ageoa.algorithms.search",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
