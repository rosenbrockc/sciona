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

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, DependencyEdge, NodeStatus
from ageom.synthesizer.toposort import toposort_nodes
from ageom.types import MatchResult

logger = logging.getLogger(__name__)

try:
    from ageoa.ghost.simulator import SimNode, SimResult, simulate_graph, PlanError
    from ageoa.ghost.registry import REGISTRY, list_registered

    _GHOST_AVAILABLE = True
except ImportError:
    _GHOST_AVAILABLE = False


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

    if not _GHOST_AVAILABLE:
        logger.info("Ghost simulation skipped: ageoa package not installed")
        return report

    # Ensure atom modules are imported so witnesses are registered
    _ensure_atoms_imported()

    registered = set(list_registered())
    if not registered:
        logger.info("Ghost simulation skipped: no witnesses registered")
        return report

    # Index match results by node_id (predicate_id)
    match_map: dict[str, MatchResult] = {}
    for mr in match_results:
        match_map[mr.pdg_node.predicate_id] = mr

    # Index nodes by id
    node_map: dict[str, AlgorithmicNode] = {n.node_id: n for n in cdg.nodes}

    # Get atomic leaves
    atomic_leaves = {n.node_id for n in cdg.nodes if n.status == NodeStatus.ATOMIC}

    # Build edge index: target_id -> list of incoming edges
    incoming_edges: dict[str, list[DependencyEdge]] = {nid: [] for nid in node_map}
    for edge in cdg.edges:
        if edge.target_id in incoming_edges:
            incoming_edges[edge.target_id].append(edge)

    # Topological sort
    try:
        sorted_ids = toposort_nodes(cdg.nodes, cdg.edges)
    except ValueError as exc:
        report.ran = True
        report.error = f"Topological sort failed: {exc}"
        return report

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
    try:
        result: SimResult = simulate_graph(
            sim_nodes, initial_state,
        )
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


def _ensure_atoms_imported() -> None:
    """Import atom modules to trigger @register_atom decorators."""
    import importlib

    modules = [
        "ageoa.numpy.fft",
        "ageoa.scipy.fft",
        "ageoa.scipy.signal",
        "ageoa.scipy.sparse_graph",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
