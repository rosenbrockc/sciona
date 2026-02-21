"""Pre-built skeleton graphs for algorithmic paradigms.

Each skeleton is a template CDG that can be instantiated for a specific goal,
giving the Decomposer a head start on structure.
"""

from __future__ import annotations

import uuid

from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
    SkeletonGraph,
)


def _node(
    name: str,
    desc: str,
    concept: ConceptType,
    *,
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
    depth: int = 1,
    parallelizable: bool = False,
) -> AlgorithmicNode:
    """Helper to build a template node."""
    return AlgorithmicNode(
        node_id=f"tpl_{name.lower().replace(' ', '_')}",
        name=name,
        description=desc,
        concept_type=concept,
        inputs=inputs or [],
        outputs=outputs or [],
        depth=depth,
        status=NodeStatus.PENDING,
        parallelizable=parallelizable,
    )


def _edge(
    source: AlgorithmicNode,
    target: AlgorithmicNode,
    output_name: str = "result",
    input_name: str = "data",
    data_type: str = "any",
) -> DependencyEdge:
    """Helper to build a template edge."""
    return DependencyEdge(
        source_id=source.node_id,
        target_id=target.node_id,
        output_name=output_name,
        input_name=input_name,
        source_type=data_type,
        target_type=data_type,
    )


def _build_divide_and_conquer() -> SkeletonGraph:
    split = _node("Split", "Divide the input into sub-problems",
                   ConceptType.DIVIDE_AND_CONQUER,
                   inputs=[IOSpec(name="input", type_desc="any")],
                   outputs=[IOSpec(name="left", type_desc="any"),
                            IOSpec(name="right", type_desc="any")])
    recurse_left = _node("Recurse Left", "Solve the left sub-problem recursively",
                          ConceptType.DIVIDE_AND_CONQUER,
                          inputs=[IOSpec(name="subproblem", type_desc="any")],
                          outputs=[IOSpec(name="result", type_desc="any")])
    recurse_right = _node("Recurse Right", "Solve the right sub-problem recursively",
                           ConceptType.DIVIDE_AND_CONQUER,
                           inputs=[IOSpec(name="subproblem", type_desc="any")],
                           outputs=[IOSpec(name="result", type_desc="any")])
    merge = _node("Merge", "Combine results of sub-problems",
                   ConceptType.DIVIDE_AND_CONQUER,
                   inputs=[IOSpec(name="left_result", type_desc="any"),
                           IOSpec(name="right_result", type_desc="any")],
                   outputs=[IOSpec(name="result", type_desc="any")])

    edges = [
        _edge(split, recurse_left, "left", "subproblem"),
        _edge(split, recurse_right, "right", "subproblem"),
        _edge(recurse_left, merge, "result", "left_result"),
        _edge(recurse_right, merge, "result", "right_result"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.DIVIDE_AND_CONQUER,
        name="Divide and Conquer",
        description="Split input, solve sub-problems recursively, merge results",
        template_nodes=[split, recurse_left, recurse_right, merge],
        template_edges=edges,
        variants=["merge_sort", "quicksort", "strassen", "closest_pair"],
    )


def _build_dynamic_programming() -> SkeletonGraph:
    define = _node("Define Subproblems", "Define the subproblem structure and table dimensions",
                    ConceptType.DYNAMIC_PROGRAMMING,
                    inputs=[IOSpec(name="input", type_desc="any")],
                    outputs=[IOSpec(name="table_def", type_desc="table structure")])
    base = _node("Base Case", "Initialize base cases in the DP table",
                  ConceptType.DYNAMIC_PROGRAMMING,
                  inputs=[IOSpec(name="table_def", type_desc="table structure")],
                  outputs=[IOSpec(name="init_table", type_desc="partially filled table")])
    recurrence = _node("Recurrence", "Fill the DP table using the recurrence relation",
                        ConceptType.DYNAMIC_PROGRAMMING,
                        inputs=[IOSpec(name="init_table", type_desc="partially filled table")],
                        outputs=[IOSpec(name="filled_table", type_desc="completed table")])
    memoize = _node("Memoize", "Store and reuse computed subproblem solutions",
                     ConceptType.DYNAMIC_PROGRAMMING,
                     inputs=[IOSpec(name="filled_table", type_desc="completed table")],
                     outputs=[IOSpec(name="memo_table", type_desc="memoized table")])
    extract = _node("Extract Solution", "Extract the final answer from the completed table",
                     ConceptType.DYNAMIC_PROGRAMMING,
                     inputs=[IOSpec(name="memo_table", type_desc="memoized table")],
                     outputs=[IOSpec(name="result", type_desc="any")])

    edges = [
        _edge(define, base, "table_def", "table_def", "table structure"),
        _edge(base, recurrence, "init_table", "init_table", "partially filled table"),
        _edge(recurrence, memoize, "filled_table", "filled_table", "completed table"),
        _edge(memoize, extract, "memo_table", "memo_table", "memoized table"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.DYNAMIC_PROGRAMMING,
        name="Dynamic Programming",
        description="Define subproblems, establish base cases, apply recurrence, extract solution",
        template_nodes=[define, base, recurrence, memoize, extract],
        template_edges=edges,
        variants=["lcs", "matrix_chain", "optimal_bst", "knapsack", "edit_distance"],
    )


def _build_greedy() -> SkeletonGraph:
    sort_cands = _node("Sort Candidates", "Order candidates by greedy criterion",
                        ConceptType.GREEDY,
                        inputs=[IOSpec(name="candidates", type_desc="list[any]")],
                        outputs=[IOSpec(name="sorted", type_desc="list[any]")])
    choose = _node("Greedy Choice", "Select the locally optimal candidate",
                    ConceptType.GREEDY,
                    inputs=[IOSpec(name="sorted", type_desc="list[any]")],
                    outputs=[IOSpec(name="chosen", type_desc="any"),
                             IOSpec(name="remaining", type_desc="list[any]")])
    feasible = _node("Feasibility Check", "Verify the choice maintains feasibility",
                      ConceptType.GREEDY,
                      inputs=[IOSpec(name="chosen", type_desc="any"),
                              IOSpec(name="solution", type_desc="partial solution")],
                      outputs=[IOSpec(name="is_feasible", type_desc="bool")])
    update = _node("Update Solution", "Add the chosen element to the partial solution",
                    ConceptType.GREEDY,
                    inputs=[IOSpec(name="chosen", type_desc="any"),
                            IOSpec(name="solution", type_desc="partial solution")],
                    outputs=[IOSpec(name="solution", type_desc="partial solution")])

    edges = [
        _edge(sort_cands, choose, "sorted", "sorted", "list[any]"),
        _edge(choose, feasible, "chosen", "chosen", "any"),
        _edge(feasible, update, "is_feasible", "chosen", "any"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.GREEDY,
        name="Greedy",
        description="Sort candidates, make greedy choices, check feasibility, update solution",
        template_nodes=[sort_cands, choose, feasible, update],
        template_edges=edges,
        variants=["huffman", "activity_selector", "fractional_knapsack", "prim_mst"],
    )


def _build_graph_traversal() -> SkeletonGraph:
    init = _node("Init Visited", "Initialize visited set and data structures",
                  ConceptType.GRAPH_TRAVERSAL,
                  inputs=[IOSpec(name="graph", type_desc="Graph")],
                  outputs=[IOSpec(name="state", type_desc="traversal state")])
    pick = _node("Pick Next", "Select the next node to visit from the frontier",
                  ConceptType.GRAPH_TRAVERSAL,
                  inputs=[IOSpec(name="state", type_desc="traversal state")],
                  outputs=[IOSpec(name="current", type_desc="node"),
                           IOSpec(name="state", type_desc="traversal state")])
    process = _node("Process Node", "Process the current node (record distance, parent, etc.)",
                     ConceptType.GRAPH_TRAVERSAL,
                     inputs=[IOSpec(name="current", type_desc="node"),
                             IOSpec(name="state", type_desc="traversal state")],
                     outputs=[IOSpec(name="state", type_desc="traversal state")])
    update_frontier = _node("Update Frontier", "Add unvisited neighbors to the frontier",
                             ConceptType.GRAPH_TRAVERSAL,
                             inputs=[IOSpec(name="current", type_desc="node"),
                                     IOSpec(name="state", type_desc="traversal state")],
                             outputs=[IOSpec(name="state", type_desc="traversal state")])
    check = _node("Check Termination", "Check if traversal is complete",
                   ConceptType.GRAPH_TRAVERSAL,
                   inputs=[IOSpec(name="state", type_desc="traversal state")],
                   outputs=[IOSpec(name="done", type_desc="bool"),
                            IOSpec(name="result", type_desc="traversal result")])

    edges = [
        _edge(init, pick, "state", "state", "traversal state"),
        _edge(pick, process, "current", "current", "node"),
        _edge(process, update_frontier, "state", "state", "traversal state"),
        _edge(update_frontier, check, "state", "state", "traversal state"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.GRAPH_TRAVERSAL,
        name="Graph Traversal",
        description="Initialize, pick next, process, update frontier, check termination",
        template_nodes=[init, pick, process, update_frontier, check],
        template_edges=edges,
        variants=["bfs", "dfs", "topological_sort"],
    )


def _build_graph_optimization() -> SkeletonGraph:
    init = _node("Init Weights", "Initialize distance/weight arrays",
                  ConceptType.GRAPH_OPTIMIZATION,
                  inputs=[IOSpec(name="graph", type_desc="weighted Graph"),
                          IOSpec(name="source", type_desc="node")],
                  outputs=[IOSpec(name="distances", type_desc="dict[node, float]")])
    relax = _node("Relax Edges", "Relax edges to find shorter paths",
                   ConceptType.GRAPH_OPTIMIZATION,
                   inputs=[IOSpec(name="distances", type_desc="dict[node, float]"),
                           IOSpec(name="graph", type_desc="weighted Graph")],
                   outputs=[IOSpec(name="distances", type_desc="dict[node, float]")])
    check_neg = _node("Check Negative Cycle", "Detect negative-weight cycles",
                       ConceptType.GRAPH_OPTIMIZATION,
                       inputs=[IOSpec(name="distances", type_desc="dict[node, float]"),
                               IOSpec(name="graph", type_desc="weighted Graph")],
                       outputs=[IOSpec(name="has_negative_cycle", type_desc="bool")])
    extract = _node("Extract Path", "Extract the shortest path from predecessor array",
                     ConceptType.GRAPH_OPTIMIZATION,
                     inputs=[IOSpec(name="distances", type_desc="dict[node, float]"),
                             IOSpec(name="predecessors", type_desc="dict[node, node]")],
                     outputs=[IOSpec(name="path", type_desc="list[node]")])

    edges = [
        _edge(init, relax, "distances", "distances", "dict[node, float]"),
        _edge(relax, check_neg, "distances", "distances", "dict[node, float]"),
        _edge(check_neg, extract, "has_negative_cycle", "distances", "dict[node, float]"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.GRAPH_OPTIMIZATION,
        name="Graph Optimization",
        description="Initialize weights, relax edges, check negative cycles, extract path",
        template_nodes=[init, relax, check_neg, extract],
        template_edges=edges,
        variants=["dijkstra", "bellman_ford", "floyd_warshall"],
    )


def _build_sorting() -> SkeletonGraph:
    compare = _node("Compare", "Compare two elements",
                     ConceptType.SORTING,
                     inputs=[IOSpec(name="a", type_desc="comparable"),
                             IOSpec(name="b", type_desc="comparable")],
                     outputs=[IOSpec(name="order", type_desc="bool")])
    swap = _node("Swap", "Swap elements if out of order",
                  ConceptType.SORTING,
                  inputs=[IOSpec(name="array", type_desc="list[comparable]"),
                          IOSpec(name="i", type_desc="int"),
                          IOSpec(name="j", type_desc="int")],
                  outputs=[IOSpec(name="array", type_desc="list[comparable]")])
    recurse = _node("Recurse/Iterate", "Repeat comparison-swap until sorted",
                     ConceptType.SORTING,
                     inputs=[IOSpec(name="array", type_desc="list[comparable]")],
                     outputs=[IOSpec(name="sorted", type_desc="list[comparable]")])

    edges = [
        _edge(compare, swap, "order", "i", "int"),
        _edge(swap, recurse, "array", "array", "list[comparable]"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SORTING,
        name="Sorting",
        description="Compare elements, swap if needed, recurse or iterate",
        template_nodes=[compare, swap, recurse],
        template_edges=edges,
        variants=["insertion_sort", "heapsort", "quicksort", "merge_sort"],
    )


def _build_string_matching() -> SkeletonGraph:
    preprocess = _node("Preprocess", "Build auxiliary data structure from pattern",
                        ConceptType.STRING_MATCHING,
                        inputs=[IOSpec(name="pattern", type_desc="str")],
                        outputs=[IOSpec(name="table", type_desc="preprocessed data")])
    scan = _node("Scan", "Scan through the text character by character",
                  ConceptType.STRING_MATCHING,
                  inputs=[IOSpec(name="text", type_desc="str"),
                          IOSpec(name="table", type_desc="preprocessed data")],
                  outputs=[IOSpec(name="position", type_desc="int"),
                           IOSpec(name="matched", type_desc="bool")])
    advance = _node("Match/Advance", "On match report position, on mismatch advance using table",
                     ConceptType.STRING_MATCHING,
                     inputs=[IOSpec(name="position", type_desc="int"),
                             IOSpec(name="matched", type_desc="bool")],
                     outputs=[IOSpec(name="matches", type_desc="list[int]")])

    edges = [
        _edge(preprocess, scan, "table", "table", "preprocessed data"),
        _edge(scan, advance, "position", "position", "int"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.STRING_MATCHING,
        name="String Matching",
        description="Preprocess pattern, scan text, match or advance",
        template_nodes=[preprocess, scan, advance],
        template_edges=edges,
        variants=["kmp", "naive_string_matcher", "rabin_karp"],
    )


def _build_searching() -> SkeletonGraph:
    init = _node("Init Bounds", "Set initial search boundaries",
                  ConceptType.SEARCHING,
                  inputs=[IOSpec(name="data", type_desc="sorted list[comparable]"),
                          IOSpec(name="target", type_desc="comparable")],
                  outputs=[IOSpec(name="lo", type_desc="int"),
                           IOSpec(name="hi", type_desc="int")])
    probe = _node("Probe", "Examine the element at the probe position",
                   ConceptType.SEARCHING,
                   inputs=[IOSpec(name="data", type_desc="sorted list[comparable]"),
                           IOSpec(name="lo", type_desc="int"),
                           IOSpec(name="hi", type_desc="int")],
                   outputs=[IOSpec(name="mid", type_desc="int"),
                            IOSpec(name="comparison", type_desc="int")])
    narrow = _node("Narrow", "Narrow the search range based on comparison",
                    ConceptType.SEARCHING,
                    inputs=[IOSpec(name="lo", type_desc="int"),
                            IOSpec(name="hi", type_desc="int"),
                            IOSpec(name="mid", type_desc="int"),
                            IOSpec(name="comparison", type_desc="int")],
                    outputs=[IOSpec(name="lo", type_desc="int"),
                             IOSpec(name="hi", type_desc="int"),
                             IOSpec(name="found", type_desc="bool")])

    edges = [
        _edge(init, probe, "lo", "lo", "int"),
        _edge(probe, narrow, "mid", "mid", "int"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SEARCHING,
        name="Searching",
        description="Initialize bounds, probe, narrow range until found",
        template_nodes=[init, probe, narrow],
        template_edges=edges,
        variants=["binary_search", "linear_search", "interpolation_search"],
    )


def _build_geometry() -> SkeletonGraph:
    preprocess = _node("Preprocess Points", "Sort or prepare geometric input",
                        ConceptType.GEOMETRY,
                        inputs=[IOSpec(name="points", type_desc="list[point]")],
                        outputs=[IOSpec(name="sorted_points", type_desc="list[point]")])
    construct = _node("Construct", "Build geometric structure incrementally",
                       ConceptType.GEOMETRY,
                       inputs=[IOSpec(name="sorted_points", type_desc="list[point]")],
                       outputs=[IOSpec(name="structure", type_desc="geometric structure")])
    verify = _node("Verify Invariant", "Check geometric invariant (e.g., convexity)",
                    ConceptType.GEOMETRY,
                    inputs=[IOSpec(name="structure", type_desc="geometric structure")],
                    outputs=[IOSpec(name="valid", type_desc="bool"),
                             IOSpec(name="result", type_desc="geometric structure")])

    edges = [
        _edge(preprocess, construct, "sorted_points", "sorted_points", "list[point]"),
        _edge(construct, verify, "structure", "structure", "geometric structure"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.GEOMETRY,
        name="Geometry",
        description="Preprocess points, construct structure, verify invariant",
        template_nodes=[preprocess, construct, verify],
        template_edges=edges,
        variants=["convex_hull", "closest_pair", "segment_intersection"],
    )


def _build_number_theory() -> SkeletonGraph:
    reduce = _node("Reduce", "Reduce the problem using modular arithmetic or divisibility",
                    ConceptType.NUMBER_THEORY,
                    inputs=[IOSpec(name="n", type_desc="nat")],
                    outputs=[IOSpec(name="reduced", type_desc="nat")])
    iterate = _node("Iterate", "Apply iterative/recursive number-theoretic step",
                     ConceptType.NUMBER_THEORY,
                     inputs=[IOSpec(name="reduced", type_desc="nat")],
                     outputs=[IOSpec(name="intermediate", type_desc="nat")])
    conclude = _node("Conclude", "Derive final result or prove property",
                      ConceptType.NUMBER_THEORY,
                      inputs=[IOSpec(name="intermediate", type_desc="nat")],
                      outputs=[IOSpec(name="result", type_desc="nat or Prop")])

    edges = [
        _edge(reduce, iterate, "reduced", "reduced", "nat"),
        _edge(iterate, conclude, "intermediate", "intermediate", "nat"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.NUMBER_THEORY,
        name="Number Theory",
        description="Reduce via modular arithmetic, iterate, conclude",
        template_nodes=[reduce, iterate, conclude],
        template_edges=edges,
        variants=["euclidean_gcd", "prime_sieve", "modular_exponentiation"],
    )


def _build_signal_transform() -> SkeletonGraph:
    window = _node("Window", "Apply window function to input signal segment",
                    ConceptType.SIGNAL_TRANSFORM,
                    inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
                    outputs=[IOSpec(name="windowed", type_desc="np.ndarray")])
    forward = _node("Forward Transform", "Apply forward transform (FFT, DCT, etc.)",
                     ConceptType.SIGNAL_TRANSFORM,
                     inputs=[IOSpec(name="windowed", type_desc="np.ndarray")],
                     outputs=[IOSpec(name="spectrum", type_desc="np.ndarray")])
    spectral = _node("Spectral Processing", "Modify spectral coefficients",
                      ConceptType.SIGNAL_TRANSFORM,
                      inputs=[IOSpec(name="spectrum", type_desc="np.ndarray")],
                      outputs=[IOSpec(name="modified_spectrum", type_desc="np.ndarray")])
    inverse = _node("Inverse Transform", "Apply inverse transform to recover signal",
                     ConceptType.SIGNAL_TRANSFORM,
                     inputs=[IOSpec(name="modified_spectrum", type_desc="np.ndarray")],
                     outputs=[IOSpec(name="result", type_desc="np.ndarray")])

    edges = [
        _edge(window, forward, "windowed", "windowed", "np.ndarray"),
        _edge(forward, spectral, "spectrum", "spectrum", "np.ndarray"),
        _edge(spectral, inverse, "modified_spectrum", "modified_spectrum", "np.ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SIGNAL_TRANSFORM,
        name="Signal Transform",
        description="Window input, forward transform, spectral processing, inverse transform",
        template_nodes=[window, forward, spectral, inverse],
        template_edges=edges,
        variants=["fft_filter", "spectral_analysis", "dct_compression", "stft"],
    )


def _build_signal_filter() -> SkeletonGraph:
    design = _node("Design Filter", "Design filter coefficients from specification",
                    ConceptType.SIGNAL_FILTER,
                    inputs=[IOSpec(name="spec", type_desc="filter specification")],
                    outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")])
    validate = _node("Validate Stability", "Check filter stability via pole analysis",
                      ConceptType.SIGNAL_FILTER,
                      inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
                      outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")])
    apply_filt = _node("Apply Filter", "Apply filter to input signal",
                        ConceptType.SIGNAL_FILTER,
                        inputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients"),
                                IOSpec(name="signal", type_desc="np.ndarray")],
                        outputs=[IOSpec(name="filtered", type_desc="np.ndarray")])
    freq_resp = _node("Frequency Response", "Compute and inspect frequency response",
                       ConceptType.SIGNAL_FILTER,
                       inputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
                       outputs=[IOSpec(name="response", type_desc="tuple[np.ndarray, np.ndarray]")])

    edges = [
        _edge(design, validate, "coefficients", "coefficients", "filter coefficients"),
        _edge(validate, apply_filt, "valid_coefficients", "valid_coefficients", "filter coefficients"),
        _edge(validate, freq_resp, "valid_coefficients", "valid_coefficients", "filter coefficients"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SIGNAL_FILTER,
        name="Signal Filter",
        description="Design filter, validate stability, apply filter / compute frequency response",
        template_nodes=[design, validate, apply_filt, freq_resp],
        template_edges=edges,
        variants=["butterworth_lowpass", "chebyshev_bandpass", "fir_bandpass", "notch_filter"],
    )


def _build_graph_signal_processing() -> SkeletonGraph:
    build = _node("Build Graph", "Construct weighted adjacency matrix from data",
                   ConceptType.GRAPH_SIGNAL_PROCESSING,
                   inputs=[IOSpec(name="data", type_desc="any")],
                   outputs=[IOSpec(name="W", type_desc="sparse matrix")])
    laplacian = _node("Compute Laplacian", "Compute graph Laplacian from adjacency",
                       ConceptType.GRAPH_SIGNAL_PROCESSING,
                       inputs=[IOSpec(name="W", type_desc="sparse matrix")],
                       outputs=[IOSpec(name="L", type_desc="sparse matrix")])
    gft = _node("GFT", "Compute Graph Fourier Transform",
                 ConceptType.GRAPH_SIGNAL_PROCESSING,
                 inputs=[IOSpec(name="L", type_desc="sparse matrix"),
                         IOSpec(name="signal", type_desc="np.ndarray")],
                 outputs=[IOSpec(name="spectrum", type_desc="np.ndarray"),
                          IOSpec(name="eigenvectors", type_desc="np.ndarray")])
    graph_filter = _node("Graph Filter/Diffuse", "Apply spectral filter or heat diffusion",
                          ConceptType.GRAPH_SIGNAL_PROCESSING,
                          inputs=[IOSpec(name="spectrum", type_desc="np.ndarray"),
                                  IOSpec(name="eigenvectors", type_desc="np.ndarray"),
                                  IOSpec(name="L", type_desc="sparse matrix")],
                          outputs=[IOSpec(name="result", type_desc="np.ndarray")])

    edges = [
        _edge(build, laplacian, "W", "W", "sparse matrix"),
        _edge(laplacian, gft, "L", "L", "sparse matrix"),
        _edge(gft, graph_filter, "spectrum", "spectrum", "np.ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.GRAPH_SIGNAL_PROCESSING,
        name="Graph Signal Processing",
        description="Build graph, compute Laplacian, GFT, apply graph filter or diffusion",
        template_nodes=[build, laplacian, gft, graph_filter],
        template_edges=edges,
        variants=["graph_lowpass", "heat_diffusion", "graph_denoising", "community_detection"],
    )


# ---------------------------------------------------------------------------
# Bayesian / probabilistic inference skeletons
# ---------------------------------------------------------------------------


def _build_mcmc_hmc() -> SkeletonGraph:
    """HMC skeleton: Init -> Leapfrog loop -> Acceptance."""
    init = _node(
        "Initialization Subgraph",
        "Initialize position q_0, momentum p_0, step size epsilon, and mass matrix M",
        ConceptType.MCMC_KERNEL,
        inputs=[IOSpec(name="log_density", type_desc="Callable[[ndarray], float]"),
                IOSpec(name="initial_params", type_desc="ndarray")],
        outputs=[IOSpec(name="q", type_desc="ndarray"),
                 IOSpec(name="p", type_desc="ndarray"),
                 IOSpec(name="epsilon", type_desc="float"),
                 IOSpec(name="mass_matrix", type_desc="ndarray")],
    )
    half_step_p1 = _node(
        "Half Step Momentum Start",
        "Half-step momentum update: p <- p - (epsilon/2) * grad(log_density)(q)",
        ConceptType.MCMC_KERNEL,
        inputs=[IOSpec(name="p", type_desc="ndarray"),
                IOSpec(name="q", type_desc="ndarray"),
                IOSpec(name="epsilon", type_desc="float"),
                IOSpec(name="log_density", type_desc="Callable[[ndarray], float]")],
        outputs=[IOSpec(name="p_half", type_desc="ndarray")],
    )
    full_step_q = _node(
        "Full Step Position",
        "Full-step position update: q <- q + epsilon * M^{-1} * p",
        ConceptType.MCMC_KERNEL,
        inputs=[IOSpec(name="q", type_desc="ndarray"),
                IOSpec(name="p_half", type_desc="ndarray"),
                IOSpec(name="epsilon", type_desc="float"),
                IOSpec(name="mass_matrix", type_desc="ndarray")],
        outputs=[IOSpec(name="q_new", type_desc="ndarray")],
    )
    oracle_query = _node(
        "Oracle Query",
        "Stateless log-density and gradient evaluation at proposed position (Oracle Isolation enforced)",
        ConceptType.PROBABILISTIC_ORACLE,
        inputs=[IOSpec(name="q_new", type_desc="ndarray"),
                IOSpec(name="log_density", type_desc="Callable[[ndarray], float]")],
        outputs=[IOSpec(name="log_prob", type_desc="float"),
                 IOSpec(name="grad", type_desc="ndarray")],
    )
    half_step_p2 = _node(
        "Half Step Momentum End",
        "Half-step momentum update: p <- p_half - (epsilon/2) * grad",
        ConceptType.MCMC_KERNEL,
        inputs=[IOSpec(name="p_half", type_desc="ndarray"),
                IOSpec(name="grad", type_desc="ndarray"),
                IOSpec(name="epsilon", type_desc="float")],
        outputs=[IOSpec(name="p_new", type_desc="ndarray")],
    )
    accept = _node(
        "Acceptance Criterion",
        "Metropolis-Hastings accept/reject based on Hamiltonian energy difference",
        ConceptType.MCMC_KERNEL,
        inputs=[IOSpec(name="q", type_desc="ndarray"),
                IOSpec(name="q_new", type_desc="ndarray"),
                IOSpec(name="p", type_desc="ndarray"),
                IOSpec(name="p_new", type_desc="ndarray"),
                IOSpec(name="log_prob", type_desc="float")],
        outputs=[IOSpec(name="accepted_q", type_desc="ndarray"),
                 IOSpec(name="accepted", type_desc="bool")],
    )

    edges = [
        _edge(init, half_step_p1, "q", "q", "ndarray"),
        _edge(init, half_step_p1, "p", "p", "ndarray"),
        _edge(init, half_step_p1, "epsilon", "epsilon", "float"),
        _edge(half_step_p1, full_step_q, "p_half", "p_half", "ndarray"),
        _edge(init, full_step_q, "epsilon", "epsilon", "float"),
        _edge(init, full_step_q, "mass_matrix", "mass_matrix", "ndarray"),
        _edge(full_step_q, oracle_query, "q_new", "q_new", "ndarray"),
        _edge(oracle_query, half_step_p2, "grad", "grad", "ndarray"),
        _edge(half_step_p1, half_step_p2, "p_half", "p_half", "ndarray"),
        _edge(init, half_step_p2, "epsilon", "epsilon", "float"),
        _edge(full_step_q, accept, "q_new", "q_new", "ndarray"),
        _edge(half_step_p2, accept, "p_new", "p_new", "ndarray"),
        _edge(oracle_query, accept, "log_prob", "log_prob", "float"),
        _edge(init, accept, "q", "q", "ndarray"),
        _edge(init, accept, "p", "p", "ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.MCMC_KERNEL,
        name="MCMC HMC",
        description=(
            "Hamiltonian Monte Carlo: initialize, run leapfrog integrator "
            "(half-step momentum, full-step position, oracle query, half-step momentum), "
            "then Metropolis accept/reject"
        ),
        template_nodes=[init, half_step_p1, full_step_q, oracle_query, half_step_p2, accept],
        template_edges=edges,
        variants=["hmc", "nuts", "leapfrog_integrator"],
    )


def _build_vi_advi_deterministic() -> SkeletonGraph:
    """ADVI deterministic skeleton: Shape Alloc -> Reparam -> ELBO -> L-BFGS."""
    shape_alloc = _node(
        "Shape Alloc",
        "Allocate variational parameter arrays (mu, log_sigma) with correct shapes",
        ConceptType.VI_ELBO,
        inputs=[IOSpec(name="model_dims", type_desc="dict[str, tuple[int, ...]]")],
        outputs=[IOSpec(name="mu", type_desc="ndarray"),
                 IOSpec(name="log_sigma", type_desc="ndarray")],
    )
    reparam = _node(
        "Reparameterization",
        "Reparameterization trick: theta = mu + exp(log_sigma) * z, where z is static noise input",
        ConceptType.VI_ELBO,
        inputs=[IOSpec(name="mu", type_desc="ndarray"),
                IOSpec(name="log_sigma", type_desc="ndarray"),
                IOSpec(name="z", type_desc="ndarray", constraints="static standard normal noise")],
        outputs=[IOSpec(name="theta", type_desc="ndarray")],
    )
    elbo_eval = _node(
        "ELBO Eval",
        "Evaluate Evidence Lower Bound: E[log p(x,theta)] - E[log q(theta)] (Oracle Isolation: stateless)",
        ConceptType.PROBABILISTIC_ORACLE,
        inputs=[IOSpec(name="theta", type_desc="ndarray"),
                IOSpec(name="log_density", type_desc="Callable[[ndarray], float]")],
        outputs=[IOSpec(name="elbo", type_desc="float"),
                 IOSpec(name="elbo_grad", type_desc="ndarray")],
    )
    optimizer = _node(
        "L-BFGS Optimizer",
        "L-BFGS optimization step to update mu and log_sigma (State Decoupling: curvature state flows via edge)",
        ConceptType.VI_ELBO,
        inputs=[IOSpec(name="elbo_grad", type_desc="ndarray"),
                IOSpec(name="mu", type_desc="ndarray"),
                IOSpec(name="log_sigma", type_desc="ndarray"),
                IOSpec(name="lbfgs_state", type_desc="LBFGSState",
                       constraints="history of gradient differences and step vectors")],
        outputs=[IOSpec(name="mu_updated", type_desc="ndarray"),
                 IOSpec(name="log_sigma_updated", type_desc="ndarray"),
                 IOSpec(name="lbfgs_state", type_desc="LBFGSState")],
    )

    edges = [
        _edge(shape_alloc, reparam, "mu", "mu", "ndarray"),
        _edge(shape_alloc, reparam, "log_sigma", "log_sigma", "ndarray"),
        _edge(reparam, elbo_eval, "theta", "theta", "ndarray"),
        _edge(elbo_eval, optimizer, "elbo_grad", "elbo_grad", "ndarray"),
        _edge(shape_alloc, optimizer, "mu", "mu", "ndarray"),
        _edge(shape_alloc, optimizer, "log_sigma", "log_sigma", "ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.VI_ELBO,
        name="VI ADVI Deterministic",
        description=(
            "Automatic Differentiation Variational Inference: allocate shapes, "
            "reparameterize with static noise, evaluate ELBO, optimize with L-BFGS"
        ),
        template_nodes=[shape_alloc, reparam, elbo_eval, optimizer],
        template_edges=edges,
        variants=["advi", "meanfield_vi", "fullrank_vi"],
    )


def _build_particle_filter() -> SkeletonGraph:
    """Particle filter skeleton with parallelizable Predict step."""
    preprocess = _node(
        "Preprocess",
        "Compute effective sample size (ESS) and resample particles if below threshold",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[IOSpec(name="particles", type_desc="ndarray"),
                IOSpec(name="weights", type_desc="ndarray")],
        outputs=[IOSpec(name="resampled_particles", type_desc="ndarray"),
                 IOSpec(name="ess", type_desc="float")],
    )
    predict = _node(
        "Predict",
        "Propagate each particle through transition model (embarrassingly parallel)",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[IOSpec(name="resampled_particles", type_desc="ndarray"),
                IOSpec(name="transition_model", type_desc="Callable[[ndarray], ndarray]")],
        outputs=[IOSpec(name="predicted_particles", type_desc="ndarray")],
        parallelizable=True,
    )
    reweight = _node(
        "Reweight",
        "Compute log-likelihood of observation for each particle and update weights (Oracle Isolation: stateless)",
        ConceptType.PROBABILISTIC_ORACLE,
        inputs=[IOSpec(name="predicted_particles", type_desc="ndarray"),
                IOSpec(name="observation", type_desc="ndarray"),
                IOSpec(name="log_likelihood", type_desc="Callable[[ndarray, ndarray], float]")],
        outputs=[IOSpec(name="log_weights", type_desc="ndarray")],
    )
    postprocess = _node(
        "Postprocess",
        "Normalize weights, compute posterior mean/covariance, package state estimate",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[IOSpec(name="predicted_particles", type_desc="ndarray"),
                IOSpec(name="log_weights", type_desc="ndarray")],
        outputs=[IOSpec(name="particles", type_desc="ndarray"),
                 IOSpec(name="weights", type_desc="ndarray"),
                 IOSpec(name="state_estimate", type_desc="ndarray")],
    )

    edges = [
        _edge(preprocess, predict, "resampled_particles", "resampled_particles", "ndarray"),
        _edge(predict, reweight, "predicted_particles", "predicted_particles", "ndarray"),
        _edge(reweight, postprocess, "log_weights", "log_weights", "ndarray"),
        _edge(predict, postprocess, "predicted_particles", "predicted_particles", "ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SEQUENTIAL_FILTER,
        name="Particle Filter",
        description=(
            "Sequential Monte Carlo: preprocess (ESS/resample), predict (parallel), "
            "reweight via log-likelihood, postprocess"
        ),
        template_nodes=[preprocess, predict, reweight, postprocess],
        template_edges=edges,
        variants=["bootstrap_particle_filter", "auxiliary_particle_filter", "rao_blackwellized_pf"],
    )


def _build_kalman_filter() -> SkeletonGraph:
    """Kalman filter skeleton: bipartite Predict/Update subgraphs."""
    predict_state = _node(
        "Predict State",
        "Propagate state mean: x_pred = F @ x + B @ u",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[IOSpec(name="x", type_desc="ndarray"),
                IOSpec(name="F", type_desc="ndarray"),
                IOSpec(name="u", type_desc="ndarray"),
                IOSpec(name="B", type_desc="ndarray")],
        outputs=[IOSpec(name="x_pred", type_desc="ndarray")],
    )
    predict_cov = _node(
        "Predict Covariance",
        "Propagate covariance: P_pred = F @ P @ F^T + Q (State Decoupling: P flows explicitly)",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[IOSpec(name="P", type_desc="ndarray"),
                IOSpec(name="F", type_desc="ndarray"),
                IOSpec(name="Q", type_desc="ndarray")],
        outputs=[IOSpec(name="P_pred", type_desc="ndarray")],
    )
    innovation = _node(
        "Innovation",
        "Compute innovation (measurement residual): y_tilde = z - H @ x_pred",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[IOSpec(name="z", type_desc="ndarray"),
                IOSpec(name="x_pred", type_desc="ndarray"),
                IOSpec(name="H", type_desc="ndarray")],
        outputs=[IOSpec(name="y_tilde", type_desc="ndarray")],
    )
    kalman_gain = _node(
        "Kalman Gain",
        "Compute Kalman gain: K = P_pred @ H^T @ (H @ P_pred @ H^T + R)^{-1}",
        ConceptType.CONJUGATE_UPDATE,
        inputs=[IOSpec(name="P_pred", type_desc="ndarray"),
                IOSpec(name="H", type_desc="ndarray"),
                IOSpec(name="R", type_desc="ndarray")],
        outputs=[IOSpec(name="K", type_desc="ndarray"),
                 IOSpec(name="S", type_desc="ndarray")],
    )
    update_state = _node(
        "Update State",
        "Update state estimate: x = x_pred + K @ y_tilde",
        ConceptType.CONJUGATE_UPDATE,
        inputs=[IOSpec(name="x_pred", type_desc="ndarray"),
                IOSpec(name="K", type_desc="ndarray"),
                IOSpec(name="y_tilde", type_desc="ndarray")],
        outputs=[IOSpec(name="x_updated", type_desc="ndarray")],
    )
    update_cov = _node(
        "Update Covariance",
        "Update covariance: P = (I - K @ H) @ P_pred (State Decoupling: P flows explicitly)",
        ConceptType.CONJUGATE_UPDATE,
        inputs=[IOSpec(name="K", type_desc="ndarray"),
                IOSpec(name="H", type_desc="ndarray"),
                IOSpec(name="P_pred", type_desc="ndarray")],
        outputs=[IOSpec(name="P_updated", type_desc="ndarray")],
    )

    edges = [
        # Predict subgraph -> Update subgraph (bipartite)
        _edge(predict_state, innovation, "x_pred", "x_pred", "ndarray"),
        _edge(predict_cov, kalman_gain, "P_pred", "P_pred", "ndarray"),
        _edge(innovation, update_state, "y_tilde", "y_tilde", "ndarray"),
        _edge(kalman_gain, update_state, "K", "K", "ndarray"),
        _edge(kalman_gain, update_cov, "K", "K", "ndarray"),
        _edge(predict_state, update_state, "x_pred", "x_pred", "ndarray"),
        _edge(predict_cov, update_cov, "P_pred", "P_pred", "ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SEQUENTIAL_FILTER,
        name="Kalman Filter",
        description=(
            "Kalman filter with bipartite predict/update subgraphs: "
            "predict state and covariance, then compute innovation, Kalman gain, "
            "and update state and covariance"
        ),
        template_nodes=[predict_state, predict_cov, innovation, kalman_gain, update_state, update_cov],
        template_edges=edges,
        variants=["kalman_filter", "extended_kalman_filter", "unscented_kalman_filter"],
    )


def _build_belief_propagation() -> SkeletonGraph:
    """Belief propagation skeleton with memoization state to prevent cyclic failures."""
    var_to_factor = _node(
        "Variable to Factor",
        "Compute messages from variable nodes to factor nodes: product of incoming messages excluding recipient",
        ConceptType.MESSAGE_PASSING,
        inputs=[IOSpec(name="incoming_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="memo_state", type_desc="dict[str, ndarray]")],
        outputs=[IOSpec(name="var_messages", type_desc="dict[str, ndarray]")],
    )
    factor_to_var = _node(
        "Factor to Variable",
        "Compute messages from factor nodes to variable nodes: marginalize factor over all variables except recipient",
        ConceptType.MESSAGE_PASSING,
        inputs=[IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="factor_potentials", type_desc="dict[str, ndarray]"),
                IOSpec(name="memo_state", type_desc="dict[str, ndarray]")],
        outputs=[IOSpec(name="factor_messages", type_desc="dict[str, ndarray]")],
    )
    marginal = _node(
        "Marginal Computation",
        "Compute marginal beliefs by combining all incoming messages at each variable node",
        ConceptType.MESSAGE_PASSING,
        inputs=[IOSpec(name="factor_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="var_messages", type_desc="dict[str, ndarray]")],
        outputs=[IOSpec(name="marginals", type_desc="dict[str, ndarray]")],
    )
    memo = _node(
        "Memoization State",
        "Cache previous-iteration messages to detect convergence and prevent cyclic recomputation during toposort",
        ConceptType.MESSAGE_PASSING,
        inputs=[IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="factor_messages", type_desc="dict[str, ndarray]")],
        outputs=[IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
                 IOSpec(name="converged", type_desc="bool")],
    )

    edges = [
        _edge(var_to_factor, factor_to_var, "var_messages", "var_messages", "dict[str, ndarray]"),
        _edge(factor_to_var, marginal, "factor_messages", "factor_messages", "dict[str, ndarray]"),
        _edge(var_to_factor, marginal, "var_messages", "var_messages", "dict[str, ndarray]"),
        # Memoization: receives both message types, emits state back for next iteration
        _edge(var_to_factor, memo, "var_messages", "var_messages", "dict[str, ndarray]"),
        _edge(factor_to_var, memo, "factor_messages", "factor_messages", "dict[str, ndarray]"),
        _edge(memo, var_to_factor, "memo_state", "memo_state", "dict[str, ndarray]"),
        _edge(memo, factor_to_var, "memo_state", "memo_state", "dict[str, ndarray]"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.MESSAGE_PASSING,
        name="Belief Propagation",
        description=(
            "Sum-product belief propagation: variable-to-factor messages, "
            "factor-to-variable messages, marginal computation. "
            "Memoization state node prevents cyclic failures during toposort."
        ),
        template_nodes=[var_to_factor, factor_to_var, marginal, memo],
        template_edges=edges,
        variants=["sum_product", "max_product", "loopy_bp"],
    )


# Registry of all skeleton templates
SKELETON_TEMPLATES: dict[ConceptType, SkeletonGraph] = {
    ConceptType.DIVIDE_AND_CONQUER: _build_divide_and_conquer(),
    ConceptType.DYNAMIC_PROGRAMMING: _build_dynamic_programming(),
    ConceptType.GREEDY: _build_greedy(),
    ConceptType.GRAPH_TRAVERSAL: _build_graph_traversal(),
    ConceptType.GRAPH_OPTIMIZATION: _build_graph_optimization(),
    ConceptType.SORTING: _build_sorting(),
    ConceptType.STRING_MATCHING: _build_string_matching(),
    ConceptType.SEARCHING: _build_searching(),
    ConceptType.GEOMETRY: _build_geometry(),
    ConceptType.NUMBER_THEORY: _build_number_theory(),
    ConceptType.SIGNAL_TRANSFORM: _build_signal_transform(),
    ConceptType.SIGNAL_FILTER: _build_signal_filter(),
    ConceptType.GRAPH_SIGNAL_PROCESSING: _build_graph_signal_processing(),
    ConceptType.MCMC_KERNEL: _build_mcmc_hmc(),
    ConceptType.VI_ELBO: _build_vi_advi_deterministic(),
    ConceptType.SEQUENTIAL_FILTER: _build_particle_filter(),
    ConceptType.MESSAGE_PASSING: _build_belief_propagation(),
}


# Named skeleton variants for paradigms with multiple topologies
NAMED_SKELETONS: dict[str, SkeletonGraph] = {
    "kalman_filter": _build_kalman_filter(),
    "particle_filter": SKELETON_TEMPLATES[ConceptType.SEQUENTIAL_FILTER],
    "hmc": SKELETON_TEMPLATES[ConceptType.MCMC_KERNEL],
    "advi": SKELETON_TEMPLATES[ConceptType.VI_ELBO],
    "belief_propagation": SKELETON_TEMPLATES[ConceptType.MESSAGE_PASSING],
}


def get_skeleton(
    concept_type: ConceptType,
    *,
    variant: str | None = None,
) -> SkeletonGraph | None:
    """Look up the skeleton template for a paradigm.

    If *variant* is given, check ``NAMED_SKELETONS`` first so that
    paradigms with multiple topologies (e.g. ``SEQUENTIAL_FILTER`` has
    both particle-filter and kalman-filter) can be disambiguated.
    """
    if variant and variant in NAMED_SKELETONS:
        return NAMED_SKELETONS[variant]
    return SKELETON_TEMPLATES.get(concept_type)


def instantiate_skeleton(
    skeleton: SkeletonGraph,
    goal_desc: str,
    *,
    parent_id: str | None = None,
    base_depth: int = 0,
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Create concrete CDG nodes from a skeleton template.

    Generates fresh node_ids and wires up parent/children relationships.
    Node descriptions are prefixed with the goal context.

    Returns:
        (nodes, edges) — ready to insert into the CDG.
    """
    # Map old template node_ids to fresh ones
    id_map: dict[str, str] = {}
    nodes: list[AlgorithmicNode] = []

    for tpl_node in skeleton.template_nodes:
        new_id = f"{tpl_node.node_id}_{uuid.uuid4().hex[:8]}"
        id_map[tpl_node.node_id] = new_id

        node = AlgorithmicNode(
            node_id=new_id,
            parent_id=parent_id,
            name=tpl_node.name,
            description=f"[{goal_desc}] {tpl_node.description}",
            concept_type=tpl_node.concept_type,
            inputs=tpl_node.inputs,
            outputs=tpl_node.outputs,
            status=NodeStatus.PENDING,
            depth=base_depth + tpl_node.depth,
            parallelizable=tpl_node.parallelizable,
        )
        nodes.append(node)

    # Remap edges
    edges: list[DependencyEdge] = []
    for tpl_edge in skeleton.template_edges:
        edge = DependencyEdge(
            source_id=id_map[tpl_edge.source_id],
            target_id=id_map[tpl_edge.target_id],
            output_name=tpl_edge.output_name,
            input_name=tpl_edge.input_name,
            source_type=tpl_edge.source_type,
            target_type=tpl_edge.target_type,
            requires_glue=tpl_edge.requires_glue,
        )
        edges.append(edge)

    return nodes, edges
