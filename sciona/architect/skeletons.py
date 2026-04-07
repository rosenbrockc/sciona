"""Pre-built skeleton graphs for algorithmic paradigms.

Each skeleton is a template CDG that can be instantiated for a specific goal,
giving the Decomposer a head start on structure.
"""

from __future__ import annotations

import uuid

from sciona.architect.models import (
    AlgorithmicNode,
    BaselineAnalyzerSpec,
    BaselineComponentShape,
    BaselinePredictorAliasSpec,
    BaselineStageSpec,
    BaselineWindowSpec,
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
    matched_primitive: str | None = None,
    status: NodeStatus = NodeStatus.PENDING,
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
        status=status,
        parallelizable=parallelizable,
        matched_primitive=matched_primitive,
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


def infer_boundary_ports(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> tuple[list[IOSpec], list[IOSpec]]:
    """Infer external graph inputs/outputs from unresolved node ports."""
    incoming = {(edge.target_id, edge.input_name) for edge in edges}
    outgoing = {(edge.source_id, edge.output_name) for edge in edges}

    inputs: list[IOSpec] = []
    outputs: list[IOSpec] = []
    seen_inputs: set[str] = set()
    seen_outputs: set[str] = set()

    for node in nodes:
        for port in node.inputs:
            key = (node.node_id, port.name)
            if key in incoming or port.name in seen_inputs:
                continue
            inputs.append(port.model_copy())
            seen_inputs.add(port.name)
        for port in node.outputs:
            key = (node.node_id, port.name)
            if key in outgoing or port.name in seen_outputs:
                continue
            outputs.append(port.model_copy())
            seen_outputs.add(port.name)

    return inputs, outputs


def _build_divide_and_conquer() -> SkeletonGraph:
    split = _node(
        "Split",
        "Divide the input into sub-problems",
        ConceptType.DIVIDE_AND_CONQUER,
        inputs=[IOSpec(name="input", type_desc="any")],
        outputs=[
            IOSpec(name="left", type_desc="any"),
            IOSpec(name="right", type_desc="any"),
        ],
    )
    recurse_left = _node(
        "Recurse Left",
        "Solve the left sub-problem recursively",
        ConceptType.DIVIDE_AND_CONQUER,
        inputs=[IOSpec(name="subproblem", type_desc="any")],
        outputs=[IOSpec(name="result", type_desc="any")],
    )
    recurse_right = _node(
        "Recurse Right",
        "Solve the right sub-problem recursively",
        ConceptType.DIVIDE_AND_CONQUER,
        inputs=[IOSpec(name="subproblem", type_desc="any")],
        outputs=[IOSpec(name="result", type_desc="any")],
    )
    merge = _node(
        "Merge",
        "Combine results of sub-problems",
        ConceptType.DIVIDE_AND_CONQUER,
        inputs=[
            IOSpec(name="left_result", type_desc="any"),
            IOSpec(name="right_result", type_desc="any"),
        ],
        outputs=[IOSpec(name="result", type_desc="any")],
    )

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
    define = _node(
        "Define Subproblems",
        "Define the subproblem structure and table dimensions",
        ConceptType.DYNAMIC_PROGRAMMING,
        inputs=[IOSpec(name="input", type_desc="any")],
        outputs=[IOSpec(name="table_def", type_desc="table structure")],
    )
    base = _node(
        "Base Case",
        "Initialize base cases in the DP table",
        ConceptType.DYNAMIC_PROGRAMMING,
        inputs=[IOSpec(name="table_def", type_desc="table structure")],
        outputs=[IOSpec(name="init_table", type_desc="partially filled table")],
    )
    recurrence = _node(
        "Recurrence",
        "Fill the DP table using the recurrence relation",
        ConceptType.DYNAMIC_PROGRAMMING,
        inputs=[IOSpec(name="init_table", type_desc="partially filled table")],
        outputs=[IOSpec(name="filled_table", type_desc="completed table")],
    )
    memoize = _node(
        "Memoize",
        "Store and reuse computed subproblem solutions",
        ConceptType.DYNAMIC_PROGRAMMING,
        inputs=[IOSpec(name="filled_table", type_desc="completed table")],
        outputs=[IOSpec(name="memo_table", type_desc="memoized table")],
    )
    extract = _node(
        "Extract Solution",
        "Extract the final answer from the completed table",
        ConceptType.DYNAMIC_PROGRAMMING,
        inputs=[IOSpec(name="memo_table", type_desc="memoized table")],
        outputs=[IOSpec(name="result", type_desc="any")],
    )

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
    sort_cands = _node(
        "Sort Candidates",
        "Order candidates by greedy criterion",
        ConceptType.GREEDY,
        inputs=[IOSpec(name="candidates", type_desc="list[any]")],
        outputs=[IOSpec(name="sorted", type_desc="list[any]")],
    )
    choose = _node(
        "Greedy Choice",
        "Select the locally optimal candidate",
        ConceptType.GREEDY,
        inputs=[IOSpec(name="sorted", type_desc="list[any]")],
        outputs=[
            IOSpec(name="chosen", type_desc="any"),
            IOSpec(name="remaining", type_desc="list[any]"),
        ],
    )
    feasible = _node(
        "Feasibility Check",
        "Verify the choice maintains feasibility",
        ConceptType.GREEDY,
        inputs=[
            IOSpec(name="chosen", type_desc="any"),
            IOSpec(name="solution", type_desc="partial solution"),
        ],
        outputs=[IOSpec(name="is_feasible", type_desc="bool")],
    )
    update = _node(
        "Update Solution",
        "Add the chosen element to the partial solution",
        ConceptType.GREEDY,
        inputs=[
            IOSpec(name="chosen", type_desc="any"),
            IOSpec(name="solution", type_desc="partial solution"),
        ],
        outputs=[IOSpec(name="solution", type_desc="partial solution")],
    )

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
    init = _node(
        "Init Visited",
        "Initialize visited set and data structures",
        ConceptType.GRAPH_TRAVERSAL,
        inputs=[IOSpec(name="graph", type_desc="Graph")],
        outputs=[IOSpec(name="state", type_desc="traversal state")],
    )
    pick = _node(
        "Pick Next",
        "Select the next node to visit from the frontier",
        ConceptType.GRAPH_TRAVERSAL,
        inputs=[IOSpec(name="state", type_desc="traversal state")],
        outputs=[
            IOSpec(name="current", type_desc="node"),
            IOSpec(name="state", type_desc="traversal state"),
        ],
    )
    process = _node(
        "Process Node",
        "Process the current node (record distance, parent, etc.)",
        ConceptType.GRAPH_TRAVERSAL,
        inputs=[
            IOSpec(name="current", type_desc="node"),
            IOSpec(name="state", type_desc="traversal state"),
        ],
        outputs=[IOSpec(name="state", type_desc="traversal state")],
    )
    update_frontier = _node(
        "Update Frontier",
        "Add unvisited neighbors to the frontier",
        ConceptType.GRAPH_TRAVERSAL,
        inputs=[
            IOSpec(name="current", type_desc="node"),
            IOSpec(name="state", type_desc="traversal state"),
        ],
        outputs=[IOSpec(name="state", type_desc="traversal state")],
    )
    check = _node(
        "Check Termination",
        "Check if traversal is complete",
        ConceptType.GRAPH_TRAVERSAL,
        inputs=[IOSpec(name="state", type_desc="traversal state")],
        outputs=[
            IOSpec(name="done", type_desc="bool"),
            IOSpec(name="result", type_desc="traversal result"),
        ],
    )

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
    init = _node(
        "Init Weights",
        "Initialize distance/weight arrays",
        ConceptType.GRAPH_OPTIMIZATION,
        inputs=[
            IOSpec(name="graph", type_desc="weighted Graph"),
            IOSpec(name="source", type_desc="node"),
        ],
        outputs=[IOSpec(name="distances", type_desc="dict[node, float]")],
    )
    relax = _node(
        "Relax Edges",
        "Relax edges to find shorter paths",
        ConceptType.GRAPH_OPTIMIZATION,
        inputs=[
            IOSpec(name="distances", type_desc="dict[node, float]"),
            IOSpec(name="graph", type_desc="weighted Graph"),
        ],
        outputs=[IOSpec(name="distances", type_desc="dict[node, float]")],
    )
    check_neg = _node(
        "Check Negative Cycle",
        "Detect negative-weight cycles",
        ConceptType.GRAPH_OPTIMIZATION,
        inputs=[
            IOSpec(name="distances", type_desc="dict[node, float]"),
            IOSpec(name="graph", type_desc="weighted Graph"),
        ],
        outputs=[IOSpec(name="has_negative_cycle", type_desc="bool")],
    )
    extract = _node(
        "Extract Path",
        "Extract the shortest path from predecessor array",
        ConceptType.GRAPH_OPTIMIZATION,
        inputs=[
            IOSpec(name="distances", type_desc="dict[node, float]"),
            IOSpec(name="predecessors", type_desc="dict[node, node]"),
        ],
        outputs=[IOSpec(name="path", type_desc="list[node]")],
    )

    edges = [
        _edge(init, relax, "distances", "distances", "dict[node, float]"),
        _edge(relax, check_neg, "distances", "distances", "dict[node, float]"),
        _edge(
            check_neg, extract, "has_negative_cycle", "distances", "dict[node, float]"
        ),
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
    compare = _node(
        "Compare",
        "Compare two elements",
        ConceptType.SORTING,
        inputs=[
            IOSpec(name="a", type_desc="comparable"),
            IOSpec(name="b", type_desc="comparable"),
        ],
        outputs=[IOSpec(name="order", type_desc="bool")],
    )
    swap = _node(
        "Swap",
        "Swap elements if out of order",
        ConceptType.SORTING,
        inputs=[
            IOSpec(name="array", type_desc="list[comparable]"),
            IOSpec(name="i", type_desc="int"),
            IOSpec(name="j", type_desc="int"),
        ],
        outputs=[IOSpec(name="array", type_desc="list[comparable]")],
    )
    recurse = _node(
        "Recurse/Iterate",
        "Repeat comparison-swap until sorted",
        ConceptType.SORTING,
        inputs=[IOSpec(name="array", type_desc="list[comparable]")],
        outputs=[IOSpec(name="sorted", type_desc="list[comparable]")],
    )

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
    preprocess = _node(
        "Preprocess",
        "Build auxiliary data structure from pattern",
        ConceptType.STRING_MATCHING,
        inputs=[IOSpec(name="pattern", type_desc="str")],
        outputs=[IOSpec(name="table", type_desc="preprocessed data")],
    )
    scan = _node(
        "Scan",
        "Scan through the text character by character",
        ConceptType.STRING_MATCHING,
        inputs=[
            IOSpec(name="text", type_desc="str"),
            IOSpec(name="table", type_desc="preprocessed data"),
        ],
        outputs=[
            IOSpec(name="position", type_desc="int"),
            IOSpec(name="matched", type_desc="bool"),
        ],
    )
    advance = _node(
        "Match/Advance",
        "On match report position, on mismatch advance using table",
        ConceptType.STRING_MATCHING,
        inputs=[
            IOSpec(name="position", type_desc="int"),
            IOSpec(name="matched", type_desc="bool"),
        ],
        outputs=[IOSpec(name="matches", type_desc="list[int]")],
    )

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
    init = _node(
        "Init Bounds",
        "Set initial search boundaries",
        ConceptType.SEARCHING,
        inputs=[
            IOSpec(name="data", type_desc="sorted list[comparable]"),
            IOSpec(name="target", type_desc="comparable"),
        ],
        outputs=[
            IOSpec(name="lo", type_desc="int"),
            IOSpec(name="hi", type_desc="int"),
        ],
    )
    probe = _node(
        "Probe",
        "Examine the element at the probe position",
        ConceptType.SEARCHING,
        inputs=[
            IOSpec(name="data", type_desc="sorted list[comparable]"),
            IOSpec(name="lo", type_desc="int"),
            IOSpec(name="hi", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="mid", type_desc="int"),
            IOSpec(name="comparison", type_desc="int"),
        ],
    )
    narrow = _node(
        "Narrow",
        "Narrow the search range based on comparison",
        ConceptType.SEARCHING,
        inputs=[
            IOSpec(name="lo", type_desc="int"),
            IOSpec(name="hi", type_desc="int"),
            IOSpec(name="mid", type_desc="int"),
            IOSpec(name="comparison", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="lo", type_desc="int"),
            IOSpec(name="hi", type_desc="int"),
            IOSpec(name="found", type_desc="bool"),
        ],
    )

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
    preprocess = _node(
        "Preprocess Points",
        "Sort or prepare geometric input",
        ConceptType.GEOMETRY,
        inputs=[IOSpec(name="points", type_desc="list[point]")],
        outputs=[IOSpec(name="sorted_points", type_desc="list[point]")],
    )
    construct = _node(
        "Construct",
        "Build geometric structure incrementally",
        ConceptType.GEOMETRY,
        inputs=[IOSpec(name="sorted_points", type_desc="list[point]")],
        outputs=[IOSpec(name="structure", type_desc="geometric structure")],
    )
    verify = _node(
        "Verify Invariant",
        "Check geometric invariant (e.g., convexity)",
        ConceptType.GEOMETRY,
        inputs=[IOSpec(name="structure", type_desc="geometric structure")],
        outputs=[
            IOSpec(name="valid", type_desc="bool"),
            IOSpec(name="result", type_desc="geometric structure"),
        ],
    )

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
    reduce = _node(
        "Reduce",
        "Reduce the problem using modular arithmetic or divisibility",
        ConceptType.NUMBER_THEORY,
        inputs=[IOSpec(name="n", type_desc="nat")],
        outputs=[IOSpec(name="reduced", type_desc="nat")],
    )
    iterate = _node(
        "Iterate",
        "Apply iterative/recursive number-theoretic step",
        ConceptType.NUMBER_THEORY,
        inputs=[IOSpec(name="reduced", type_desc="nat")],
        outputs=[IOSpec(name="intermediate", type_desc="nat")],
    )
    conclude = _node(
        "Conclude",
        "Derive final result or prove property",
        ConceptType.NUMBER_THEORY,
        inputs=[IOSpec(name="intermediate", type_desc="nat")],
        outputs=[IOSpec(name="result", type_desc="nat or Prop")],
    )

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
    window = _node(
        "Window",
        "Apply window function to input signal segment",
        ConceptType.SIGNAL_TRANSFORM,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="windowed", type_desc="np.ndarray")],
    )
    forward = _node(
        "Forward Transform",
        "Apply forward transform (FFT, DCT, etc.)",
        ConceptType.SIGNAL_TRANSFORM,
        inputs=[IOSpec(name="windowed", type_desc="np.ndarray")],
        outputs=[IOSpec(name="spectrum", type_desc="np.ndarray")],
    )
    spectral = _node(
        "Spectral Processing",
        "Modify spectral coefficients",
        ConceptType.SIGNAL_TRANSFORM,
        inputs=[IOSpec(name="spectrum", type_desc="np.ndarray")],
        outputs=[IOSpec(name="modified_spectrum", type_desc="np.ndarray")],
    )
    inverse = _node(
        "Inverse Transform",
        "Apply inverse transform to recover signal",
        ConceptType.SIGNAL_TRANSFORM,
        inputs=[IOSpec(name="modified_spectrum", type_desc="np.ndarray")],
        outputs=[IOSpec(name="result", type_desc="np.ndarray")],
    )

    edges = [
        _edge(window, forward, "windowed", "windowed", "np.ndarray"),
        _edge(forward, spectral, "spectrum", "spectrum", "np.ndarray"),
        _edge(
            spectral, inverse, "modified_spectrum", "modified_spectrum", "np.ndarray"
        ),
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
    design = _node(
        "Design Filter",
        "Design filter coefficients from specification",
        ConceptType.SIGNAL_FILTER,
        inputs=[IOSpec(name="spec", type_desc="filter specification")],
        outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
    )
    validate = _node(
        "Validate Stability",
        "Check filter stability via pole analysis",
        ConceptType.SIGNAL_FILTER,
        inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
        outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
    )
    apply_filt = _node(
        "Apply Filter",
        "Apply filter to input signal",
        ConceptType.SIGNAL_FILTER,
        inputs=[
            IOSpec(name="valid_coefficients", type_desc="filter coefficients"),
            IOSpec(name="signal", type_desc="np.ndarray"),
        ],
        outputs=[IOSpec(name="filtered", type_desc="np.ndarray")],
    )
    freq_resp = _node(
        "Frequency Response",
        "Compute and inspect frequency response",
        ConceptType.SIGNAL_FILTER,
        inputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
        outputs=[IOSpec(name="response", type_desc="tuple[np.ndarray, np.ndarray]")],
    )

    edges = [
        _edge(design, validate, "coefficients", "coefficients", "filter coefficients"),
        _edge(
            validate,
            apply_filt,
            "valid_coefficients",
            "valid_coefficients",
            "filter coefficients",
        ),
        _edge(
            validate,
            freq_resp,
            "valid_coefficients",
            "valid_coefficients",
            "filter coefficients",
        ),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SIGNAL_FILTER,
        name="Signal Filter",
        description="Design filter, validate stability, apply filter / compute frequency response",
        template_nodes=[design, validate, apply_filt, freq_resp],
        template_edges=edges,
        variants=[
            "butterworth_lowpass",
            "chebyshev_bandpass",
            "fir_bandpass",
            "notch_filter",
        ],
    )


def _build_signal_detect_measure() -> SkeletonGraph:
    preprocess = _node(
        "Filter Signal For Detection",
        "Filter or denoise the raw signal into a cleaned trace suitable for downstream event detection.",
        ConceptType.SIGNAL_FILTER,
        inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="conditioned_signal", type_desc="np.ndarray")],
        matched_primitive="filter_signal_for_detection",
    )
    detect = _node(
        "Detect Peaks In Signal",
        "Detect salient peaks or event locations in the conditioned signal.",
        ConceptType.DATA_EXTRACTION,
        inputs=[
            IOSpec(name="conditioned_signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="events", type_desc="np.ndarray")],
        matched_primitive="detect_peaks_in_signal",
    )
    compute = _node(
        "Compute Event Rate",
        "Compute a target rate or cadence from inter-event intervals in the detected event sequence.",
        ConceptType.ANALYSIS,
        inputs=[
            IOSpec(name="events", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="rate", type_desc="tuple[np.ndarray, np.ndarray]")],
        matched_primitive="compute_event_rate",
    )

    edges = [
        _edge(
            preprocess,
            detect,
            "conditioned_signal",
            "conditioned_signal",
            "np.ndarray",
        ),
        _edge(
            detect,
            compute,
            "events",
            "events",
            "np.ndarray",
        ),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.SIGNAL_FILTER,
        name="Signal Detect and Measure",
        description=(
            "Condition a raw signal, detect salient events, and compute a downstream "
            "rate or cadence from those events"
        ),
        template_nodes=[preprocess, detect, compute],
        template_edges=edges,
        variants=[
            "signal_detect_measure",
            "feature_detection_metric",
            "event_rate_estimation",
        ],
    )


def _build_graph_signal_processing() -> SkeletonGraph:
    build = _node(
        "Build Graph",
        "Construct weighted adjacency matrix from data",
        ConceptType.GRAPH_SIGNAL_PROCESSING,
        inputs=[IOSpec(name="data", type_desc="any")],
        outputs=[IOSpec(name="W", type_desc="sparse matrix")],
    )
    laplacian = _node(
        "Compute Laplacian",
        "Compute graph Laplacian from adjacency",
        ConceptType.GRAPH_SIGNAL_PROCESSING,
        inputs=[IOSpec(name="W", type_desc="sparse matrix")],
        outputs=[IOSpec(name="L", type_desc="sparse matrix")],
    )
    gft = _node(
        "GFT",
        "Compute Graph Fourier Transform",
        ConceptType.GRAPH_SIGNAL_PROCESSING,
        inputs=[
            IOSpec(name="L", type_desc="sparse matrix"),
            IOSpec(name="signal", type_desc="np.ndarray"),
        ],
        outputs=[
            IOSpec(name="spectrum", type_desc="np.ndarray"),
            IOSpec(name="eigenvectors", type_desc="np.ndarray"),
        ],
    )
    graph_filter = _node(
        "Graph Filter/Diffuse",
        "Apply spectral filter or heat diffusion",
        ConceptType.GRAPH_SIGNAL_PROCESSING,
        inputs=[
            IOSpec(name="spectrum", type_desc="np.ndarray"),
            IOSpec(name="eigenvectors", type_desc="np.ndarray"),
            IOSpec(name="L", type_desc="sparse matrix"),
        ],
        outputs=[IOSpec(name="result", type_desc="np.ndarray")],
    )

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
        variants=[
            "graph_lowpass",
            "heat_diffusion",
            "graph_denoising",
            "community_detection",
        ],
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
        inputs=[
            IOSpec(name="log_density", type_desc="Callable[[ndarray], float]"),
            IOSpec(name="initial_params", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="q", type_desc="ndarray"),
            IOSpec(name="p", type_desc="ndarray"),
            IOSpec(name="epsilon", type_desc="float"),
            IOSpec(name="mass_matrix", type_desc="ndarray"),
        ],
    )
    half_step_p1 = _node(
        "Half Step Momentum Start",
        "Half-step momentum update: p <- p - (epsilon/2) * grad(log_density)(q)",
        ConceptType.MCMC_KERNEL,
        inputs=[
            IOSpec(name="p", type_desc="ndarray"),
            IOSpec(name="q", type_desc="ndarray"),
            IOSpec(name="epsilon", type_desc="float"),
            IOSpec(name="log_density", type_desc="Callable[[ndarray], float]"),
        ],
        outputs=[IOSpec(name="p_half", type_desc="ndarray")],
    )
    full_step_q = _node(
        "Full Step Position",
        "Full-step position update: q <- q + epsilon * M^{-1} * p",
        ConceptType.MCMC_KERNEL,
        inputs=[
            IOSpec(name="q", type_desc="ndarray"),
            IOSpec(name="p_half", type_desc="ndarray"),
            IOSpec(name="epsilon", type_desc="float"),
            IOSpec(name="mass_matrix", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="q_new", type_desc="ndarray")],
    )
    oracle_query = _node(
        "Oracle Query",
        "Stateless log-density and gradient evaluation at proposed position (Oracle Isolation enforced)",
        ConceptType.PROBABILISTIC_ORACLE,
        inputs=[
            IOSpec(name="q_new", type_desc="ndarray"),
            IOSpec(name="log_density", type_desc="Callable[[ndarray], float]"),
        ],
        outputs=[
            IOSpec(name="log_prob", type_desc="float"),
            IOSpec(name="grad", type_desc="ndarray"),
        ],
    )
    half_step_p2 = _node(
        "Half Step Momentum End",
        "Half-step momentum update: p <- p_half - (epsilon/2) * grad",
        ConceptType.MCMC_KERNEL,
        inputs=[
            IOSpec(name="p_half", type_desc="ndarray"),
            IOSpec(name="grad", type_desc="ndarray"),
            IOSpec(name="epsilon", type_desc="float"),
        ],
        outputs=[IOSpec(name="p_new", type_desc="ndarray")],
    )
    accept = _node(
        "Acceptance Criterion",
        "Metropolis-Hastings accept/reject based on Hamiltonian energy difference",
        ConceptType.MCMC_KERNEL,
        inputs=[
            IOSpec(name="q", type_desc="ndarray"),
            IOSpec(name="q_new", type_desc="ndarray"),
            IOSpec(name="p", type_desc="ndarray"),
            IOSpec(name="p_new", type_desc="ndarray"),
            IOSpec(name="log_prob", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="accepted_q", type_desc="ndarray"),
            IOSpec(name="accepted", type_desc="bool"),
        ],
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
        template_nodes=[
            init,
            half_step_p1,
            full_step_q,
            oracle_query,
            half_step_p2,
            accept,
        ],
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
        outputs=[
            IOSpec(name="mu", type_desc="ndarray"),
            IOSpec(name="log_sigma", type_desc="ndarray"),
        ],
    )
    reparam = _node(
        "Reparameterization",
        "Reparameterization trick: theta = mu + exp(log_sigma) * z, where z is static noise input",
        ConceptType.VI_ELBO,
        inputs=[
            IOSpec(name="mu", type_desc="ndarray"),
            IOSpec(name="log_sigma", type_desc="ndarray"),
            IOSpec(
                name="z",
                type_desc="ndarray",
                constraints="static standard normal noise",
            ),
        ],
        outputs=[IOSpec(name="theta", type_desc="ndarray")],
    )
    elbo_eval = _node(
        "ELBO Eval",
        "Evaluate Evidence Lower Bound: E[log p(x,theta)] - E[log q(theta)] (Oracle Isolation: stateless)",
        ConceptType.PROBABILISTIC_ORACLE,
        inputs=[
            IOSpec(name="theta", type_desc="ndarray"),
            IOSpec(name="log_density", type_desc="Callable[[ndarray], float]"),
        ],
        outputs=[
            IOSpec(name="elbo", type_desc="float"),
            IOSpec(name="elbo_grad", type_desc="ndarray"),
        ],
    )
    optimizer = _node(
        "L-BFGS Optimizer",
        "L-BFGS optimization step to update mu and log_sigma (State Decoupling: curvature state flows via edge)",
        ConceptType.VI_ELBO,
        inputs=[
            IOSpec(name="elbo_grad", type_desc="ndarray"),
            IOSpec(name="mu", type_desc="ndarray"),
            IOSpec(name="log_sigma", type_desc="ndarray"),
            IOSpec(
                name="lbfgs_state",
                type_desc="LBFGSState",
                constraints="history of gradient differences and step vectors",
            ),
        ],
        outputs=[
            IOSpec(name="mu_updated", type_desc="ndarray"),
            IOSpec(name="log_sigma_updated", type_desc="ndarray"),
            IOSpec(name="lbfgs_state", type_desc="LBFGSState"),
        ],
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
        inputs=[
            IOSpec(name="particles", type_desc="ndarray"),
            IOSpec(name="weights", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="resampled_particles", type_desc="ndarray"),
            IOSpec(name="ess", type_desc="float"),
        ],
    )
    predict = _node(
        "Predict",
        "Propagate each particle through transition model (embarrassingly parallel)",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[
            IOSpec(name="resampled_particles", type_desc="ndarray"),
            IOSpec(name="transition_model", type_desc="Callable[[ndarray], ndarray]"),
        ],
        outputs=[IOSpec(name="predicted_particles", type_desc="ndarray")],
        parallelizable=True,
    )
    reweight = _node(
        "Reweight",
        "Compute log-likelihood of observation for each particle and update weights (Oracle Isolation: stateless)",
        ConceptType.PROBABILISTIC_ORACLE,
        inputs=[
            IOSpec(name="predicted_particles", type_desc="ndarray"),
            IOSpec(name="observation", type_desc="ndarray"),
            IOSpec(
                name="log_likelihood", type_desc="Callable[[ndarray, ndarray], float]"
            ),
        ],
        outputs=[IOSpec(name="log_weights", type_desc="ndarray")],
    )
    postprocess = _node(
        "Postprocess",
        "Normalize weights, compute posterior mean/covariance, package state estimate",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[
            IOSpec(name="predicted_particles", type_desc="ndarray"),
            IOSpec(name="log_weights", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="particles", type_desc="ndarray"),
            IOSpec(name="weights", type_desc="ndarray"),
            IOSpec(name="state_estimate", type_desc="ndarray"),
        ],
    )

    edges = [
        _edge(
            preprocess, predict, "resampled_particles", "resampled_particles", "ndarray"
        ),
        _edge(
            predict, reweight, "predicted_particles", "predicted_particles", "ndarray"
        ),
        _edge(reweight, postprocess, "log_weights", "log_weights", "ndarray"),
        _edge(
            predict,
            postprocess,
            "predicted_particles",
            "predicted_particles",
            "ndarray",
        ),
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
        variants=[
            "bootstrap_particle_filter",
            "auxiliary_particle_filter",
            "rao_blackwellized_pf",
        ],
    )


def _build_kalman_filter() -> SkeletonGraph:
    """Kalman filter skeleton: bipartite Predict/Update subgraphs."""
    predict_state = _node(
        "Predict State",
        "Propagate state mean: x_pred = F @ x + B @ u",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[
            IOSpec(name="x", type_desc="ndarray"),
            IOSpec(name="F", type_desc="ndarray"),
            IOSpec(name="u", type_desc="ndarray"),
            IOSpec(name="B", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="x_pred", type_desc="ndarray")],
    )
    predict_cov = _node(
        "Predict Covariance",
        "Propagate covariance: P_pred = F @ P @ F^T + Q (State Decoupling: P flows explicitly)",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[
            IOSpec(name="P", type_desc="ndarray"),
            IOSpec(name="F", type_desc="ndarray"),
            IOSpec(name="Q", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="P_pred", type_desc="ndarray")],
    )
    innovation = _node(
        "Innovation",
        "Compute innovation (measurement residual): y_tilde = z - H @ x_pred",
        ConceptType.SEQUENTIAL_FILTER,
        inputs=[
            IOSpec(name="z", type_desc="ndarray"),
            IOSpec(name="x_pred", type_desc="ndarray"),
            IOSpec(name="H", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="y_tilde", type_desc="ndarray")],
    )
    kalman_gain = _node(
        "Kalman Gain",
        "Compute Kalman gain: K = P_pred @ H^T @ (H @ P_pred @ H^T + R)^{-1}",
        ConceptType.CONJUGATE_UPDATE,
        inputs=[
            IOSpec(name="P_pred", type_desc="ndarray"),
            IOSpec(name="H", type_desc="ndarray"),
            IOSpec(name="R", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="K", type_desc="ndarray"),
            IOSpec(name="S", type_desc="ndarray"),
        ],
    )
    update_state = _node(
        "Update State",
        "Update state estimate: x = x_pred + K @ y_tilde",
        ConceptType.CONJUGATE_UPDATE,
        inputs=[
            IOSpec(name="x_pred", type_desc="ndarray"),
            IOSpec(name="K", type_desc="ndarray"),
            IOSpec(name="y_tilde", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="x_updated", type_desc="ndarray")],
    )
    update_cov = _node(
        "Update Covariance",
        "Update covariance: P = (I - K @ H) @ P_pred (State Decoupling: P flows explicitly)",
        ConceptType.CONJUGATE_UPDATE,
        inputs=[
            IOSpec(name="K", type_desc="ndarray"),
            IOSpec(name="H", type_desc="ndarray"),
            IOSpec(name="P_pred", type_desc="ndarray"),
        ],
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
        template_nodes=[
            predict_state,
            predict_cov,
            innovation,
            kalman_gain,
            update_state,
            update_cov,
        ],
        template_edges=edges,
        variants=["kalman_filter", "extended_kalman_filter", "unscented_kalman_filter"],
    )


def _build_belief_propagation() -> SkeletonGraph:
    """Belief propagation skeleton with memoization state to prevent cyclic failures."""
    var_to_factor = _node(
        "Variable to Factor",
        "Compute messages from variable nodes to factor nodes: product of incoming messages excluding recipient",
        ConceptType.MESSAGE_PASSING,
        inputs=[
            IOSpec(name="incoming_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
        ],
        outputs=[IOSpec(name="var_messages", type_desc="dict[str, ndarray]")],
    )
    factor_to_var = _node(
        "Factor to Variable",
        "Compute messages from factor nodes to variable nodes: marginalize factor over all variables except recipient",
        ConceptType.MESSAGE_PASSING,
        inputs=[
            IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="factor_potentials", type_desc="dict[str, ndarray]"),
            IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
        ],
        outputs=[IOSpec(name="factor_messages", type_desc="dict[str, ndarray]")],
    )
    marginal = _node(
        "Marginal Computation",
        "Compute marginal beliefs by combining all incoming messages at each variable node",
        ConceptType.MESSAGE_PASSING,
        inputs=[
            IOSpec(name="factor_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
        ],
        outputs=[IOSpec(name="marginals", type_desc="dict[str, ndarray]")],
    )
    memo = _node(
        "Memoization State",
        "Cache previous-iteration messages to detect convergence and prevent cyclic recomputation during toposort",
        ConceptType.MESSAGE_PASSING,
        inputs=[
            IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="factor_messages", type_desc="dict[str, ndarray]"),
        ],
        outputs=[
            IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
            IOSpec(name="converged", type_desc="bool"),
        ],
    )

    edges = [
        _edge(
            var_to_factor,
            factor_to_var,
            "var_messages",
            "var_messages",
            "dict[str, ndarray]",
        ),
        _edge(
            factor_to_var,
            marginal,
            "factor_messages",
            "factor_messages",
            "dict[str, ndarray]",
        ),
        _edge(
            var_to_factor,
            marginal,
            "var_messages",
            "var_messages",
            "dict[str, ndarray]",
        ),
        # Memoization: receives both message types, emits state back for next iteration
        _edge(
            var_to_factor, memo, "var_messages", "var_messages", "dict[str, ndarray]"
        ),
        _edge(
            factor_to_var,
            memo,
            "factor_messages",
            "factor_messages",
            "dict[str, ndarray]",
        ),
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


def _build_linear_algebra() -> SkeletonGraph:
    factorize = _node(
        "Factorize",
        "Decompose the matrix into factors (LU, QR, Cholesky, SVD)",
        ConceptType.ALGEBRA,
        inputs=[IOSpec(name="A", type_desc="ndarray")],
        outputs=[IOSpec(name="factors", type_desc="tuple[ndarray, ...]")],
    )
    solve = _node(
        "Solve/Transform",
        "Solve the system or apply the transformation using the factors",
        ConceptType.ALGEBRA,
        inputs=[IOSpec(name="factors", type_desc="tuple[ndarray, ...]"),
                IOSpec(name="b", type_desc="ndarray")],
        outputs=[IOSpec(name="x", type_desc="ndarray")],
    )
    validate = _node(
        "Validate",
        "Check residual, orthogonality, or reconstruction accuracy",
        ConceptType.ALGEBRA,
        inputs=[IOSpec(name="x", type_desc="ndarray"),
                IOSpec(name="A", type_desc="ndarray")],
        outputs=[IOSpec(name="residual", type_desc="float")],
    )
    edges = [
        _edge(factorize, solve, "factors", "factors", "tuple[ndarray, ...]"),
        _edge(solve, validate, "x", "x", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.ALGEBRA,
        name="Linear Algebra",
        description="Factorize matrix, solve/transform, validate result",
        template_nodes=[factorize, solve, validate],
        template_edges=edges,
        variants=["lu_decomposition", "qr_decomposition", "cholesky", "svd",
                  "eigendecomposition"],
    )


def _build_optimization() -> SkeletonGraph:
    initialize = _node(
        "Initialize",
        "Set initial parameters, learning rate, and optimizer state",
        ConceptType.OPTIMIZATION,
        inputs=[IOSpec(name="x0", type_desc="ndarray")],
        outputs=[IOSpec(name="params", type_desc="ndarray")],
    )
    gradient = _node(
        "Compute Gradient",
        "Evaluate the objective gradient at current parameters",
        ConceptType.OPTIMIZATION,
        inputs=[IOSpec(name="params", type_desc="ndarray")],
        outputs=[IOSpec(name="grad", type_desc="ndarray")],
    )
    update = _node(
        "Update Parameters",
        "Apply the optimization step (gradient descent, Newton, L-BFGS)",
        ConceptType.OPTIMIZATION,
        inputs=[IOSpec(name="params", type_desc="ndarray"),
                IOSpec(name="grad", type_desc="ndarray")],
        outputs=[IOSpec(name="params_new", type_desc="ndarray")],
    )
    converge = _node(
        "Check Convergence",
        "Evaluate stopping criteria (gradient norm, objective change)",
        ConceptType.OPTIMIZATION,
        inputs=[IOSpec(name="params_new", type_desc="ndarray")],
        outputs=[IOSpec(name="converged", type_desc="bool")],
    )
    edges = [
        _edge(initialize, gradient, "params", "params", "ndarray"),
        _edge(gradient, update, "grad", "grad", "ndarray"),
        _edge(update, converge, "params_new", "params_new", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.OPTIMIZATION,
        name="Continuous Optimization",
        description="Initialize, compute gradient, update parameters, check convergence",
        template_nodes=[initialize, gradient, update, converge],
        template_edges=edges,
        variants=["gradient_descent", "newton_method", "lbfgs",
                  "conjugate_gradient", "nelder_mead"],
    )


def _build_combinatorics() -> SkeletonGraph:
    bound = _node(
        "Bound",
        "Compute upper/lower bound on the objective for the current subproblem",
        ConceptType.COMBINATORICS,
        inputs=[IOSpec(name="subproblem", type_desc="any")],
        outputs=[IOSpec(name="bound", type_desc="float")],
    )
    branch = _node(
        "Branch",
        "Split the subproblem into smaller subproblems",
        ConceptType.COMBINATORICS,
        inputs=[IOSpec(name="subproblem", type_desc="any")],
        outputs=[IOSpec(name="children", type_desc="list[any]")],
    )
    prune = _node(
        "Prune",
        "Eliminate subproblems that cannot improve the best known solution",
        ConceptType.COMBINATORICS,
        inputs=[IOSpec(name="children", type_desc="list[any]"),
                IOSpec(name="bound", type_desc="float")],
        outputs=[IOSpec(name="surviving", type_desc="list[any]")],
    )
    select = _node(
        "Select",
        "Choose the next subproblem to explore or return the best solution",
        ConceptType.COMBINATORICS,
        inputs=[IOSpec(name="surviving", type_desc="list[any]")],
        outputs=[IOSpec(name="solution", type_desc="any")],
    )
    edges = [
        _edge(bound, branch, "bound", "subproblem", "any"),
        _edge(branch, prune, "children", "children", "list[any]"),
        _edge(prune, select, "surviving", "surviving", "list[any]"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.COMBINATORICS,
        name="Combinatorial Optimization",
        description="Bound, branch, prune, select — exact/approximate discrete optimization",
        template_nodes=[bound, branch, prune, select],
        template_edges=edges,
        variants=["branch_and_bound", "constraint_propagation", "sat_solver",
                  "integer_programming"],
    )


def _build_neural_network() -> SkeletonGraph:
    forward = _node(
        "Forward Pass",
        "Compute predictions by propagating inputs through network layers",
        ConceptType.NEURAL_NETWORK,
        inputs=[IOSpec(name="inputs", type_desc="ndarray"),
                IOSpec(name="weights", type_desc="list[ndarray]")],
        outputs=[IOSpec(name="activations", type_desc="ndarray")],
    )
    loss = _node(
        "Loss Computation",
        "Compute scalar loss comparing predictions to targets",
        ConceptType.NEURAL_NETWORK,
        inputs=[IOSpec(name="activations", type_desc="ndarray"),
                IOSpec(name="targets", type_desc="ndarray")],
        outputs=[IOSpec(name="loss", type_desc="float")],
    )
    backward = _node(
        "Backward Pass",
        "Compute gradients of loss with respect to all parameters via backpropagation",
        ConceptType.NEURAL_NETWORK,
        inputs=[IOSpec(name="loss", type_desc="float")],
        outputs=[IOSpec(name="gradients", type_desc="list[ndarray]")],
    )
    update = _node(
        "Parameter Update",
        "Update network weights using computed gradients",
        ConceptType.NEURAL_NETWORK,
        inputs=[IOSpec(name="gradients", type_desc="list[ndarray]"),
                IOSpec(name="weights", type_desc="list[ndarray]")],
        outputs=[IOSpec(name="weights_new", type_desc="list[ndarray]")],
    )
    edges = [
        _edge(forward, loss, "activations", "activations", "ndarray"),
        _edge(loss, backward, "loss", "loss", "float"),
        _edge(backward, update, "gradients", "gradients", "list[ndarray]"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.NEURAL_NETWORK,
        name="Neural Network",
        description="Forward pass, loss computation, backward pass, parameter update",
        template_nodes=[forward, loss, backward, update],
        template_edges=edges,
        variants=["mlp", "cnn", "rnn", "transformer", "autoencoder"],
    )


def _build_clustering() -> SkeletonGraph:
    init = _node(
        "Initialize Centers",
        "Initialize cluster centers (random, k-means++, or given)",
        ConceptType.CLUSTERING,
        inputs=[IOSpec(name="data", type_desc="ndarray")],
        outputs=[IOSpec(name="centers", type_desc="ndarray")],
    )
    assign = _node(
        "Assign Points",
        "Assign each data point to the nearest cluster center",
        ConceptType.CLUSTERING,
        inputs=[IOSpec(name="data", type_desc="ndarray"),
                IOSpec(name="centers", type_desc="ndarray")],
        outputs=[IOSpec(name="assignments", type_desc="ndarray")],
    )
    update = _node(
        "Update Centers",
        "Recompute cluster centers from current assignments",
        ConceptType.CLUSTERING,
        inputs=[IOSpec(name="data", type_desc="ndarray"),
                IOSpec(name="assignments", type_desc="ndarray")],
        outputs=[IOSpec(name="centers_new", type_desc="ndarray")],
    )
    edges = [
        _edge(init, assign, "centers", "centers", "ndarray"),
        _edge(assign, update, "assignments", "assignments", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.CLUSTERING,
        name="Clustering",
        description="Initialize centers, assign points, update centers — iterative refinement",
        template_nodes=[init, assign, update],
        template_edges=edges,
        variants=["kmeans", "kmedoids", "em_gmm", "spectral_clustering"],
    )


def _build_dimensionality_reduction() -> SkeletonGraph:
    center = _node(
        "Center/Scale",
        "Center and optionally scale the data matrix",
        ConceptType.DIMENSIONALITY_REDUCTION,
        inputs=[IOSpec(name="X", type_desc="ndarray")],
        outputs=[IOSpec(name="X_centered", type_desc="ndarray")],
    )
    project = _node(
        "Project",
        "Project data onto lower-dimensional subspace",
        ConceptType.DIMENSIONALITY_REDUCTION,
        inputs=[IOSpec(name="X_centered", type_desc="ndarray")],
        outputs=[IOSpec(name="X_projected", type_desc="ndarray"),
                 IOSpec(name="components", type_desc="ndarray")],
    )
    validate = _node(
        "Validate Reconstruction",
        "Validate reconstruction quality and information preservation",
        ConceptType.DIMENSIONALITY_REDUCTION,
        inputs=[IOSpec(name="X_projected", type_desc="ndarray"),
                IOSpec(name="components", type_desc="ndarray")],
        outputs=[IOSpec(name="reconstruction_error", type_desc="float")],
    )
    edges = [
        _edge(center, project, "X_centered", "X_centered", "ndarray"),
        _edge(project, validate, "X_projected", "X_projected", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.DIMENSIONALITY_REDUCTION,
        name="Dimensionality Reduction",
        description="Center/scale data, project to lower dimension, validate reconstruction",
        template_nodes=[center, project, validate],
        template_edges=edges,
        variants=["pca", "svd_truncated", "tsne", "umap", "kernel_pca"],
    )


def _build_ode_solver() -> SkeletonGraph:
    evaluate = _node(
        "Evaluate Derivative",
        "Evaluate the derivative function at the current state and time.",
        ConceptType.ODE_SOLVER,
        inputs=[
            IOSpec(name="t", type_desc="float"),
            IOSpec(name="y", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="dy", type_desc="ndarray")],
    )
    advance = _node(
        "Advance State",
        "Advance the state using the current derivative estimate and step size.",
        ConceptType.ODE_SOLVER,
        inputs=[
            IOSpec(name="y", type_desc="ndarray"),
            IOSpec(name="dy", type_desc="ndarray"),
            IOSpec(name="h", type_desc="float"),
        ],
        outputs=[IOSpec(name="y_new", type_desc="ndarray")],
    )
    error = _node(
        "Estimate Error",
        "Estimate local truncation error for the candidate step.",
        ConceptType.ODE_SOLVER,
        inputs=[IOSpec(name="y_new", type_desc="ndarray")],
        outputs=[IOSpec(name="error_estimate", type_desc="float")],
    )
    adapt = _node(
        "Adapt Step Size",
        "Adapt the step size based on the local error estimate and acceptance test.",
        ConceptType.ODE_SOLVER,
        inputs=[
            IOSpec(name="error_estimate", type_desc="float"),
            IOSpec(name="h", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="h_new", type_desc="float"),
            IOSpec(name="accepted", type_desc="bool"),
        ],
    )
    edges = [
        _edge(evaluate, advance, "dy", "dy", "ndarray"),
        _edge(advance, error, "y_new", "y_new", "ndarray"),
        _edge(error, adapt, "error_estimate", "error_estimate", "float"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.ODE_SOLVER,
        name="ODE Solver",
        description="Evaluate derivative, advance state, estimate error, and adapt the step size.",
        template_nodes=[evaluate, advance, error, adapt],
        template_edges=edges,
        variants=["euler", "runge_kutta_4", "dormand_prince", "bdf", "adams_bashforth"],
    )


def _build_quadrature() -> SkeletonGraph:
    sample = _node(
        "Sample Points",
        "Generate quadrature sample points and weights over the integration domain.",
        ConceptType.QUADRATURE,
        inputs=[IOSpec(name="domain", type_desc="tuple[float, float]")],
        outputs=[
            IOSpec(name="points", type_desc="ndarray"),
            IOSpec(name="weights", type_desc="ndarray"),
        ],
    )
    evaluate = _node(
        "Evaluate Integrand",
        "Evaluate the integrand at the sampled points.",
        ConceptType.QUADRATURE,
        inputs=[IOSpec(name="points", type_desc="ndarray")],
        outputs=[IOSpec(name="values", type_desc="ndarray")],
    )
    refine = _node(
        "Estimate Error/Refine",
        "Aggregate weighted values, estimate quadrature error, and decide whether to refine.",
        ConceptType.QUADRATURE,
        inputs=[
            IOSpec(name="values", type_desc="ndarray"),
            IOSpec(name="weights", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="integral", type_desc="float"),
            IOSpec(name="error_estimate", type_desc="float"),
        ],
    )
    edges = [
        _edge(sample, evaluate, "points", "points", "ndarray"),
        _edge(evaluate, refine, "values", "values", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.QUADRATURE,
        name="Quadrature",
        description="Sample points, evaluate the integrand, and estimate integral/error adaptively.",
        template_nodes=[sample, evaluate, refine],
        template_edges=edges,
        variants=["trapezoidal", "simpsons", "gauss_legendre", "monte_carlo_integration", "adaptive_quadrature"],
    )


def _build_randomized() -> SkeletonGraph:
    generate = _node(
        "Generate Samples",
        "Generate randomized samples or projections from the input data.",
        ConceptType.RANDOMIZED,
        inputs=[IOSpec(name="data", type_desc="ndarray")],
        outputs=[IOSpec(name="samples", type_desc="ndarray")],
    )
    sketch = _node(
        "Sketch/Hash",
        "Hash or sketch the sampled data into a compact randomized summary.",
        ConceptType.RANDOMIZED,
        inputs=[IOSpec(name="samples", type_desc="ndarray")],
        outputs=[IOSpec(name="sketch", type_desc="ndarray")],
    )
    estimate = _node(
        "Estimate",
        "Estimate the target quantity from the randomized sketch.",
        ConceptType.RANDOMIZED,
        inputs=[IOSpec(name="sketch", type_desc="ndarray")],
        outputs=[IOSpec(name="estimate", type_desc="ndarray")],
    )
    edges = [
        _edge(generate, sketch, "samples", "samples", "ndarray"),
        _edge(sketch, estimate, "sketch", "sketch", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.RANDOMIZED,
        name="Randomized",
        description="Generate samples, sketch/hash them, and estimate the target quantity.",
        template_nodes=[generate, sketch, estimate],
        template_edges=edges,
        variants=["reservoir_sampling", "count_min_sketch", "locality_sensitive_hashing", "random_projection", "importance_sampling"],
    )


def _build_information_theory() -> SkeletonGraph:
    estimate = _node(
        "Estimate Distribution",
        "Estimate the underlying distribution or empirical mass function from data.",
        ConceptType.INFORMATION_THEORY,
        inputs=[IOSpec(name="samples", type_desc="ndarray")],
        outputs=[IOSpec(name="probabilities", type_desc="ndarray")],
    )
    compute = _node(
        "Compute Entropy/Divergence",
        "Compute entropy, divergence, or related information-theoretic quantities.",
        ConceptType.INFORMATION_THEORY,
        inputs=[IOSpec(name="probabilities", type_desc="ndarray")],
        outputs=[IOSpec(name="information_value", type_desc="float")],
    )
    validate = _node(
        "Validate Bounds",
        "Validate information-theoretic inequalities and numerical bounds.",
        ConceptType.INFORMATION_THEORY,
        inputs=[IOSpec(name="information_value", type_desc="float")],
        outputs=[IOSpec(name="validated_value", type_desc="float")],
    )
    edges = [
        _edge(estimate, compute, "probabilities", "probabilities", "ndarray"),
        _edge(compute, validate, "information_value", "information_value", "float"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.INFORMATION_THEORY,
        name="Information Theory",
        description="Estimate distributions, compute entropy/divergence, and validate information-theoretic bounds.",
        template_nodes=[estimate, compute, validate],
        template_edges=edges,
        variants=["entropy_estimation", "kl_divergence", "mutual_information", "rate_distortion"],
    )


def _build_compression() -> SkeletonGraph:
    model = _node(
        "Model Source",
        "Model the source distribution or structure before encoding.",
        ConceptType.COMPRESSION,
        inputs=[IOSpec(name="symbols", type_desc="ndarray")],
        outputs=[IOSpec(name="model", type_desc="ndarray")],
    )
    encode = _node(
        "Encode",
        "Encode the source symbols using the learned or specified source model.",
        ConceptType.COMPRESSION,
        inputs=[IOSpec(name="model", type_desc="ndarray")],
        outputs=[IOSpec(name="bitstream", type_desc="ndarray")],
    )
    decode = _node(
        "Decode/Verify",
        "Decode the compressed representation and verify correctness or fidelity.",
        ConceptType.COMPRESSION,
        inputs=[IOSpec(name="bitstream", type_desc="ndarray")],
        outputs=[IOSpec(name="decoded", type_desc="ndarray")],
    )
    edges = [
        _edge(model, encode, "model", "model", "ndarray"),
        _edge(encode, decode, "bitstream", "bitstream", "ndarray"),
    ]
    return SkeletonGraph(
        paradigm=ConceptType.COMPRESSION,
        name="Compression",
        description="Model the source, encode it, then decode and verify the representation.",
        template_nodes=[model, encode, decode],
        template_edges=edges,
        variants=["huffman_coding", "arithmetic_coding", "lempel_ziv", "dictionary_coding"],
    )


def _build_fixed_point() -> SkeletonGraph:
    """Fixed-point iteration pattern (e.g., iterative solvers, convergence loops)."""
    root = _node(
        "Fixed Point Root",
        "Top-level fixed-point combinator node",
        ConceptType.FIXED_POINT,
        inputs=[IOSpec(name="initial_state", type_desc="any")],
        outputs=[IOSpec(name="converged_state", type_desc="any")],
        depth=1,
    )
    # Override status to DECOMPOSED for the root placeholder
    root = root.model_copy(
        update={"status": NodeStatus.PENDING, "fixed_point_max_iterations": 100}
    )

    body_init = _node(
        "Body Init",
        "Seed the iteration state",
        ConceptType.STATE_INIT,
        inputs=[IOSpec(name="initial_state", type_desc="any")],
        outputs=[IOSpec(name="state", type_desc="any")],
    )
    body_step = _node(
        "Body Step",
        "Perform one iteration step",
        ConceptType.CUSTOM,
        inputs=[IOSpec(name="state", type_desc="any")],
        outputs=[IOSpec(name="next_state", type_desc="any")],
    )
    convergence_check = _node(
        "Convergence Check",
        "Test stopping criterion",
        ConceptType.CUSTOM,
        inputs=[
            IOSpec(name="prev_state", type_desc="any"),
            IOSpec(name="next_state", type_desc="any"),
        ],
        outputs=[IOSpec(name="converged", type_desc="bool")],
    )

    edges = [
        _edge(body_init, body_step, "state", "state", "any"),
        _edge(body_step, convergence_check, "next_state", "next_state", "any"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.FIXED_POINT,
        name="Fixed Point",
        description=(
            "Iterative fixed-point combinator: initialise state, "
            "apply a body step repeatedly, and check convergence."
        ),
        template_nodes=[root, body_init, body_step, convergence_check],
        template_edges=edges,
        variants=["iterative_solver", "convergence_loop", "fixed_point_combinator"],
    )


def _build_map_over() -> SkeletonGraph:
    """MAP combinator: slice input into windows and process each slice."""
    root = _node(
        "MAP Root",
        "Top-level MAP combinator node",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="results", type_desc="list[any]")],
        depth=1,
    )
    root = root.model_copy(
        update={"map_window_size": 1024, "map_hop_size": 512}
    )

    window_slicer = _node(
        "Window Slicer",
        "Produce overlapping windows from the input signal",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="window", type_desc="np.ndarray")],
    )
    body_init = _node(
        "Body Init",
        "Initialize per-window processing state",
        ConceptType.STATE_INIT,
        inputs=[IOSpec(name="window", type_desc="np.ndarray")],
        outputs=[IOSpec(name="state", type_desc="any")],
    )
    body_process = _node(
        "Body Process",
        "Process a single window",
        ConceptType.CUSTOM,
        inputs=[IOSpec(name="state", type_desc="any")],
        outputs=[IOSpec(name="result", type_desc="any")],
    )
    collect_results = _node(
        "Collect Results",
        "Aggregate per-window results into final output",
        ConceptType.CUSTOM,
        inputs=[IOSpec(name="result", type_desc="any")],
        outputs=[IOSpec(name="results", type_desc="list[any]")],
    )

    edges = [
        _edge(window_slicer, body_init, "window", "window", "np.ndarray"),
        _edge(body_init, body_process, "state", "state", "any"),
        _edge(body_process, collect_results, "result", "result", "any"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.MAP_OVER,
        name="MAP Over",
        description=(
            "MAP combinator: slice input into windows, apply a body "
            "subgraph to each window, collect results."
        ),
        template_nodes=[
            root,
            window_slicer,
            body_init,
            body_process,
            collect_results,
        ],
        template_edges=edges,
        variants=["sliding_window", "chunked_map", "strided_apply"],
    )


def _build_baseline_analysis() -> SkeletonGraph:
    """Multi-scale temporal baseline analysis pipeline.

    Topology mirrors the canonical baseline component execution:

    Per-window (MAP body):
        Mask -> Resample -> Scale -> Per-Window Fit -> Output Transform

    Post-window (top-level):
        Qualify Events -> Pad -> Normalize -> Combine -> Regionize
    """
    acquire = _node(
        "Acquire Data",
        "Load or receive input time-series data",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="source", type_desc="BaselineTimeSeries")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )

    mask = _node(
        "Mask",
        "Apply zeroing/masking to window data",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="window", type_desc="np.ndarray")],
        outputs=[IOSpec(name="masked", type_desc="np.ndarray")],
        matched_primitive="baseline_mask",
        status=NodeStatus.ATOMIC,
    )
    resample = _node(
        "Resample",
        "Resample or aggregate signal to anchor sample rate",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="masked", type_desc="np.ndarray")],
        outputs=[IOSpec(name="resampled", type_desc="np.ndarray")],
        matched_primitive="baseline_resample",
        status=NodeStatus.ATOMIC,
    )
    scale = _node(
        "Scale",
        "Normalize signal magnitude within each window",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="resampled", type_desc="np.ndarray")],
        outputs=[IOSpec(name="scaled", type_desc="np.ndarray")],
        matched_primitive="baseline_scale_constant",
        status=NodeStatus.ATOMIC,
    )
    per_window_fit = _node(
        "Per-Window Fit",
        "Run non-linear curve fitting on each window",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="scaled", type_desc="np.ndarray")],
        outputs=[
            IOSpec(
                name="fit_internals",
                type_desc="BaselineFitStackInternals",
            )
        ],
        matched_primitive="baseline_fit_exp_rise",
        status=NodeStatus.ATOMIC,
    )
    output_transform = _node(
        "Output Transform",
        "Convert per-window fit results into onset arrays",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(
                name="fit_internals",
                type_desc="BaselineFitStackInternals",
            )
        ],
        outputs=[IOSpec(name="onsets", type_desc="np.ndarray")],
        matched_primitive="baseline_output_nonzero",
        status=NodeStatus.ATOMIC,
    )
    windowed_analysis = _node(
        "Windowed Analysis",
        "Sliding window iteration over input signal; body runs per window",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="accumulated_onsets", type_desc="list[np.ndarray]")],
    )
    windowed_analysis = windowed_analysis.model_copy(
        update={
            "map_window_size": 1024,
            "map_hop_size": 512,
            "children": [
                mask.node_id,
                resample.node_id,
                scale.node_id,
                per_window_fit.node_id,
                output_transform.node_id,
            ],
        }
    )
    qualify_events = _node(
        "Qualify Events",
        "FitStack.qualify(): process accumulated fits through the ONSET->CENTER->OFFSET state machine",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="accumulated_onsets", type_desc="list[np.ndarray]")],
        outputs=[IOSpec(name="probability", type_desc="np.ndarray")],
    )
    qualify_events = qualify_events.model_copy(
        update={
            "is_opaque": True,
            "matched_primitive": "baseline_fit_stack",
        }
    )
    pad = _node(
        "Pad",
        "Apply left and right padding around onsets to build a probability vector",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="probability", type_desc="np.ndarray")],
        outputs=[IOSpec(name="padded", type_desc="np.ndarray")],
        matched_primitive="baseline_pad_constant",
        status=NodeStatus.ATOMIC,
    )
    normalize = _node(
        "Normalize",
        "Normalize per-component probability to [0, 1] via max, constant, or quantile method",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="padded", type_desc="np.ndarray")],
        outputs=[IOSpec(name="normalized", type_desc="np.ndarray")],
        matched_primitive="baseline_normalize_max",
        status=NodeStatus.ATOMIC,
    )
    combine = _node(
        "Combine",
        "Combine multiple component outputs across components",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="normalized", type_desc="np.ndarray")],
        outputs=[IOSpec(name="combined", type_desc="np.ndarray")],
        matched_primitive="baseline_combine_product",
        status=NodeStatus.ATOMIC,
    )
    regionize = _node(
        "Regionize",
        "Threshold combined signal into discrete event regions",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="combined", type_desc="np.ndarray")],
        outputs=[IOSpec(name="regions", type_desc="list[tuple[int,int]]")],
        matched_primitive="baseline_regionize",
        status=NodeStatus.ATOMIC,
    )

    body_edges = [
        _edge(mask, resample, "masked", "masked", "np.ndarray"),
        _edge(resample, scale, "resampled", "resampled", "np.ndarray"),
        _edge(scale, per_window_fit, "scaled", "scaled", "np.ndarray"),
        _edge(
            per_window_fit,
            output_transform,
            "fit_internals",
            "fit_internals",
            "BaselineFitStackInternals",
        ),
    ]
    top_edges = [
        _edge(acquire, windowed_analysis, "signal", "signal", "np.ndarray"),
        _edge(
            windowed_analysis,
            qualify_events,
            "accumulated_onsets",
            "accumulated_onsets",
            "list[np.ndarray]",
        ),
        _edge(qualify_events, pad, "probability", "probability", "np.ndarray"),
        _edge(pad, normalize, "padded", "padded", "np.ndarray"),
        _edge(normalize, combine, "normalized", "normalized", "np.ndarray"),
        _edge(combine, regionize, "combined", "combined", "np.ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.BASELINE_ANALYSIS,
        name="Baseline Analysis",
        description=(
            "Multi-scale temporal event detection: acquire signal, apply a "
            "per-window step pipeline via MAP combinator, qualify events, "
            "pad, normalize per component, combine components, and regionize."
        ),
        template_nodes=[
            acquire,
            windowed_analysis,
            mask,
            resample,
            scale,
            per_window_fit,
            output_transform,
            qualify_events,
            pad,
            normalize,
            combine,
            regionize,
        ],
        template_edges=body_edges + top_edges,
        variants=[
            "physiological_baseline",
            "multi_component_detection",
            "temporal_event_extraction",
            "sliding_window_fit",
        ],
    )


def _build_baseline_scoring() -> SkeletonGraph:
    """Baseline-core score graph that consumes analyzer outputs."""
    sqi_events = _node(
        "Analyzer Output: sqi_events",
        "Expose the SQI event regions produced by the baseline analyzer.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="prediction", type_desc="list[tuple[int,int]]")],
        status=NodeStatus.ATOMIC,
    )
    sqi_probability = _node(
        "Analyzer Output: sqi_probability",
        "Expose the SQI probability trace produced by the baseline analyzer.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="prediction", type_desc="np.ndarray")],
        status=NodeStatus.ATOMIC,
    )
    combined_events = _node(
        "Analyzer Output: combined_events",
        "Expose the combined baseline event regions produced by the analyzer.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="prediction", type_desc="list[tuple[int,int]]")],
        status=NodeStatus.ATOMIC,
    )
    pat_events = _node(
        "Analyzer Output: pat_events",
        "Expose the PAT event regions produced by the baseline analyzer.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="prediction", type_desc="list[tuple[int,int]]")],
        status=NodeStatus.ATOMIC,
    )
    pat_probability = _node(
        "Analyzer Output: pat_probability",
        "Expose the PAT probability trace produced by the baseline analyzer.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="prediction", type_desc="np.ndarray")],
        status=NodeStatus.ATOMIC,
    )
    spo2_probability = _node(
        "Analyzer Output: spo2_probability",
        "Expose the SpO2 probability output used for moderate/severe inference.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="prediction", type_desc="np.ndarray")],
        status=NodeStatus.ATOMIC,
    )
    anchor = _node(
        "Analyzer Anchor",
        "Expose the analyzer anchor time grid for duration and density scoring.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="anchor", type_desc="np.ndarray")],
        status=NodeStatus.ATOMIC,
    )
    sleep_mask = _node(
        "Analyzer Sleep Mask",
        "Expose the sleep mask used to accumulate analyzed sleep time.",
        ConceptType.DATA_EXTRACTION,
        outputs=[IOSpec(name="sleep_mask", type_desc="np.ndarray")],
        status=NodeStatus.ATOMIC,
    )
    bmi = _node(
        "Analyzer BMI",
        "Expose the per-night BMI value used by the baseline BMI correction.",
        ConceptType.DATA_EXTRACTION,
        outputs=[
            IOSpec(
                name="bmi",
                type_desc="float",
                required=False,
                default_value_repr="22.0",
            )
        ],
        status=NodeStatus.ATOMIC,
    )
    analyzed_time = _node(
        "Compute Analyzed Sleep Time",
        "Accumulate analyzed sleep time from the analyzer sleep mask and anchor.",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(name="anchor", type_desc="np.ndarray"),
            IOSpec(name="sleep_mask", type_desc="np.ndarray"),
        ],
        outputs=[IOSpec(name="analyzed_time_hours", type_desc="float")],
        matched_primitive="accumulate_analyzed_time",
        status=NodeStatus.ATOMIC,
    )
    sqi_density = _node(
        "Compute SQI Density",
        "Accumulate padded SQI prediction-window coverage from the analyzer output.",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(name="prediction", type_desc="np.ndarray"),
            IOSpec(name="anchor", type_desc="np.ndarray"),
        ],
        outputs=[IOSpec(name="density_hours", type_desc="float")],
        matched_primitive="accumulate_prediction_window_time",
        status=NodeStatus.ATOMIC,
    )
    pat_density = _node(
        "Compute PAT Density",
        "Accumulate padded PAT prediction-window coverage from the analyzer output.",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(name="prediction", type_desc="np.ndarray"),
            IOSpec(name="anchor", type_desc="np.ndarray"),
        ],
        outputs=[IOSpec(name="density_hours", type_desc="float")],
        matched_primitive="accumulate_prediction_window_time",
        status=NodeStatus.ATOMIC,
    )
    sahi = _node(
        "Score sAHI",
        "Score the SQI baseline path into an sAHI-style value.",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(name="predictor_events", type_desc="list[tuple[int,int]]"),
            IOSpec(name="combined_events", type_desc="list[tuple[int,int]]"),
            IOSpec(name="analyzed_time_hours", type_desc="float"),
            IOSpec(name="density_hours", type_desc="float"),
            IOSpec(
                name="spo2_probabilities",
                type_desc="np.ndarray",
                required=False,
            ),
        ],
        outputs=[IOSpec(name="sAHI", type_desc="float")],
        matched_primitive="score_baseline_path",
        status=NodeStatus.ATOMIC,
    )
    bahi = _node(
        "Score bAHI",
        "Score the BMI-corrected SQI baseline path into a bAHI-style value.",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(name="predictor_events", type_desc="list[tuple[int,int]]"),
            IOSpec(name="combined_events", type_desc="list[tuple[int,int]]"),
            IOSpec(name="analyzed_time_hours", type_desc="float"),
            IOSpec(name="density_hours", type_desc="float"),
            IOSpec(
                name="bmi",
                type_desc="float",
                required=False,
                default_value_repr="22.0",
            ),
            IOSpec(
                name="spo2_probabilities",
                type_desc="np.ndarray",
                required=False,
            ),
        ],
        outputs=[IOSpec(name="bAHI", type_desc="float")],
        matched_primitive="score_bmi_baseline_path",
        status=NodeStatus.ATOMIC,
    )
    pahi = _node(
        "Score pAHI",
        "Score the PAT baseline branch into a pAHI-style value.",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[
            IOSpec(name="pat_events", type_desc="list[tuple[int,int]]"),
            IOSpec(name="analyzed_time_hours", type_desc="float"),
            IOSpec(name="density_hours", type_desc="float"),
        ],
        outputs=[IOSpec(name="pAHI", type_desc="float")],
        matched_primitive="score_pat_baseline_path",
        status=NodeStatus.ATOMIC,
    )

    edges = [
        _edge(anchor, analyzed_time, "anchor", "anchor", "np.ndarray"),
        _edge(
            sleep_mask,
            analyzed_time,
            "sleep_mask",
            "sleep_mask",
            "np.ndarray",
        ),
        _edge(anchor, sqi_density, "anchor", "anchor", "np.ndarray"),
        _edge(sqi_probability, sqi_density, "prediction", "prediction", "np.ndarray"),
        _edge(anchor, pat_density, "anchor", "anchor", "np.ndarray"),
        _edge(pat_probability, pat_density, "prediction", "prediction", "np.ndarray"),
        _edge(
            sqi_events,
            sahi,
            "prediction",
            "predictor_events",
            "list[tuple[int,int]]",
        ),
        _edge(
            combined_events,
            sahi,
            "prediction",
            "combined_events",
            "list[tuple[int,int]]",
        ),
        _edge(
            analyzed_time,
            sahi,
            "analyzed_time_hours",
            "analyzed_time_hours",
            "float",
        ),
        _edge(sqi_density, sahi, "density_hours", "density_hours", "float"),
        _edge(
            spo2_probability,
            sahi,
            "prediction",
            "spo2_probabilities",
            "np.ndarray",
        ),
        _edge(
            sqi_events,
            bahi,
            "prediction",
            "predictor_events",
            "list[tuple[int,int]]",
        ),
        _edge(
            combined_events,
            bahi,
            "prediction",
            "combined_events",
            "list[tuple[int,int]]",
        ),
        _edge(
            analyzed_time,
            bahi,
            "analyzed_time_hours",
            "analyzed_time_hours",
            "float",
        ),
        _edge(sqi_density, bahi, "density_hours", "density_hours", "float"),
        _edge(bmi, bahi, "bmi", "bmi", "float"),
        _edge(
            spo2_probability,
            bahi,
            "prediction",
            "spo2_probabilities",
            "np.ndarray",
        ),
        _edge(pat_events, pahi, "prediction", "pat_events", "list[tuple[int,int]]"),
        _edge(
            analyzed_time,
            pahi,
            "analyzed_time_hours",
            "analyzed_time_hours",
            "float",
        ),
        _edge(pat_density, pahi, "density_hours", "density_hours", "float"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.BASELINE_ANALYSIS,
        name="Baseline Scoring",
        description=(
            "Baseline-core score assembly that consumes analyzer aliases and "
            "produces sAHI, bAHI, and pAHI outputs."
        ),
        template_nodes=[
            sqi_events,
            sqi_probability,
            combined_events,
            pat_events,
            pat_probability,
            spo2_probability,
            anchor,
            sleep_mask,
            bmi,
            analyzed_time,
            sqi_density,
            pat_density,
            sahi,
            bahi,
            pahi,
        ],
        template_edges=edges,
        variants=["ahi_baseline_scoring", "baseline_score_graph"],
    )


BASELINE_SCORING_SKELETON = _build_baseline_scoring()


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
    ConceptType.ALGEBRA: _build_linear_algebra(),
    ConceptType.OPTIMIZATION: _build_optimization(),
    ConceptType.COMBINATORICS: _build_combinatorics(),
    ConceptType.NEURAL_NETWORK: _build_neural_network(),
    ConceptType.CLUSTERING: _build_clustering(),
    ConceptType.DIMENSIONALITY_REDUCTION: _build_dimensionality_reduction(),
    ConceptType.ODE_SOLVER: _build_ode_solver(),
    ConceptType.QUADRATURE: _build_quadrature(),
    ConceptType.RANDOMIZED: _build_randomized(),
    ConceptType.INFORMATION_THEORY: _build_information_theory(),
    ConceptType.COMPRESSION: _build_compression(),
    ConceptType.FIXED_POINT: _build_fixed_point(),
    ConceptType.MAP_OVER: _build_map_over(),
    ConceptType.BASELINE_ANALYSIS: _build_baseline_analysis(),
}


# Named skeleton variants for paradigms with multiple topologies
NAMED_SKELETONS: dict[str, SkeletonGraph] = {
    "kalman_filter": _build_kalman_filter(),
    "particle_filter": SKELETON_TEMPLATES[ConceptType.SEQUENTIAL_FILTER],
    "hmc": SKELETON_TEMPLATES[ConceptType.MCMC_KERNEL],
    "advi": SKELETON_TEMPLATES[ConceptType.VI_ELBO],
    "belief_propagation": SKELETON_TEMPLATES[ConceptType.MESSAGE_PASSING],
    "fixed_point": SKELETON_TEMPLATES[ConceptType.FIXED_POINT],
    "iterative_solver": SKELETON_TEMPLATES[ConceptType.FIXED_POINT],
    "convergence_loop": SKELETON_TEMPLATES[ConceptType.FIXED_POINT],
    "map_over": SKELETON_TEMPLATES[ConceptType.MAP_OVER],
    "sliding_window": SKELETON_TEMPLATES[ConceptType.MAP_OVER],
    "baseline_scoring": BASELINE_SCORING_SKELETON,
    "ahi_baseline_scoring": BASELINE_SCORING_SKELETON,
    "baseline_score_graph": BASELINE_SCORING_SKELETON,
    "signal_detect_measure": _build_signal_detect_measure(),
    "event_rate_estimation": _build_signal_detect_measure(),
    "bandpass_hr_detection": _build_signal_detect_measure(),
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
        id_map[tpl_node.node_id] = f"{tpl_node.node_id}_{uuid.uuid4().hex[:8]}"

    for tpl_node in skeleton.template_nodes:
        node = AlgorithmicNode(
            node_id=id_map[tpl_node.node_id],
            parent_id=parent_id,
            name=tpl_node.name,
            description=f"[{goal_desc}] {tpl_node.description}",
            concept_type=tpl_node.concept_type,
            inputs=tpl_node.inputs,
            outputs=tpl_node.outputs,
            matched_primitive=tpl_node.matched_primitive,
            status=tpl_node.status,
            depth=base_depth + tpl_node.depth,
            parallelizable=tpl_node.parallelizable,
        )
        node = node.model_copy(
            update={
                "is_opaque": tpl_node.is_opaque,
                "type_signature": tpl_node.type_signature,
                "matched_primitive": tpl_node.matched_primitive,
                "primitive_binding_confidence": tpl_node.primitive_binding_confidence,
                "primitive_binding_source": tpl_node.primitive_binding_source,
                "conceptual_summary": tpl_node.conceptual_summary,
                "critic_notes": tpl_node.critic_notes,
                "decomposition_rationale": tpl_node.decomposition_rationale,
                "children": [
                    id_map[child_id]
                    for child_id in tpl_node.children
                    if child_id in id_map
                ],
                "fixed_point_max_iterations": tpl_node.fixed_point_max_iterations,
                "fixed_point_convergence_field": tpl_node.fixed_point_convergence_field,
                "map_window_size": tpl_node.map_window_size,
                "map_hop_size": tpl_node.map_hop_size,
            }
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


def _clone_template_node(
    tpl_node: AlgorithmicNode,
    goal_desc: str,
    *,
    parent_id: str | None,
    base_depth: int,
    name: str | None = None,
) -> AlgorithmicNode:
    """Clone a template node with a fresh ID and goal-prefixed description."""
    return tpl_node.model_copy(
        deep=True,
        update={
            "node_id": f"{tpl_node.node_id}_{uuid.uuid4().hex[:8]}",
            "parent_id": parent_id,
            "name": name or tpl_node.name,
            "description": f"[{goal_desc}] {tpl_node.description}",
            "depth": base_depth + tpl_node.depth,
        },
    )


def _remap_template_edge(
    tpl_edge: DependencyEdge,
    *,
    source_id: str,
    target_id: str,
) -> DependencyEdge:
    """Clone a template edge while swapping in fresh endpoint IDs."""
    return tpl_edge.model_copy(
        update={"source_id": source_id, "target_id": target_id}
    )


def _port_type(ports: list[IOSpec], default_name: str, default_type: str) -> tuple[str, str]:
    """Extract a single port signature from a node or stage spec."""
    if not ports:
        return default_name, default_type
    return ports[0].name, ports[0].type_desc


def _baseline_stage_ports(stage: BaselineStageSpec) -> tuple[str, str, str, str]:
    """Return canonical edge metadata for a baseline stage spec."""
    return (
        stage.input_name,
        stage.input_type,
        stage.output_name,
        stage.output_type,
    )


def _build_baseline_stage_node(
    skeleton: SkeletonGraph,
    stage: BaselineStageSpec,
    goal_desc: str,
    *,
    parent_id: str | None,
    base_depth: int,
    name: str,
) -> AlgorithmicNode:
    """Instantiate a baseline stage node, cloning a template when available."""
    template_name = stage.template_name or stage.name
    template = next(
        (node for node in skeleton.template_nodes if node.name == template_name),
        None,
    )
    if template is not None:
        node = _clone_template_node(
            template,
            goal_desc,
            parent_id=parent_id,
            base_depth=base_depth,
            name=name,
        )
    else:
        node = AlgorithmicNode(
            node_id=f"baseline_stage_{uuid.uuid4().hex[:8]}",
            parent_id=parent_id,
            name=name,
            description=f"[{goal_desc}] {stage.description or stage.name}",
            concept_type=stage.concept_type,
            depth=base_depth + 1,
        )

    return node.model_copy(
        update={
            "description": f"[{goal_desc}] {stage.description or stage.name}",
            "concept_type": stage.concept_type,
            "inputs": [IOSpec(name=stage.input_name, type_desc=stage.input_type)],
            "outputs": [IOSpec(name=stage.output_name, type_desc=stage.output_type)],
            "matched_primitive": stage.matched_primitive,
            "status": stage.status,
            "is_opaque": stage.is_opaque,
            "is_optional": stage.is_optional,
        }
    )


def _build_windowed_component_node(
    skeleton: SkeletonGraph,
    window: BaselineWindowSpec,
    goal_desc: str,
    *,
    parent_id: str | None,
    base_depth: int,
    name: str,
) -> AlgorithmicNode:
    """Instantiate a windowed-analysis node for a component."""
    template = next(
        (node for node in skeleton.template_nodes if node.name == "Windowed Analysis"),
        None,
    )
    if template is not None:
        node = _clone_template_node(
            template,
            goal_desc,
            parent_id=parent_id,
            base_depth=base_depth,
            name=name,
        )
    else:
        node = AlgorithmicNode(
            node_id=f"baseline_windowed_{uuid.uuid4().hex[:8]}",
            parent_id=parent_id,
            name=name,
            description=f"[{goal_desc}] {window.description}",
            concept_type=ConceptType.MAP_OVER,
            depth=base_depth + 1,
        )

    return node.model_copy(
        update={
            "description": f"[{goal_desc}] {window.description}",
            "concept_type": ConceptType.MAP_OVER,
            "inputs": [IOSpec(name=window.input_name, type_desc=window.input_type)],
            "outputs": [IOSpec(name=window.output_name, type_desc=window.output_type)],
            "map_window_size": window.size,
            "map_hop_size": window.hop,
        }
    )


def _build_predictor_alias_node(
    alias: BaselinePredictorAliasSpec,
    goal_desc: str,
    *,
    parent_id: str | None,
    base_depth: int,
    source_type: str,
) -> AlgorithmicNode:
    """Create a leaf node that exposes a component output under an alias."""
    name = alias.name or f"Predictor Alias: {alias.alias}"
    return AlgorithmicNode(
        node_id=f"baseline_predictor_alias_{uuid.uuid4().hex[:8]}",
        parent_id=parent_id,
        name=name,
        description=f"[{goal_desc}] {alias.description or f'Expose predictor alias {alias.alias}.'}",
        concept_type=ConceptType.DATA_EXTRACTION,
        inputs=[IOSpec(name="source", type_desc=source_type)],
        outputs=[IOSpec(name="prediction", type_desc=source_type)],
        status=NodeStatus.ATOMIC,
        depth=base_depth + 1,
    )


def instantiate_baseline_analyzer(
    skeleton: SkeletonGraph,
    goal: str,
    spec: BaselineAnalyzerSpec,
    *,
    parent_id: str | None = None,
    base_depth: int = 0,
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Instantiate a heterogeneous baseline analyzer assembly."""
    if skeleton.paradigm != ConceptType.BASELINE_ANALYSIS:
        raise ValueError("baseline analyzer instantiation requires BASELINE_ANALYSIS")

    acquire_template = next(
        (node for node in skeleton.template_nodes if node.name == "Acquire Data"),
        None,
    )
    if acquire_template is None:
        raise ValueError("baseline skeleton is missing required node: Acquire Data")

    acquire = _clone_template_node(
        acquire_template,
        goal,
        parent_id=parent_id,
        base_depth=base_depth,
    )
    nodes: list[AlgorithmicNode] = [acquire]
    edges: list[DependencyEdge] = []

    preprocessor_nodes: dict[str, AlgorithmicNode] = {}
    upstream_node = acquire
    upstream_output_name, upstream_output_type = _port_type(
        acquire.outputs,
        "signal",
        "np.ndarray",
    )
    for stage in spec.preprocessors:
        node = _build_baseline_stage_node(
            skeleton,
            stage,
            goal,
            parent_id=parent_id,
            base_depth=base_depth,
            name=stage.name,
        )
        nodes.append(node)
        edges.append(
            DependencyEdge(
                source_id=upstream_node.node_id,
                target_id=node.node_id,
                output_name=upstream_output_name,
                input_name=stage.input_name,
                source_type=upstream_output_type,
                target_type=stage.input_type,
                requires_glue=upstream_output_type != stage.input_type,
            )
        )
        preprocessor_nodes[stage.key] = node
        upstream_node = node
        upstream_output_name, upstream_output_type = _baseline_stage_ports(stage)[2:]

    component_stage_nodes: dict[str, dict[str, AlgorithmicNode]] = {}
    for component in spec.components:
        stage_nodes: dict[str, AlgorithmicNode] = {}
        component_prefix = f"({component.name})"

        if component.shape == BaselineComponentShape.WINDOWED:
            source_node = (
                preprocessor_nodes[component.source_key]
                if component.source_key is not None
                else acquire
            )
            source_output_name, source_output_type = _port_type(
                source_node.outputs,
                "signal",
                "np.ndarray",
            )

            window_node = _build_windowed_component_node(
                skeleton,
                component.window,
                goal,
                parent_id=parent_id,
                base_depth=base_depth,
                name=f"{component.window.name} {component_prefix}",
            )
            nodes.append(window_node)
            edges.append(
                DependencyEdge(
                    source_id=source_node.node_id,
                    target_id=window_node.node_id,
                    output_name=source_output_name,
                    input_name=component.window.input_name,
                    source_type=source_output_type,
                    target_type=component.window.input_type,
                    requires_glue=source_output_type != component.window.input_type,
                )
            )
            stage_nodes["windowed"] = window_node

            body_nodes: list[AlgorithmicNode] = []
            for stage in component.window_stages:
                node = _build_baseline_stage_node(
                    skeleton,
                    stage,
                    goal,
                    parent_id=parent_id,
                    base_depth=base_depth,
                    name=f"{stage.name} {component_prefix}",
                )
                body_nodes.append(node)
                stage_nodes[stage.key] = node
                nodes.append(node)

            window_node = window_node.model_copy(
                update={"children": [node.node_id for node in body_nodes]}
            )
            nodes[nodes.index(stage_nodes["windowed"])] = window_node
            stage_nodes["windowed"] = window_node

            for source_stage, target_stage in zip(body_nodes, body_nodes[1:]):
                source_output_name, source_output_type = _port_type(
                    source_stage.outputs,
                    "signal",
                    "np.ndarray",
                )
                target_input_name, target_input_type = _port_type(
                    target_stage.inputs,
                    "signal",
                    "np.ndarray",
                )
                edges.append(
                    DependencyEdge(
                        source_id=source_stage.node_id,
                        target_id=target_stage.node_id,
                        output_name=source_output_name,
                        input_name=target_input_name,
                        source_type=source_output_type,
                        target_type=target_input_type,
                        requires_glue=source_output_type != target_input_type,
                    )
                )

            previous_node = window_node
            previous_output_name, previous_output_type = _port_type(
                window_node.outputs,
                component.window.output_name,
                component.window.output_type,
            )
        else:
            combine_node = _build_baseline_stage_node(
                skeleton,
                component.combine_stage,
                goal,
                parent_id=parent_id,
                base_depth=base_depth,
                name=f"{component.combine_stage.name} {component_prefix}",
            )
            nodes.append(combine_node)
            stage_nodes[component.combine_stage.key] = combine_node

            for ref in component.combine_inputs:
                source_node = component_stage_nodes[ref.component][ref.stage_key]
                source_output_name, source_output_type = _port_type(
                    source_node.outputs,
                    "signal",
                    "np.ndarray",
                )
                target_input_name, target_input_type = _port_type(
                    combine_node.inputs,
                    component.combine_stage.input_name,
                    component.combine_stage.input_type,
                )
                edges.append(
                    DependencyEdge(
                        source_id=source_node.node_id,
                        target_id=combine_node.node_id,
                        output_name=source_output_name,
                        input_name=target_input_name,
                        source_type=source_output_type,
                        target_type=target_input_type,
                        requires_glue=source_output_type != target_input_type,
                    )
                )

            previous_node = combine_node
            previous_output_name, previous_output_type = _port_type(
                combine_node.outputs,
                component.combine_stage.output_name,
                component.combine_stage.output_type,
            )

        for stage in component.post_stages:
            node = _build_baseline_stage_node(
                skeleton,
                stage,
                goal,
                parent_id=parent_id,
                base_depth=base_depth,
                name=f"{stage.name} {component_prefix}",
            )
            nodes.append(node)
            edges.append(
                DependencyEdge(
                    source_id=previous_node.node_id,
                    target_id=node.node_id,
                    output_name=previous_output_name,
                    input_name=stage.input_name,
                    source_type=previous_output_type,
                    target_type=stage.input_type,
                    requires_glue=previous_output_type != stage.input_type,
                )
            )
            stage_nodes[stage.key] = node
            previous_node = node
            previous_output_name, previous_output_type = _baseline_stage_ports(stage)[2:]

        component_stage_nodes[component.name] = stage_nodes

    for alias in spec.predictor_aliases:
        source_node = component_stage_nodes[alias.source.component][alias.source.stage_key]
        source_output_name, source_output_type = _port_type(
            source_node.outputs,
            "signal",
            "np.ndarray",
        )
        alias_node = _build_predictor_alias_node(
            alias,
            goal,
            parent_id=parent_id,
            base_depth=base_depth,
            source_type=source_output_type,
        )
        nodes.append(alias_node)
        edges.append(
            DependencyEdge(
                source_id=source_node.node_id,
                target_id=alias_node.node_id,
                output_name=source_output_name,
                input_name="source",
                source_type=source_output_type,
                target_type=source_output_type,
            )
        )

    return nodes, edges


def instantiate_baseline_scoring(
    goal: str,
    *,
    parent_id: str | None = None,
    base_depth: int = 0,
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Instantiate the canonical baseline-core scoring graph."""
    skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS, variant="baseline_scoring")
    if skeleton is None:
        raise ValueError("baseline scoring skeleton is unavailable")
    return instantiate_skeleton(
        skeleton,
        goal,
        parent_id=parent_id,
        base_depth=base_depth,
    )


def instantiate_baseline_multi_component(
    skeleton: SkeletonGraph,
    goal: str,
    n_components: int,
    *,
    parent_id: str | None = None,
    base_depth: int = 0,
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Instantiate a multi-component baseline analysis CDG.

    Creates N copies of the per-component pipeline and wires all into a
    shared Acquire, Combine, and Regionize.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if skeleton.paradigm != ConceptType.BASELINE_ANALYSIS:
        raise ValueError("baseline multi-component instantiation requires BASELINE_ANALYSIS")

    if n_components == 1:
        return instantiate_skeleton(
            skeleton,
            goal,
            parent_id=parent_id,
            base_depth=base_depth,
        )

    name_to_template = {node.name: node for node in skeleton.template_nodes}
    required_names = [
        "Acquire Data",
        "Windowed Analysis",
        "Mask",
        "Resample",
        "Scale",
        "Per-Window Fit",
        "Output Transform",
        "Qualify Events",
        "Pad",
        "Normalize",
        "Combine",
        "Regionize",
    ]
    missing = [name for name in required_names if name not in name_to_template]
    if missing:
        raise ValueError(
            "baseline skeleton is missing required nodes: " + ", ".join(missing)
        )

    tpl_name_by_id = {node.node_id: node.name for node in skeleton.template_nodes}
    shared_acquire = _clone_template_node(
        name_to_template["Acquire Data"],
        goal,
        parent_id=parent_id,
        base_depth=base_depth,
    )
    shared_combine = _clone_template_node(
        name_to_template["Combine"],
        goal,
        parent_id=parent_id,
        base_depth=base_depth,
    )
    shared_regionize = _clone_template_node(
        name_to_template["Regionize"],
        goal,
        parent_id=parent_id,
        base_depth=base_depth,
    )

    nodes: list[AlgorithmicNode] = [shared_acquire]
    edges: list[DependencyEdge] = []
    chain_names = [
        "Windowed Analysis",
        "Mask",
        "Resample",
        "Scale",
        "Per-Window Fit",
        "Output Transform",
        "Qualify Events",
        "Pad",
        "Normalize",
    ]
    map_body_names = [
        "Mask",
        "Resample",
        "Scale",
        "Per-Window Fit",
        "Output Transform",
    ]

    for component_index in range(n_components):
        suffix = f" (Component {component_index + 1})"
        component_nodes: dict[str, AlgorithmicNode] = {}
        for name in chain_names:
            component_node = _clone_template_node(
                name_to_template[name],
                goal,
                parent_id=parent_id,
                base_depth=base_depth,
                name=f"{name}{suffix}",
            )
            component_nodes[name] = component_node
            nodes.append(component_node)
        component_nodes["Windowed Analysis"] = component_nodes[
            "Windowed Analysis"
        ].model_copy(
            update={
                "children": [
                    component_nodes[name].node_id for name in map_body_names
                ]
            }
        )
        nodes[-len(chain_names)] = component_nodes["Windowed Analysis"]

        for tpl_edge in skeleton.template_edges:
            source_name = tpl_name_by_id[tpl_edge.source_id]
            target_name = tpl_name_by_id[tpl_edge.target_id]

            if source_name == "Combine" and target_name == "Regionize":
                continue

            source_node = shared_acquire if source_name == "Acquire Data" else (
                shared_combine if source_name == "Combine" else component_nodes.get(source_name)
            )
            target_node = shared_regionize if target_name == "Regionize" else (
                shared_combine if target_name == "Combine" else component_nodes.get(target_name)
            )

            if source_node is None or target_node is None:
                raise ValueError(
                    f"cannot map baseline edge {source_name!r} -> {target_name!r}"
                )

            edges.append(
                _remap_template_edge(
                    tpl_edge,
                    source_id=source_node.node_id,
                    target_id=target_node.node_id,
                )
            )

    nodes.extend([shared_combine, shared_regionize])
    combine_regionize_edge = next(
        (
            edge
            for edge in skeleton.template_edges
            if tpl_name_by_id[edge.source_id] == "Combine"
            and tpl_name_by_id[edge.target_id] == "Regionize"
        ),
        None,
    )
    if combine_regionize_edge is None:
        raise ValueError("baseline skeleton is missing Combine -> Regionize edge")
    edges.append(
        _remap_template_edge(
            combine_regionize_edge,
            source_id=shared_combine.node_id,
            target_id=shared_regionize.node_id,
        )
    )

    return nodes, edges
