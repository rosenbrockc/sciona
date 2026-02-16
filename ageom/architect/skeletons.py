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
}


def get_skeleton(concept_type: ConceptType) -> SkeletonGraph | None:
    """Look up the skeleton template for a paradigm."""
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
