"""Benchmark case definitions."""

from __future__ import annotations

from sciona.architect.models import ConceptType
from sciona.benchmarks.core import FlowBenchmarkCase, FlowLeafSpec


def default_flow_benchmark_cases() -> list[FlowBenchmarkCase]:
    return [
        FlowBenchmarkCase(
            case_id="sorting_merge",
            domain="sorting",
            prompt=(
                "Sort a list by splitting it into halves, sorting each half, and "
                "merging the sorted halves."
            ),
            concept_type=ConceptType.SORTING,
            leaves=(
                FlowLeafSpec(
                    name="Split List",
                    description="Split a list into left and right halves.",
                    type_signature="list[int] -> tuple[list[int], list[int]]",
                    query_hint="Split List",
                    declaration_name="algorithms.split_list_halves",
                    inputs=(("items", "list[int]"),),
                    outputs=(("left", "list[int]"), ("right", "list[int]")),
                ),
                FlowLeafSpec(
                    name="Merge Sorted Halves",
                    description="Merge two sorted halves into one sorted list.",
                    type_signature="list[int] -> list[int] -> list[int]",
                    query_hint="Merge Sorted Halves",
                    declaration_name="algorithms.merge_sorted_halves",
                    inputs=(("left", "list[int]"), ("right", "list[int]")),
                    outputs=(("merged", "list[int]"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="graph_shortest_path",
            domain="graph",
            prompt="Compute shortest path distances from a source node in a weighted graph.",
            concept_type=ConceptType.GRAPH_OPTIMIZATION,
            leaves=(
                FlowLeafSpec(
                    name="Initialize Distances",
                    description="Initialize distance map for a weighted graph shortest path routine.",
                    type_signature="graph -> source -> dict[node, float]",
                    query_hint="Initialize Distance",
                    declaration_name="algorithms.initialize_distances",
                    inputs=(("graph", "Graph"), ("source", "Node")),
                    outputs=(("distances", "dict[node, float]"),),
                ),
                FlowLeafSpec(
                    name="Relax Edges",
                    description="Relax weighted edges to improve tentative shortest path distances.",
                    type_signature="graph -> dict[node, float] -> dict[node, float]",
                    query_hint="Relax Edges",
                    declaration_name="algorithms.relax_edges",
                    inputs=(("graph", "Graph"), ("distances", "dict[node, float]")),
                    outputs=(("updated", "dict[node, float]"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="dsp_bandpass_filter",
            domain="dsp",
            prompt="Design and apply a stable bandpass filter to ECG samples.",
            concept_type=ConceptType.SIGNAL_FILTER,
            leaves=(
                FlowLeafSpec(
                    name="Design Filter",
                    description="Design stable bandpass filter coefficients for ECG samples.",
                    type_signature="spec -> coefficients",
                    query_hint="Design Filter",
                    declaration_name="algorithms.design_bandpass_filter",
                    inputs=(("spec", "FilterSpec"),),
                    outputs=(("coefficients", "Coefficients"),),
                ),
                FlowLeafSpec(
                    name="Apply Filter",
                    description="Apply stable bandpass filter coefficients to ECG samples.",
                    type_signature="signal -> coefficients -> signal",
                    query_hint="Apply Filter",
                    declaration_name="algorithms.apply_bandpass_filter",
                    inputs=(("signal", "np.ndarray"), ("coefficients", "Coefficients")),
                    outputs=(("filtered_signal", "np.ndarray"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="search_binary",
            domain="sorting",
            prompt="Find the position of a target value in a sorted array using binary search.",
            concept_type=ConceptType.SEARCHING,
            leaves=(
                FlowLeafSpec(
                    name="Compute Midpoint",
                    description="Compute midpoint index for binary search bounds.",
                    type_signature="int -> int -> int",
                    query_hint="Compute Midpoint",
                    declaration_name="algorithms.binary_search_midpoint",
                    inputs=(("low", "int"), ("high", "int")),
                    outputs=(("mid", "int"),),
                ),
                FlowLeafSpec(
                    name="Compare Target",
                    description="Compare target value against midpoint element and narrow bounds.",
                    type_signature="list[int] -> int -> int -> tuple[int, int]",
                    query_hint="Compare Target",
                    declaration_name="algorithms.binary_search_compare",
                    inputs=(
                        ("sorted_array", "list[int]"),
                        ("target", "int"),
                        ("mid", "int"),
                    ),
                    outputs=(("low", "int"), ("high", "int")),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="fft_spectral_analysis",
            domain="dsp",
            prompt="Window a signal, compute its FFT, then extract the magnitude spectrum.",
            concept_type=ConceptType.SIGNAL_TRANSFORM,
            leaves=(
                FlowLeafSpec(
                    name="Apply Hann Window",
                    description="Apply a Hann window to a signal segment.",
                    type_signature="signal -> signal",
                    query_hint="Hann Window Signal",
                    declaration_name="algorithms.apply_hann_window",
                    inputs=(("signal", "np.ndarray"),),
                    outputs=(("windowed", "np.ndarray"),),
                ),
                FlowLeafSpec(
                    name="Fourier Transform",
                    description="Compute the Fast Fourier Transform of a windowed signal.",
                    type_signature="signal -> complex_spectrum",
                    query_hint="Fourier Transform Windowed",
                    declaration_name="algorithms.compute_fft",
                    inputs=(("windowed", "np.ndarray"),),
                    outputs=(("spectrum", "np.ndarray"),),
                ),
                FlowLeafSpec(
                    name="Magnitude Spectrum",
                    description="Extract magnitude spectrum from complex Fourier output.",
                    type_signature="complex_spectrum -> magnitude",
                    query_hint="Magnitude Spectrum Complex",
                    declaration_name="algorithms.extract_magnitude",
                    inputs=(("spectrum", "np.ndarray"),),
                    outputs=(("magnitude", "np.ndarray"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="matrix_multiply_strassen",
            domain="linear_algebra",
            prompt="Multiply two matrices by splitting into quadrants, recursing, then combining.",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            leaves=(
                FlowLeafSpec(
                    name="Split Into Submatrices",
                    description="Split a matrix into four submatrices for recursive multiplication.",
                    type_signature="matrix -> tuple[matrix, matrix, matrix, matrix]",
                    query_hint="Split Submatrices",
                    declaration_name="algorithms.split_matrix_quadrants",
                    inputs=(("matrix", "np.ndarray"),),
                    outputs=(
                        ("q11", "np.ndarray"),
                        ("q12", "np.ndarray"),
                        ("q21", "np.ndarray"),
                        ("q22", "np.ndarray"),
                    ),
                ),
                FlowLeafSpec(
                    name="Assemble Result Matrix",
                    description="Assemble partial products into the final result matrix.",
                    type_signature="tuple[matrix, matrix, matrix, matrix] -> matrix",
                    query_hint="Assemble Result",
                    declaration_name="algorithms.combine_matrix_quadrants",
                    inputs=(
                        ("q11", "np.ndarray"),
                        ("q12", "np.ndarray"),
                        ("q21", "np.ndarray"),
                        ("q22", "np.ndarray"),
                    ),
                    outputs=(("result", "np.ndarray"),),
                ),
            ),
        ),
        FlowBenchmarkCase(
            case_id="string_edit_distance",
            domain="strings",
            prompt="Compute the minimum edit distance between two strings using dynamic programming.",
            concept_type=ConceptType.DYNAMIC_PROGRAMMING,
            leaves=(
                FlowLeafSpec(
                    name="Initialize DP Table",
                    description="Initialize the edit distance DP table with base cases.",
                    type_signature="str -> str -> list[list[int]]",
                    query_hint="Initialize DP Table",
                    declaration_name="algorithms.init_edit_distance_table",
                    inputs=(("source", "str"), ("target", "str")),
                    outputs=(("table", "list[list[int]]"),),
                ),
                FlowLeafSpec(
                    name="Fill DP Table",
                    description="Fill the edit distance table using recurrence relation.",
                    type_signature="list[list[int]] -> str -> str -> list[list[int]]",
                    query_hint="Fill DP Table",
                    declaration_name="algorithms.fill_edit_distance_table",
                    inputs=(
                        ("table", "list[list[int]]"),
                        ("source", "str"),
                        ("target", "str"),
                    ),
                    outputs=(("filled", "list[list[int]]"),),
                ),
            ),
        ),
    ]
