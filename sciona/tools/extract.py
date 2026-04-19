"""Data-flow graph extraction from Python source files.

Wraps the ingester's AST-based extractor for agent use. All functions are
synchronous and deterministic — no LLM calls.
"""

from __future__ import annotations

from sciona.ingester.models import (
    AttributeSemanticFact,
    MethodFact,
    RawDataFlowGraph,
)
from sciona.tools._sync import run_sync


def extract_dfg(source_path: str, class_name: str) -> RawDataFlowGraph:
    """Extract a data-flow graph for a Python class.

    Parses the source file with the standard ast module and builds
    deterministic data-flow facts: method signatures, attribute access
    patterns, cross-window state, fitted/config classification, and
    internal call graph.

    Args:
        source_path: Absolute or relative path to the .py file.
        class_name: Name of the class to extract.

    Returns:
        A RawDataFlowGraph containing method facts, attribute facts,
        and dependency information.
    """
    from sciona.ingester.extractor import extract_data_flow

    return run_sync(extract_data_flow(source_path, class_name))


def extract_function_dfg(source_path: str, function_name: str) -> RawDataFlowGraph:
    """Extract a data-flow graph for a standalone function.

    Same as extract_dfg but targets a module-level function instead of
    a class.

    Args:
        source_path: Path to the .py file.
        function_name: Name of the function to extract.

    Returns:
        A RawDataFlowGraph for the function.
    """
    from sciona.ingester.extractor import extract_function_data_flow

    return run_sync(extract_function_data_flow(source_path, function_name))


def extract_procedural_dfg(
    source_path: str,
    pipeline_name: str | None = None,
    entry_block: str | None = None,
) -> RawDataFlowGraph:
    """Extract a data-flow graph via procedural SSA analysis.

    Deterministic extraction using Static Single Assignment form.
    No LLM calls. Best for standalone functions and scripts.

    Args:
        source_path: Path to the .py file.
        pipeline_name: Optional name override for the pipeline.
        entry_block: Optional entry block name.

    Returns:
        A RawDataFlowGraph built from SSA analysis.
    """
    from sciona.ingester.extractor import extract_procedural_data_flow

    return run_sync(
        extract_procedural_data_flow(source_path, pipeline_name, entry_block)
    )


__all__ = [
    "extract_dfg",
    "extract_function_dfg",
    "extract_procedural_dfg",
    "RawDataFlowGraph",
    "MethodFact",
    "AttributeSemanticFact",
]
