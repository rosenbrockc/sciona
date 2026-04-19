"""Agent-callable deterministic tools for atom ingestion.

This package exposes the sciona ingester's deterministic tools as clean,
synchronous Python functions. Designed for use by CLI agents (Claude Code,
Codex, etc.) performing human-supervised atom ingestion.

Usage::

    from sciona.tools import extract_dfg, run_mypy, detect_cycles

See AGENT_INGESTION.md in the sciona-atoms repo for the full ingestion
workflow guide.
"""

from sciona.tools.audit import run_contribution_check, run_dejargon_check
from sciona.tools.context import SyncContextStore
from sciona.tools.cycle import break_cycle, detect_cycles
from sciona.tools.extract import (
    AttributeSemanticFact,
    MethodFact,
    RawDataFlowGraph,
    extract_dfg,
    extract_function_dfg,
    extract_procedural_dfg,
)
from sciona.tools.template import generate_abstract_profile, match_witness_template
from sciona.tools.validate_cdg import validate_cdg_ir
from sciona.tools.verify_ghost import build_ghost_fixes, classify_ghost_failure
from sciona.tools.verify_types import build_type_fixes, classify_type_failure, run_mypy

__all__ = [
    # Extraction
    "extract_dfg",
    "extract_function_dfg",
    "extract_procedural_dfg",
    "RawDataFlowGraph",
    "MethodFact",
    "AttributeSemanticFact",
    # Type verification
    "run_mypy",
    "classify_type_failure",
    "build_type_fixes",
    # Ghost verification
    "classify_ghost_failure",
    "build_ghost_fixes",
    # Cycle detection
    "detect_cycles",
    "break_cycle",
    # CDG validation
    "validate_cdg_ir",
    # Template matching
    "match_witness_template",
    "generate_abstract_profile",
    # Shared context
    "SyncContextStore",
    # Audit
    "run_contribution_check",
    "run_dejargon_check",
]
