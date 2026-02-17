"""End-to-end orchestration for Phase 1: assemble and check."""

from __future__ import annotations

import logging

from ageom.architect.handoff import CDGExport
from ageom.protocols import ProofEnvironment
from ageom.synthesizer.assembler import Assembler
from ageom.synthesizer.compiler import SkeletonCompiler
from ageom.synthesizer.ghost_sim import GhostSimReport, run_ghost_simulation
from ageom.synthesizer.models import AssemblyResult
from ageom.types import MatchResult

logger = logging.getLogger(__name__)


async def assemble_and_check(
    cdg: CDGExport,
    match_results: list[MatchResult],
    env: ProofEnvironment,
    *,
    skip_ghost_sim: bool = False,
) -> AssemblyResult:
    """Assemble a CDG + match results into a skeleton, then compile it.

    Optionally runs a ghost witness simulation before assembly to catch
    structural mismatches early.  The simulation is best-effort — if
    ``ageoa`` is not installed or no witnesses are available, it is
    silently skipped.

    Args:
        cdg: The Conceptual Dependency Graph from Round 1.
        match_results: Verified matches from Round 2.
        env: Proof environment with prover and compiler config.
        skip_ghost_sim: Set True to bypass the ghost simulation pass.

    Returns:
        AssemblyResult with the skeleton, compiler feedback,
        and whether compilation succeeded.
    """
    # Ghost simulation pass — runs before assembly
    ghost_report = GhostSimReport()
    if not skip_ghost_sim:
        ghost_report = run_ghost_simulation(cdg, match_results)
        if ghost_report.ran and not ghost_report.passed:
            logger.warning(
                "Ghost simulation detected issues at node '%s' (%s): %s",
                ghost_report.error_node,
                ghost_report.error_function,
                ghost_report.error,
            )

    assembler = Assembler(prover=env.prover_name)
    skeleton = assembler.assemble(cdg, match_results)

    # Attach ghost report to skeleton metadata
    if ghost_report.ran:
        skeleton.metadata["ghost_simulation"] = {
            "passed": ghost_report.passed,
            "node_count": ghost_report.node_count,
            "skipped_nodes": ghost_report.skipped_nodes,
            "trace": ghost_report.trace,
            "error": ghost_report.error,
        }

    compiler = SkeletonCompiler(env)
    result = await compiler.compile(skeleton)

    return result
