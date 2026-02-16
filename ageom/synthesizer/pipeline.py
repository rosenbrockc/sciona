"""End-to-end orchestration for Phase 1: assemble and check."""

from __future__ import annotations

from ageom.architect.handoff import CDGExport
from ageom.protocols import ProofEnvironment
from ageom.synthesizer.assembler import Assembler
from ageom.synthesizer.compiler import SkeletonCompiler
from ageom.synthesizer.models import AssemblyResult
from ageom.types import MatchResult


async def assemble_and_check(
    cdg: CDGExport,
    match_results: list[MatchResult],
    env: ProofEnvironment,
) -> AssemblyResult:
    """Assemble a CDG + match results into a skeleton, then compile it.

    Returns an AssemblyResult with the skeleton, compiler feedback,
    and whether compilation succeeded.
    """
    assembler = Assembler(prover=env.prover_name)
    skeleton = assembler.assemble(cdg, match_results)

    compiler = SkeletonCompiler(env)
    result = await compiler.compile(skeleton)

    return result
