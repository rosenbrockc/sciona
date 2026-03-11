"""Service wrapper around the Synthesizer assembly/check/repair tools."""

from __future__ import annotations

from ageom.synthesizer.assembler import Assembler
from ageom.synthesizer.compiler import SkeletonCompiler
from ageom.synthesizer.pipeline import assemble_and_check
from ageom.services.models import (
    SynthesizerAssembleAndCheckRequest,
    SynthesizerAssembleAndCheckResult,
    SynthesizerAssembleRequest,
    SynthesizerAssembleResult,
    SynthesizerCompileRequest,
    SynthesizerCompileResult,
    SynthesizerRepairRequest,
    SynthesizerRepairResult,
)


class SynthesizerService:
    """Stable service entrypoint for assembly, compile, and repair operations."""

    def __init__(
        self,
        *,
        prover: object,
        repair_agent: object | None = None,
        assemble_and_check_fn: object = assemble_and_check,
    ) -> None:
        self._prover = prover
        self._repair_agent = repair_agent
        self._assemble_and_check = assemble_and_check_fn

    def assemble(
        self,
        request: SynthesizerAssembleRequest,
    ) -> SynthesizerAssembleResult:
        assembler = Assembler(self._prover)
        skeleton = assembler.assemble(request.cdg, request.match_results)
        return SynthesizerAssembleResult(skeleton=skeleton)

    async def compile(
        self,
        request: SynthesizerCompileRequest,
    ) -> SynthesizerCompileResult:
        compiler = SkeletonCompiler(request.env)
        result = await compiler.compile(request.skeleton)
        return SynthesizerCompileResult(result=result)

    async def assemble_and_check(
        self,
        request: SynthesizerAssembleAndCheckRequest,
    ) -> SynthesizerAssembleAndCheckResult:
        result = await self._assemble_and_check(
            request.cdg,
            request.match_results,
            request.env,
            skip_ghost_sim=request.skip_ghost_sim,
        )
        return SynthesizerAssembleAndCheckResult(result=result)

    async def repair(
        self,
        request: SynthesizerRepairRequest,
    ) -> SynthesizerRepairResult:
        if self._repair_agent is None:
            raise RuntimeError("repair_agent is required for SynthesizerService.repair")
        result = await self._repair_agent.synthesize(request.skeleton)
        return SynthesizerRepairResult(result=result)
