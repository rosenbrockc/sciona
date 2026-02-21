"""SynthesizerAgent: high-level wrapper for the repair pipeline."""

from __future__ import annotations

from ageom.hunter.llm import LLMClient
from ageom.protocols import ProofEnvironment
from ageom.synthesizer.models import SkeletonFile, SynthesisResult
from ageom.synthesizer.patcher import find_sorry_locations
from ageom.synthesizer.repair import (
    CompileCheck,
    RepairDeps,
    RepairState,
    repair_graph,
)


class SynthesizerAgent:
    """Drives the repair graph to eliminate sorrys and fix compilation errors."""

    def __init__(
        self,
        env: ProofEnvironment,
        llm: LLMClient,
        max_iterations: int = 10,
    ) -> None:
        self._deps = RepairDeps(env=env, llm=llm)
        self._max_iterations = max_iterations

    async def synthesize(self, skeleton: SkeletonFile) -> SynthesisResult:
        """Run the repair loop on a skeleton file."""
        sorry_count = len(find_sorry_locations(skeleton.source_code, skeleton.prover))

        state = RepairState(
            skeleton=skeleton,
            max_iterations=self._max_iterations,
            sorry_remaining=sorry_count,
        )

        result = await repair_graph.run(
            CompileCheck(),
            state=state,
            deps=self._deps,
        )

        final_sorrys = len(
            find_sorry_locations(result.output.source_code, result.output.prover)
        )

        return SynthesisResult(
            skeleton=result.output,
            compiled_ok=state.compiled_ok,
            sorry_remaining=final_sorrys,
            patches_applied=len(state.patches_applied),
            iterations_used=state.iteration,
            error_history=[
                (it, cat.value, text) for it, cat, text in state.error_history
            ],
        )
