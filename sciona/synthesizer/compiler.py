"""Wrapper around ProofEnvironment for whole-file and per-unit compilation."""

from __future__ import annotations

from sciona.judge.models import CompilerFeedback
from sciona.protocols import ProofEnvironment
from sciona.synthesizer.assembler import sanitize_python_source_annotations
from sciona.synthesizer.dim_checker import check_dimensional_consistency
from sciona.synthesizer.models import AssemblyResult, AssemblyUnit, SkeletonFile


class SkeletonCompiler:
    """Compiles skeleton files via a ProofEnvironment."""

    def __init__(self, env: ProofEnvironment) -> None:
        self._env = env

    async def compile(
        self,
        skeleton: SkeletonFile,
        *,
        skip_dim_check: bool = False,
    ) -> AssemblyResult:
        """Compile the full skeleton source and return the result.

        Args:
            skeleton: The skeleton to compile.
            skip_dim_check: If ``True``, skip dimensional analysis.
                Useful as a backward-compat escape hatch during migration.
        """
        # --- Phase 0: Dimensional analysis (pre-synthesis) ---
        dim_errors: list[str] = []
        if not skip_dim_check and skeleton.units and skeleton.glue_edges:
            dim_result = check_dimensional_consistency(
                skeleton.units, skeleton.glue_edges,
            )
            if not dim_result.passed:
                dim_errors = [e.message for e in dim_result.errors]

        if skeleton.prover == "python":
            skeleton = skeleton.model_copy(
                update={
                    "source_code": sanitize_python_source_annotations(
                        skeleton.source_code
                    )
                }
            )

        # --- Phase 1: Proof-environment / syntax check ---
        # Use check_proof with a dummy wrapper so we can send arbitrary code.
        # The env._run() method handles raw code execution under the hood.
        # We use check_term with a trivial expression to validate the file as
        # a whole — but since ProofEnvironment only exposes check_term and
        # check_proof, we send the source as a proof body for a trivial goal.
        # In practice, we call the internal _run if available, else fall back.
        if hasattr(self._env, "_run"):
            feedback = await self._env._run(skeleton.source_code)  # type: ignore[attr-defined]
        else:
            # Fallback: use check_proof with the source embedded
            success, output = await self._env.check_proof("True", skeleton.source_code)
            feedback = CompilerFeedback(
                raw_output=output,
                errors=[] if success else [output],
            )

        # Merge dimensional errors into compiler feedback
        if dim_errors:
            all_errors = dim_errors + (feedback.errors or [])
            feedback = CompilerFeedback(
                raw_output=feedback.raw_output,
                errors=all_errors,
            )

        return AssemblyResult(
            skeleton=skeleton,
            feedback=feedback,
            compiled_ok=feedback.success and not dim_errors,
        )

    async def check_unit(self, unit: AssemblyUnit) -> CompilerFeedback:
        """Compile a single unit's definition in isolation."""
        success, output = await self._env.check_term(
            f"@{unit.declaration_name}", unit.type_signature
        )
        return CompilerFeedback(
            raw_output=output,
            errors=[] if success else [output],
        )
