"""Wrapper around ProofEnvironment for whole-file and per-unit compilation."""

from __future__ import annotations

from sciona.judge.models import CompilerFeedback
from sciona.protocols import ProofEnvironment
from sciona.synthesizer.assembler import sanitize_python_source_annotations
from sciona.synthesizer.models import AssemblyResult, AssemblyUnit, SkeletonFile


class SkeletonCompiler:
    """Compiles skeleton files via a ProofEnvironment."""

    def __init__(self, env: ProofEnvironment) -> None:
        self._env = env

    async def compile(self, skeleton: SkeletonFile) -> AssemblyResult:
        """Compile the full skeleton source and return the result."""
        if skeleton.prover == "python":
            skeleton = skeleton.model_copy(
                update={
                    "source_code": sanitize_python_source_annotations(
                        skeleton.source_code
                    )
                }
            )

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

        return AssemblyResult(
            skeleton=skeleton,
            feedback=feedback,
            compiled_ok=feedback.success,
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
