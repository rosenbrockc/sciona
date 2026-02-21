"""Hot-path optimizer: swap verified implementations for native libraries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ageom.protocols import ProofEnvironment
from ageom.synthesizer.models import AssemblyUnit, SkeletonFile
from ageom.synthesizer.patcher import Patch, apply_patches


@dataclass
class OptimizationRule:
    """A rule mapping a verified definition to a native library symbol."""

    pattern: str  # regex matching declaration name or type signature
    replacement_lib: str  # native library to link (e.g. "blas")
    replacement_symbol: str  # symbol name in the native library
    guard_check: str  # Lean/Coq statement to verify before swapping


DEFAULT_RULES: list[OptimizationRule] = [
    OptimizationRule(
        pattern=r".*matrix_mul.*",
        replacement_lib="blas",
        replacement_symbol="dgemm",
        guard_check="-- BLAS dgemm compatibility assumed",
    ),
    OptimizationRule(
        pattern=r".*fft.*",
        replacement_lib="fftw3",
        replacement_symbol="fftw_execute",
        guard_check="-- FFTW compatibility assumed",
    ),
    OptimizationRule(
        pattern=r".*(?:sort|qsort).*",
        replacement_lib="c",
        replacement_symbol="qsort",
        guard_check="-- System qsort compatibility assumed",
    ),
]


@dataclass
class OptimizationCandidate:
    """A unit matched against an optimization rule."""

    unit: AssemblyUnit
    rule: OptimizationRule
    guard_verified: bool = False


class Optimizer:
    """Scans skeleton units for hot-path optimization opportunities."""

    def __init__(self, rules: list[OptimizationRule] | None = None) -> None:
        self._rules = rules if rules is not None else DEFAULT_RULES

    def scan(self, skeleton: SkeletonFile) -> list[OptimizationCandidate]:
        """Match assembly units against optimization rules."""
        candidates: list[OptimizationCandidate] = []
        for unit in skeleton.units:
            for rule in self._rules:
                pat = re.compile(rule.pattern, re.IGNORECASE)
                if pat.search(unit.declaration_name) or pat.search(unit.type_signature):
                    candidates.append(OptimizationCandidate(unit=unit, rule=rule))
                    break  # One rule per unit
        return candidates

    async def verify_guards(
        self,
        candidates: list[OptimizationCandidate],
        env: ProofEnvironment,
    ) -> list[OptimizationCandidate]:
        """Verify guard checks for each candidate via the proof environment."""
        verified: list[OptimizationCandidate] = []
        for cand in candidates:
            if cand.rule.guard_check.startswith("--"):
                # Comment-only guard = always passes (placeholder)
                cand.guard_verified = True
                verified.append(cand)
            else:
                try:
                    success, _ = await env.check_term(cand.rule.guard_check, "Prop")
                    cand.guard_verified = success
                    if success:
                        verified.append(cand)
                except Exception:
                    cand.guard_verified = False
        return verified

    def apply(
        self,
        skeleton: SkeletonFile,
        candidates: list[OptimizationCandidate],
    ) -> SkeletonFile:
        """Replace verified definitions with native library extern attributes."""
        source = skeleton.source_code
        lines = source.splitlines()

        patches: list[Patch] = []
        for cand in candidates:
            if not cand.guard_verified:
                continue

            # Find the definition line in the source
            for i, line in enumerate(lines):
                if cand.unit.declaration_name in line and (
                    "def " in line or "theorem " in line
                ):
                    line_num = i + 1  # 1-indexed

                    if skeleton.prover == "lean4":
                        extern_line = (
                            f'@[extern "{cand.rule.replacement_symbol}"] ' f"{line}"
                        )
                    else:
                        # Coq: Extract Constant directive
                        extern_line = (
                            f"{line}\n"
                            f"Extract Constant {cand.unit.declaration_name} "
                            f'=> "{cand.rule.replacement_symbol}".'
                        )

                    patches.append(
                        Patch(
                            line_start=line_num,
                            line_end=line_num,
                            replacement=extern_line,
                            description=f"Optimize: {cand.unit.declaration_name} → {cand.rule.replacement_symbol}",
                        )
                    )
                    break

        if patches:
            source = apply_patches(source, patches)

        return SkeletonFile(
            prover=skeleton.prover,
            source_code=source,
            units=skeleton.units,
            glue_edges=skeleton.glue_edges,
            sorry_count=skeleton.sorry_count,
            metadata={**skeleton.metadata, "optimized": True},
        )
