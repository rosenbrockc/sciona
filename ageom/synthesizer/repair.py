"""Pydantic-graph repair state machine for skeleton compilation.

Graph topology:
    CompileCheck -> DeterministicFix -> CompileCheck
                 -> LLMRepair -> CompileCheck
                 -> SorryElimination -> CompileCheck
                 -> End[SkeletonFile]  (on success or budget exhausted)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ageom.hunter.llm import LLMClient
from ageom.judge.models import CompilerFeedback
from ageom.protocols import ProofEnvironment
from ageom.synthesizer.classifier import (
    ErrorCategory,
    classify_feedback,
    suggest_deterministic_fix,
)
from ageom.synthesizer.models import SkeletonFile
from ageom.synthesizer.patcher import (
    Patch,
    apply_patches,
    extract_error_context,
    find_sorry_locations,
)
from ageom.synthesizer.prompts import (
    ANALYZE_ERROR_SYSTEM,
    ANALYZE_ERROR_USER,
    GENERATE_TACTIC_SYSTEM,
    GENERATE_TACTIC_USER,
)

# Error category priority for LLM repair (lower = higher priority)
_ERROR_PRIORITY: dict[ErrorCategory, int] = {
    ErrorCategory.TYPE_MISMATCH: 0,
    ErrorCategory.UNIVERSE_MISMATCH: 1,
    ErrorCategory.MISSING_IMPORT: 2,
    ErrorCategory.SYNTAX: 3,
    ErrorCategory.UNKNOWN: 4,
    ErrorCategory.UNSOLVED_GOAL: 5,
}


@dataclass
class RepairState:
    """Mutable state threaded through the repair graph."""

    skeleton: SkeletonFile
    max_iterations: int = 10
    iteration: int = 0
    patches_applied: list[Patch] = field(default_factory=list)
    error_history: list[tuple[int, ErrorCategory, str]] = field(default_factory=list)
    sorry_remaining: int = 0
    compiled_ok: bool = False
    _last_feedback: CompilerFeedback | None = field(default=None, repr=False)


@dataclass
class RepairDeps:
    """External dependencies for the repair graph."""

    env: ProofEnvironment
    llm: LLMClient


async def _compile_source(env: ProofEnvironment, source: str) -> CompilerFeedback:
    """Compile source code via the proof environment."""
    if hasattr(env, "_run"):
        return await env._run(source)  # type: ignore[attr-defined]
    success, output = await env.check_proof("True", source)
    return CompilerFeedback(
        raw_output=output,
        errors=[] if success else [output],
    )


@dataclass
class CompileCheck(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Compile the skeleton and route based on result."""

    async def run(
        self, ctx: GraphRunContext[RepairState, RepairDeps]
    ) -> DeterministicFix | LLMRepair | SorryElimination | End[SkeletonFile]:
        state = ctx.state
        deps = ctx.deps

        feedback = await _compile_source(deps.env, state.skeleton.source_code)
        state._last_feedback = feedback

        if feedback.success:
            state.compiled_ok = True
            state.sorry_remaining = 0
            return End(state.skeleton)

        # Budget check
        if state.iteration >= state.max_iterations:
            return End(state.skeleton)

        # Classify errors
        classified = classify_feedback(feedback)

        # Check if any have deterministic fixes
        has_deterministic = any(
            suggest_deterministic_fix(cat, text) is not None
            for cat, text in classified
        )
        if has_deterministic:
            return DeterministicFix()

        # Check if only unsolved goals remain (no hard errors)
        hard_errors = [
            (cat, text)
            for cat, text in classified
            if cat != ErrorCategory.UNSOLVED_GOAL
        ]
        if not hard_errors and classified:
            return SorryElimination()

        return LLMRepair()


@dataclass
class DeterministicFix(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Apply deterministic fixes (e.g. missing imports)."""

    async def run(
        self, ctx: GraphRunContext[RepairState, RepairDeps]
    ) -> CompileCheck:
        state = ctx.state

        if state._last_feedback is None:
            state.iteration += 1
            return CompileCheck()

        classified = classify_feedback(state._last_feedback)
        lines = state.skeleton.source_code.splitlines()

        for cat, text in classified:
            fix = suggest_deterministic_fix(cat, text)
            if fix is not None:
                # Insert fix at top of file (after first line)
                patch = Patch(
                    line_start=1,
                    line_end=1,
                    replacement=lines[0] + "\n" + fix if lines else fix,
                    description=f"Deterministic fix: {fix}",
                )
                state.skeleton.source_code = apply_patches(
                    state.skeleton.source_code, [patch]
                )
                state.patches_applied.append(patch)
                state.error_history.append((state.iteration, cat, text))
                break  # One fix per iteration to avoid patch conflicts

        state.iteration += 1
        return CompileCheck()


@dataclass
class LLMRepair(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Use LLM to generate a patch for the highest-priority error."""

    async def run(
        self, ctx: GraphRunContext[RepairState, RepairDeps]
    ) -> CompileCheck:
        state = ctx.state
        deps = ctx.deps

        if state._last_feedback is None:
            state.iteration += 1
            return CompileCheck()

        classified = classify_feedback(state._last_feedback)
        classified.sort(key=lambda x: _ERROR_PRIORITY.get(x[0], 99))

        if not classified:
            state.iteration += 1
            return CompileCheck()

        cat, error_text = classified[0]
        state.error_history.append((state.iteration, cat, error_text))

        error_line = _extract_line_number(error_text)
        context = extract_error_context(
            state.skeleton.source_code, error_line, radius=3
        )

        user_msg = ANALYZE_ERROR_USER.format(
            source_code=state.skeleton.source_code,
            error_text=error_text,
            error_category=cat.value,
            error_context=context,
        )

        try:
            response = await deps.llm.complete(ANALYZE_ERROR_SYSTEM, user_msg)
            patch = _parse_patch_response(response)
            if patch is not None:
                state.skeleton.source_code = apply_patches(
                    state.skeleton.source_code, [patch]
                )
                state.patches_applied.append(patch)
        except Exception:
            pass  # LLM or parse failure — move on

        state.iteration += 1
        return CompileCheck()


@dataclass
class SorryElimination(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Replace sorry/Admitted placeholders with LLM-generated tactic proofs."""

    async def run(
        self, ctx: GraphRunContext[RepairState, RepairDeps]
    ) -> CompileCheck:
        state = ctx.state
        deps = ctx.deps

        locations = find_sorry_locations(
            state.skeleton.source_code, state.skeleton.prover
        )
        if not locations:
            return CompileCheck()

        line_num, context = locations[0]  # Fix one sorry per iteration

        goal_type = _extract_goal_from_context(
            state.skeleton.source_code, line_num
        )

        user_msg = GENERATE_TACTIC_USER.format(
            goal_type=goal_type or "(unknown — see context above)",
            hypotheses=context,
            available_lemmas="(use standard Mathlib tactics)",
        )

        try:
            response = await deps.llm.complete(GENERATE_TACTIC_SYSTEM, user_msg)
            tactic_body = response.strip()
            if tactic_body and "sorry" not in tactic_body.lower():
                patch = Patch(
                    line_start=line_num,
                    line_end=line_num,
                    replacement=tactic_body,
                    description=f"Sorry elimination at line {line_num}",
                )
                state.skeleton.source_code = apply_patches(
                    state.skeleton.source_code, [patch]
                )
                state.patches_applied.append(patch)
        except Exception:
            pass

        state.sorry_remaining = len(locations) - 1
        state.iteration += 1
        return CompileCheck()


# --- Helpers ---


def _extract_line_number(error_text: str) -> int:
    """Extract a line number from an error message, defaulting to 1."""
    m = re.search(r":(\d+):", error_text)
    if m:
        return int(m.group(1))
    m = re.search(r"line (\d+)", error_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1


def _parse_patch_response(response: str) -> Patch | None:
    """Parse an LLM JSON response into a Patch."""
    response = response.strip()
    # Handle markdown code blocks
    if "```" in response:
        lines = response.splitlines()
        json_lines: list[str] = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip() == "```" and in_block:
                break
            if in_block:
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        data = json.loads(response)
        return Patch(
            line_start=int(data["line_start"]),
            line_end=int(data["line_end"]),
            replacement=str(data["replacement"]),
            description=str(data.get("description", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _extract_goal_from_context(source: str, sorry_line: int) -> str:
    """Try to extract the theorem/lemma goal type from lines before a sorry."""
    lines = source.splitlines()
    for i in range(sorry_line - 1, max(sorry_line - 20, -1), -1):
        if i < 0 or i >= len(lines):
            continue
        line = lines[i]
        m = re.match(
            r"\s*(?:theorem|lemma|def|noncomputable\s+def)\s+\w+\s*.*?:\s*(.*)",
            line,
        )
        if m:
            goal = m.group(1).strip()
            goal = re.sub(r"\s*:=\s*(by)?\s*$", "", goal)
            if goal:
                return goal
    return ""


repair_graph: Graph[RepairState, RepairDeps, SkeletonFile] = Graph(
    nodes=[CompileCheck, DeterministicFix, LLMRepair, SorryElimination]
)
