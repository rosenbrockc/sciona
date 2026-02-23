"""Pydantic-graph repair state machine for skeleton compilation.

Graph topology:
    CompileCheck -> DeterministicFix -> CompileCheck
                 -> LLMRepair -> CompileCheck
                 -> SorryElimination -> CompileCheck
                 -> End[SkeletonFile]  (on success or budget exhausted)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ageom.json_utils import extract_json

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
from ageom.llm_router import SYNTHESIZER_REPAIR, SYNTHESIZER_TACTIC, select_llm
from ageom.synthesizer.prompts import (
    ANALYZE_ERROR_SYSTEM,
    ANALYZE_ERROR_SYSTEM_PYTHON,
    ANALYZE_ERROR_USER,
    GENERATE_IMPLEMENTATION_SYSTEM_PYTHON,
    GENERATE_TACTIC_SYSTEM,
    GENERATE_TACTIC_USER,
)

logger = logging.getLogger(__name__)

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

    # Resilience tracking (Issue 8)
    _source_snapshots: list[str] = field(default_factory=list, repr=False)
    _last_error_count: int = field(default=-1, repr=False)
    llm_attempts: int = 0
    llm_successes: int = 0  # produced usable patches
    best_source: str = ""
    best_error_count: int = field(default=999999, repr=False)


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

        # Track error count for regression detection
        current_error_count = len(feedback.errors) if not feedback.success else 0

        # Regression detection: if errors increased, rollback
        if (
            state._last_error_count >= 0
            and current_error_count > state._last_error_count
            and state._source_snapshots
        ):
            logger.info(
                "Regression detected: errors %d -> %d, rolling back",
                state._last_error_count,
                current_error_count,
            )
            state.skeleton.source_code = state._source_snapshots[-1]
            # Re-compile after rollback
            feedback = await _compile_source(deps.env, state.skeleton.source_code)
            state._last_feedback = feedback
            current_error_count = len(feedback.errors) if not feedback.success else 0

        state._last_error_count = current_error_count

        # Track best source
        if current_error_count < state.best_error_count:
            state.best_error_count = current_error_count
            state.best_source = state.skeleton.source_code

        if feedback.success:
            state.compiled_ok = True
            state.sorry_remaining = 0
            return End(state.skeleton)

        # Budget check — return best version if exhausted
        if state.iteration >= state.max_iterations:
            if state.best_source and state.best_error_count < current_error_count:
                state.skeleton.source_code = state.best_source
            return End(state.skeleton)

        # Snapshot current source before any modifications
        state._source_snapshots.append(state.skeleton.source_code)

        # Classify errors
        classified = classify_feedback(feedback)

        # Check if any have deterministic fixes
        has_deterministic = any(
            suggest_deterministic_fix(cat, text) is not None for cat, text in classified
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
    """Apply deterministic fixes (e.g. missing imports).

    Applies ALL deterministic fixes in one pass instead of one-per-iteration.
    """

    async def run(self, ctx: GraphRunContext[RepairState, RepairDeps]) -> CompileCheck:
        state = ctx.state

        if state._last_feedback is None:
            state.iteration += 1
            return CompileCheck()

        classified = classify_feedback(state._last_feedback)
        lines = state.skeleton.source_code.splitlines()

        # Collect ALL deterministic fixes in one pass
        fixes_to_apply: list[str] = []
        for cat, text in classified:
            fix = suggest_deterministic_fix(cat, text)
            if fix is not None:
                if fix not in fixes_to_apply:
                    fixes_to_apply.append(fix)
                state.error_history.append((state.iteration, cat, text))

        if fixes_to_apply:
            # Insert all fixes at top of file (after first line)
            combined_fix = "\n".join(fixes_to_apply)
            patch = Patch(
                line_start=1,
                line_end=1,
                replacement=lines[0] + "\n" + combined_fix if lines else combined_fix,
                description=f"Deterministic fixes: {len(fixes_to_apply)} fix(es)",
            )
            state.skeleton.source_code = apply_patches(
                state.skeleton.source_code, [patch]
            )
            state.patches_applied.append(patch)

        state.iteration += 1
        return CompileCheck()


@dataclass
class LLMRepair(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Use LLM to generate a patch for the highest-priority error."""

    async def run(self, ctx: GraphRunContext[RepairState, RepairDeps]) -> CompileCheck:
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

        system_prompt = (
            ANALYZE_ERROR_SYSTEM_PYTHON
            if state.skeleton.prover == "python"
            else ANALYZE_ERROR_SYSTEM
        )

        state.llm_attempts += 1
        try:
            response = await select_llm(deps.llm, SYNTHESIZER_REPAIR).complete(
                system_prompt, user_msg
            )
            patch = _parse_patch_response(response)
            if patch is not None:
                state.skeleton.source_code = apply_patches(
                    state.skeleton.source_code, [patch]
                )
                state.patches_applied.append(patch)
                state.llm_successes += 1
        except json.JSONDecodeError as exc:
            state.error_history.append(
                (
                    state.iteration,
                    ErrorCategory.UNKNOWN,
                    f"LLM_FAILURE: JSON parse error: {exc}",
                )
            )
            logger.warning("LLM repair JSON parse error: %s", exc)
        except ValueError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("LLM repair value error: %s", exc)
        except RuntimeError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("LLM repair runtime error: %s", exc)

        state.iteration += 1
        return CompileCheck()


@dataclass
class SorryElimination(BaseNode[RepairState, RepairDeps, SkeletonFile]):
    """Replace sorry/Admitted placeholders with LLM-generated tactic proofs."""

    async def run(self, ctx: GraphRunContext[RepairState, RepairDeps]) -> CompileCheck:
        state = ctx.state
        deps = ctx.deps

        locations = find_sorry_locations(
            state.skeleton.source_code, state.skeleton.prover
        )
        if not locations:
            return CompileCheck()

        line_num, context = locations[0]  # Fix one sorry per iteration

        goal_type = _extract_goal_from_context(state.skeleton.source_code, line_num)

        user_msg = GENERATE_TACTIC_USER.format(
            goal_type=goal_type or "(unknown — see context above)",
            hypotheses=context,
            available_lemmas="(use standard Mathlib tactics)",
        )

        system_prompt = (
            GENERATE_IMPLEMENTATION_SYSTEM_PYTHON
            if state.skeleton.prover == "python"
            else GENERATE_TACTIC_SYSTEM
        )

        state.llm_attempts += 1
        try:
            response = await select_llm(deps.llm, SYNTHESIZER_TACTIC).complete(
                system_prompt, user_msg
            )
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
                state.llm_successes += 1
        except json.JSONDecodeError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"PARSE_FAILURE: {exc}")
            )
            logger.warning("Sorry elimination parse error: %s", exc)
        except ValueError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("Sorry elimination value error: %s", exc)
        except RuntimeError as exc:
            state.error_history.append(
                (state.iteration, ErrorCategory.UNKNOWN, f"LLM_FAILURE: {exc}")
            )
            logger.warning("Sorry elimination runtime error: %s", exc)

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
    try:
        data = extract_json(response)
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
        if not m:
            # Python: def func_name(params) -> ReturnType:
            m = re.match(
                r"\s*def\s+\w+\s*\(.*?\)\s*(?:->\s*(.+?))?\s*:",
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
