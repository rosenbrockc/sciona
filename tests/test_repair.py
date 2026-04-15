"""Tests for the Phase 2 repair agent: classifier, patcher, repair graph, agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sciona.judge.models import CompilerFeedback
from sciona.shared_context import InMemorySharedContextStore
from sciona.synthesizer.classifier import (
    ErrorCategory,
    classify_error,
    classify_feedback,
    suggest_deterministic_fix,
)
from sciona.synthesizer.models import SkeletonFile, SynthesisResult
from sciona.synthesizer.patcher import (
    Patch,
    apply_patches,
    extract_error_context,
    find_sorry_locations,
)
from sciona.synthesizer.repair import (
    CompileCheck,
    RepairDeps,
    RepairState,
    _extract_goal_from_context,
    _extract_line_number,
    _parse_patch_response,
    repair_graph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEAN_SOURCE_WITH_SORRY = """\
import Mathlib

noncomputable def merge_sort (l : List Nat) : List Nat :=
  sorry

theorem merge_sort_sorted (l : List Nat) : Sorted (merge_sort l) :=
  sorry
"""

COQ_SOURCE_WITH_ADMITTED = """\
Require Import Coq.Lists.List.

Definition merge_sort (l : list nat) : list nat.
Admitted.

Theorem merge_sort_sorted : forall l, Sorted (merge_sort l).
Admitted.
"""


def _make_skeleton(source: str, prover: str = "lean4") -> SkeletonFile:
    sorry_count = (
        source.lower().count("sorry")
        if prover == "lean4"
        else source.count("Admitted.")
    )
    return SkeletonFile(
        prover=prover,
        source_code=source,
        sorry_count=sorry_count,
    )


def _make_mock_env(feedback_sequence: list[CompilerFeedback]) -> AsyncMock:
    """Create a mock ProofEnvironment that returns feedback in sequence."""
    env = AsyncMock()
    env.prover_name = "lean4"
    env._run = AsyncMock(side_effect=feedback_sequence)
    env.close = AsyncMock()
    return env


def _make_mock_llm(responses: list[str]) -> AsyncMock:
    """Create a mock LLMClient that returns responses in sequence."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=responses)
    return llm


# ===========================================================================
# TestErrorClassifier
# ===========================================================================


class TestErrorClassifier:
    def test_classify_type_mismatch(self):
        assert (
            classify_error("type mismatch, expected List Nat, got List Int")
            == ErrorCategory.TYPE_MISMATCH
        )

    def test_classify_type_mismatch_expected_got(self):
        assert (
            classify_error("expected `Nat` but got `Int`")
            == ErrorCategory.TYPE_MISMATCH
        )

    def test_classify_missing_import(self):
        assert (
            classify_error("unknown identifier 'Nat.add_comm'")
            == ErrorCategory.MISSING_IMPORT
        )

    def test_classify_missing_import_namespace(self):
        assert (
            classify_error("unknown namespace 'Finset'") == ErrorCategory.MISSING_IMPORT
        )

    def test_classify_unsolved_goal(self):
        assert classify_error("unsolved goals") == ErrorCategory.UNSOLVED_GOAL

    def test_classify_unsolved_goal_marker(self):
        assert classify_error("⊢ Nat → Nat") == ErrorCategory.UNSOLVED_GOAL

    def test_classify_universe_mismatch(self):
        assert (
            classify_error("universe level mismatch") == ErrorCategory.UNIVERSE_MISMATCH
        )

    def test_classify_syntax(self):
        assert classify_error("expected token") == ErrorCategory.SYNTAX

    def test_classify_syntax_parse(self):
        assert classify_error("parse error at position 5") == ErrorCategory.SYNTAX

    def test_classify_unknown(self):
        assert classify_error("something completely random") == ErrorCategory.UNKNOWN

    def test_classify_feedback_errors_and_goals(self):
        feedback = CompilerFeedback(
            raw_output="",
            errors=["type mismatch", "unknown identifier 'foo'"],
            goals_remaining=["⊢ Nat"],
        )
        results = classify_feedback(feedback)
        assert len(results) == 3
        assert results[0] == (ErrorCategory.TYPE_MISMATCH, "type mismatch")
        assert results[1] == (ErrorCategory.MISSING_IMPORT, "unknown identifier 'foo'")
        assert results[2] == (ErrorCategory.UNSOLVED_GOAL, "⊢ Nat")

    def test_deterministic_fix_import_with_namespace(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.MISSING_IMPORT,
            "unknown identifier 'Nat.add_comm'",
        )
        assert fix == "open Nat"

    def test_deterministic_fix_import_bare(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.MISSING_IMPORT,
            "unknown identifier 'omega'",
        )
        assert fix is not None
        assert "import" in fix.lower() or "Mathlib" in fix

    def test_deterministic_fix_namespace(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.MISSING_IMPORT,
            "unknown namespace 'Finset'",
        )
        assert fix == "open Finset"

    def test_deterministic_fix_handles_type_mismatch(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.TYPE_MISMATCH,
            "type mismatch, expected Nat, got Int",
        )
        assert fix is not None
        assert "Int.toNat" in fix

    def test_deterministic_fix_returns_none_for_unknown(self):
        fix = suggest_deterministic_fix(
            ErrorCategory.UNKNOWN,
            "weird error",
        )
        assert fix is None


# ===========================================================================
# TestPatcher
# ===========================================================================


class TestPatcher:
    def test_apply_single_patch(self):
        source = "line1\nline2\nline3\nline4\n"
        patch = Patch(line_start=2, line_end=3, replacement="new2\nnew3")
        result = apply_patches(source, [patch])
        assert "new2\nnew3" in result
        assert "line1" in result
        assert "line4" in result
        assert "line2" not in result

    def test_apply_multiple_patches(self):
        source = "a\nb\nc\nd\ne\n"
        patches = [
            Patch(line_start=2, line_end=2, replacement="B"),
            Patch(line_start=4, line_end=4, replacement="D"),
        ]
        result = apply_patches(source, patches)
        lines = result.splitlines()
        assert lines[0] == "a"
        assert lines[1] == "B"
        assert lines[2] == "c"
        assert lines[3] == "D"
        assert lines[4] == "e"

    def test_overlapping_patches_raises(self):
        source = "a\nb\nc\n"
        patches = [
            Patch(line_start=1, line_end=2, replacement="X"),
            Patch(line_start=2, line_end=3, replacement="Y"),
        ]
        with pytest.raises(ValueError, match="Overlapping"):
            apply_patches(source, patches)

    def test_empty_patches(self):
        source = "hello\n"
        assert apply_patches(source, []) == source

    def test_find_sorry_locations_lean(self):
        locations = find_sorry_locations(LEAN_SOURCE_WITH_SORRY, "lean4")
        assert len(locations) == 2
        line_nums = [loc[0] for loc in locations]
        assert 4 in line_nums
        assert 7 in line_nums

    def test_find_sorry_locations_coq(self):
        locations = find_sorry_locations(COQ_SOURCE_WITH_ADMITTED, "coq")
        assert len(locations) == 2

    def test_extract_error_context(self):
        source = "a\nb\nc\nd\ne\nf\ng\n"
        ctx = extract_error_context(source, 4, radius=2)
        assert ">> " in ctx  # Error line marker
        assert "d" in ctx

    def test_extract_error_context_at_start(self):
        source = "a\nb\nc\n"
        ctx = extract_error_context(source, 1, radius=2)
        assert "a" in ctx


# ===========================================================================
# TestRepairHelpers
# ===========================================================================


class TestRepairHelpers:
    def test_extract_line_number_colon_format(self):
        assert _extract_line_number("file.lean:42:5: error") == 42

    def test_extract_line_number_line_keyword(self):
        assert _extract_line_number("error at line 7") == 7

    def test_extract_line_number_default(self):
        assert _extract_line_number("some error text") == 1

    def test_parse_patch_response_valid(self):
        response = json.dumps(
            {
                "line_start": 3,
                "line_end": 3,
                "replacement": "  exact Nat.add_comm",
                "description": "Fix tactic",
            }
        )
        patch = _parse_patch_response(response)
        assert patch is not None
        assert patch.line_start == 3
        assert patch.line_end == 3
        assert patch.replacement == "  exact Nat.add_comm"

    def test_parse_patch_response_with_markdown(self):
        response = (
            "```json\n"
            + json.dumps(
                {
                    "line_start": 1,
                    "line_end": 1,
                    "replacement": "import Mathlib",
                }
            )
            + "\n```"
        )
        patch = _parse_patch_response(response)
        assert patch is not None
        assert patch.line_start == 1

    def test_parse_patch_response_invalid(self):
        assert _parse_patch_response("not json at all") is None

    def test_parse_patch_response_missing_keys(self):
        assert _parse_patch_response('{"line_start": 1}') is None

    def test_extract_goal_from_context(self):
        source = "import Mathlib\n\ntheorem foo : Nat → Nat := by\n  sorry\n"
        goal = _extract_goal_from_context(source, 4)
        assert "Nat" in goal

    def test_extract_goal_from_context_no_match(self):
        source = "x = 1\ny = 2\nsorry\n"
        goal = _extract_goal_from_context(source, 3)
        assert goal == ""


# ===========================================================================
# TestRepairGraph
# ===========================================================================


class TestRepairGraph:
    @pytest.mark.asyncio
    async def test_happy_path_no_errors(self):
        """Skeleton compiles on first try."""
        skeleton = _make_skeleton("-- correct code\n")
        env = _make_mock_env(
            [
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm([])

        state = RepairState(skeleton=skeleton, max_iterations=5)
        result = await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        assert state.compiled_ok is True
        assert state.iteration == 0
        assert result.output.source_code == "-- correct code\n"

    @pytest.mark.asyncio
    async def test_deterministic_fix_resolves(self):
        """Missing import → deterministic fix → compiles."""
        skeleton = _make_skeleton("def foo := Nat.add_comm\n")
        env = _make_mock_env(
            [
                # First compile: missing import
                CompilerFeedback(
                    raw_output="unknown identifier 'Nat.add_comm'",
                    errors=["unknown identifier 'Nat.add_comm'"],
                ),
                # Second compile: success
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm([])

        state = RepairState(skeleton=skeleton, max_iterations=5)
        result = await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        assert state.compiled_ok is True
        assert state.iteration == 1
        assert len(state.patches_applied) == 1
        assert "open Nat" in result.output.source_code

    @pytest.mark.asyncio
    async def test_llm_repair_type_mismatch(self):
        """Type mismatch → LLM generates patch → compiles."""
        skeleton = _make_skeleton("def foo : Nat := (1 : Int)\n")
        patch_response = json.dumps(
            {
                "line_start": 1,
                "line_end": 1,
                "replacement": "def foo : Nat := 1",
                "description": "Remove type annotation",
            }
        )
        env = _make_mock_env(
            [
                # First compile: type mismatch
                CompilerFeedback(
                    raw_output="type mismatch",
                    errors=["type mismatch, expected Nat, got Int"],
                ),
                # Second compile: success
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm([patch_response])

        state = RepairState(skeleton=skeleton, max_iterations=5)
        await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        assert state.compiled_ok is True
        assert state.iteration == 1

    @pytest.mark.asyncio
    async def test_sorry_elimination(self):
        """Skeleton with sorry → LLM generates tactic → compiles."""
        source = "theorem foo : 1 + 1 = 2 := by\n  sorry\n"
        skeleton = _make_skeleton(source)
        env = _make_mock_env(
            [
                # First compile: unsolved goals (sorry)
                CompilerFeedback(
                    raw_output="unsolved goals\n⊢ 1 + 1 = 2",
                    errors=[],
                    goals_remaining=["⊢ 1 + 1 = 2"],
                ),
                # Second compile: success
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm(["omega"])

        state = RepairState(skeleton=skeleton, max_iterations=5, sorry_remaining=1)
        result = await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        assert state.compiled_ok is True
        assert "sorry" not in result.output.source_code

    @pytest.mark.asyncio
    async def test_budget_exhaustion(self):
        """Errors persist → agent stops at max_iterations."""
        skeleton = _make_skeleton("broken code\n")
        # Always return errors
        error_fb = CompilerFeedback(
            raw_output="type mismatch",
            errors=["type mismatch"],
        )
        env = _make_mock_env([error_fb] * 5)

        # LLM always returns invalid response
        llm = _make_mock_llm(["not valid json"] * 5)

        state = RepairState(skeleton=skeleton, max_iterations=3)
        await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        assert state.compiled_ok is False
        assert state.iteration >= 3

    @pytest.mark.asyncio
    async def test_mixed_errors(self):
        """Deterministic fix + LLM fix in one session."""
        source = "def bar := unknown_func\ndef foo : Nat := (1 : Int)\n"
        skeleton = _make_skeleton(source)

        patch_response = json.dumps(
            {
                "line_start": 2,
                "line_end": 2,
                "replacement": "def foo : Nat := 1",
                "description": "Fix type",
            }
        )

        env = _make_mock_env(
            [
                # First: missing import
                CompilerFeedback(
                    raw_output="",
                    errors=["unknown identifier 'unknown_func'"],
                ),
                # Second: type mismatch (import fixed)
                CompilerFeedback(
                    raw_output="",
                    errors=["type mismatch, expected Nat, got Int"],
                ),
                # Third: success
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm([patch_response])

        state = RepairState(skeleton=skeleton, max_iterations=10)
        await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        assert state.compiled_ok is True
        assert state.iteration == 2

    @pytest.mark.asyncio
    async def test_llm_repair_uses_and_writes_shared_context(self):
        """LLM repair prompt includes shared context and writes repair memory."""
        skeleton = _make_skeleton("def foo : Nat := (1 : Int)\n")
        patch_response = json.dumps(
            {
                "line_start": 1,
                "line_end": 1,
                "replacement": "def foo : Nat := 1",
                "description": "Fix cast",
            }
        )
        env = _make_mock_env(
            [
                CompilerFeedback(
                    raw_output="application type mismatch",
                    errors=["application type mismatch in complex term elaboration"],
                ),
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = AsyncMock()
        captured_users: list[str] = []

        async def complete(system: str, user: str) -> str:
            captured_users.append(user)
            return patch_response

        llm.complete = complete

        store = InMemorySharedContextStore()
        await store.put(
            "synth/test/repair",
            "For Nat/Int mismatch, prefer a direct Nat literal when safe.",
        )

        state = RepairState(skeleton=skeleton, max_iterations=5)
        await repair_graph.run(
            CompileCheck(),
            state=state,
            deps=RepairDeps(
                env=env,
                llm=llm,
                shared_context=store,
                context_namespace="synth/test",
            ),
        )

        assert state.compiled_ok is True
        assert captured_users
        assert "Shared Context" in captured_users[0]
        records = await store.recent("synth/test/repair", limit=5)
        assert any("Patch:" in r.text for r in records)

    @pytest.mark.asyncio
    async def test_compile_check_sanitizes_python_annotations_before_compile(self):
        """Python repair should not loop on conceptual annotation syntax."""
        skeleton = _make_skeleton(
            "def apply_filter(spec: filter specification) -> filter design targets:\n"
            "    return spec\n",
            prover="python",
        )
        env = AsyncMock()
        env.prover_name = "python"
        env._run = AsyncMock(
            return_value=CompilerFeedback(raw_output="", errors=[], goals_remaining=[])
        )
        env.close = AsyncMock()
        llm = _make_mock_llm([])

        state = RepairState(skeleton=skeleton, max_iterations=3)
        result = await repair_graph.run(
            CompileCheck(),
            state=state,
            deps=RepairDeps(env=env, llm=llm),
        )

        compiled_source = env._run.await_args_list[0].args[0]
        assert "spec: 'filter specification'" in compiled_source
        assert "-> 'filter design targets':" in compiled_source
        assert state.compiled_ok is True
        assert result.output.source_code == compiled_source


# ===========================================================================
# TestSynthesizerAgent
# ===========================================================================


class TestSynthesizerAgent:
    @pytest.mark.asyncio
    async def test_synthesize_end_to_end(self):
        """Full pipeline: skeleton → repair → SynthesisResult."""
        from sciona.synthesizer.agent import SynthesizerAgent

        skeleton = _make_skeleton("-- clean code\n")
        env = _make_mock_env(
            [
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm([])

        agent = SynthesizerAgent(env=env, llm=llm, max_iterations=5)
        result = await agent.synthesize(skeleton)

        assert isinstance(result, SynthesisResult)
        assert result.compiled_ok is True
        assert result.sorry_remaining == 0
        assert result.iterations_used == 0

    @pytest.mark.asyncio
    async def test_synthesize_preserves_correct_code(self):
        """Patches don't corrupt working definitions."""
        from sciona.synthesizer.agent import SynthesizerAgent

        source = "def good : Nat := 42\ndef bad : Nat := (1 : Int)\n"
        skeleton = _make_skeleton(source)

        patch_response = json.dumps(
            {
                "line_start": 2,
                "line_end": 2,
                "replacement": "def bad : Nat := 1",
                "description": "Fix cast",
            }
        )

        env = _make_mock_env(
            [
                CompilerFeedback(
                    raw_output="",
                    errors=["type mismatch on line 2"],
                ),
                CompilerFeedback(raw_output="", errors=[], goals_remaining=[]),
            ]
        )
        llm = _make_mock_llm([patch_response])

        agent = SynthesizerAgent(env=env, llm=llm, max_iterations=5)
        result = await agent.synthesize(skeleton)

        assert result.compiled_ok is True
        assert "def good : Nat := 42" in result.skeleton.source_code

    @pytest.mark.asyncio
    async def test_synthesize_budget_exhaustion(self):
        """Agent stops after max_iterations even with remaining errors."""
        from sciona.synthesizer.agent import SynthesizerAgent

        skeleton = _make_skeleton("broken\n")
        env = _make_mock_env(
            [
                CompilerFeedback(raw_output="", errors=["syntax error"])
                for _ in range(10)
            ]
        )
        llm = _make_mock_llm(["invalid"] * 10)

        agent = SynthesizerAgent(env=env, llm=llm, max_iterations=2)
        result = await agent.synthesize(skeleton)

        assert result.compiled_ok is False
        assert result.iterations_used >= 2


# ===========================================================================
# TestCLI
# ===========================================================================


class TestCLIParserAcceptsSynthesize:
    def test_synthesize_subcommand_exists(self):
        """The CLI parser accepts the 'synthesize' subcommand."""

        import argparse
        import sys
        from unittest.mock import patch

        with patch.object(
            sys, "argv", ["sciona", "synthesize", "cdg.json", "matches.json"]
        ):

            parser = argparse.ArgumentParser(prog="sciona")
            subparsers = parser.add_subparsers(dest="command")
            synth = subparsers.add_parser("synthesize")
            synth.add_argument("cdg_file")
            synth.add_argument("matches_file")
            synth.add_argument("--prover", default="lean4")
            synth.add_argument("--output", default=None)
            synth.add_argument("--max-iterations", type=int, default=None)

            args = parser.parse_args(["synthesize", "cdg.json", "matches.json"])
            assert args.command == "synthesize"
            assert args.cdg_file == "cdg.json"
            assert args.matches_file == "matches.json"


# ===========================================================================
# TestRepairResilience — silent-failure / regression-rollback / budget paths
# ===========================================================================


class TestRepairResilience:
    @pytest.mark.asyncio
    async def test_repair_regression_rollback_restores_previous_source(self):
        """When an LLM patch causes more errors, the graph rolls back to the
        pre-patch snapshot and does NOT continue from the regressed version."""
        original_source = "def foo := original\n"
        skeleton = _make_skeleton(original_source)

        # LLM returns a patch that will be applied (mutating source)
        patch_response = json.dumps(
            {
                "line_start": 1,
                "line_end": 1,
                "replacement": "def foo := patched_worse",
                "description": "Bad patch",
            }
        )

        env = _make_mock_env(
            [
                # Iteration 0 compile: 1 error
                CompilerFeedback(
                    raw_output="err",
                    errors=["type mismatch"],
                ),
                # Iteration 1 compile (after LLM patch applied): 3 errors (regression!)
                CompilerFeedback(
                    raw_output="err",
                    errors=["type mismatch", "unknown id", "syntax error"],
                ),
                # Re-compile after rollback: still 1 error (back to snapshot)
                CompilerFeedback(
                    raw_output="err",
                    errors=["type mismatch"],
                ),
                # Budget exhausted at iteration 2 — need one more compile
                CompilerFeedback(
                    raw_output="err",
                    errors=["type mismatch"],
                ),
                # Extra safety compile in case graph routes again
                CompilerFeedback(
                    raw_output="err",
                    errors=["type mismatch"],
                ),
            ]
        )
        llm = _make_mock_llm([patch_response, "invalid"] * 3)

        state = RepairState(skeleton=skeleton, max_iterations=2)
        result = await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        # The final source must NOT contain the regressed patch
        assert "patched_worse" not in result.output.source_code
        # It should have rolled back to the original (snapshot) source
        assert "original" in result.output.source_code

    @pytest.mark.asyncio
    async def test_repair_budget_exhaustion_recovers_best_source(self):
        """When budget exhausts, the graph returns the best-seen source (lowest
        error count), not whatever source was current at exhaustion time."""
        source_v0 = "def foo := v0\n"  # 3 errors
        source_v1 = "def foo := v1\n"  # 1 error (best)
        source_v2 = "def foo := v2\n"  # 5 errors (worst)

        skeleton = _make_skeleton(source_v0)

        # Patches that change source in a predictable way
        patch_v1 = json.dumps(
            {
                "line_start": 1,
                "line_end": 1,
                "replacement": "def foo := v1",
                "description": "Patch to v1",
            }
        )
        patch_v2 = json.dumps(
            {
                "line_start": 1,
                "line_end": 1,
                "replacement": "def foo := v2",
                "description": "Patch to v2",
            }
        )

        env = _make_mock_env(
            [
                # Iteration 0 compile: v0 has 3 errors
                CompilerFeedback(
                    raw_output="err",
                    errors=["e1", "e2", "e3"],
                ),
                # Iteration 1 compile: v1 has 1 error (improvement, no rollback)
                CompilerFeedback(
                    raw_output="err",
                    errors=["e1"],
                ),
                # Iteration 2 compile: v2 has 5 errors (regression -> rollback to v1)
                CompilerFeedback(
                    raw_output="err",
                    errors=["e1", "e2", "e3", "e4", "e5"],
                ),
                # Re-compile after rollback to v1: 1 error
                CompilerFeedback(
                    raw_output="err",
                    errors=["e1"],
                ),
                # Budget exhausted at iteration 3 — extra compiles for safety
                CompilerFeedback(
                    raw_output="err",
                    errors=["e1"],
                ),
                CompilerFeedback(
                    raw_output="err",
                    errors=["e1"],
                ),
            ]
        )
        llm = _make_mock_llm([patch_v1, patch_v2, "invalid"] * 3)

        state = RepairState(skeleton=skeleton, max_iterations=3)
        result = await repair_graph.run(
            CompileCheck(), state=state, deps=RepairDeps(env=env, llm=llm)
        )

        # Best source was v1 (1 error). Budget exhaustion should recover it.
        assert state.best_error_count == 1
        assert "v1" in result.output.source_code
        # Must NOT end on the v2 (regressed) version
        assert "v2" not in result.output.source_code
