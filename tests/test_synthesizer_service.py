from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.judge.models import CompilerFeedback
from sciona.services import (
    SynthesizerAssembleAndCheckRequest,
    SynthesizerAssembleRequest,
    SynthesizerCompileRequest,
    SynthesizerRepairRequest,
    SynthesizerService,
)
from sciona.synthesizer.models import AssemblyResult, SkeletonFile, SynthesisResult
from sciona.types import CandidateMatch, Declaration, MatchResult, PDGNode, Prover, VerificationResult


def _sample_cdg() -> CDGExport:
    node = AlgorithmicNode(
        node_id="leaf",
        name="Heapsort",
        description="Sort array using heapsort",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        matched_primitive="heapsort",
        type_signature="list Nat -> list Nat",
        inputs=[IOSpec(name="arr", type_desc="list Nat")],
        outputs=[IOSpec(name="sorted", type_desc="list Nat")],
        depth=1,
    )
    return CDGExport(nodes=[node], edges=[], metadata={"goal": "sort"})


def _sample_matches() -> list[MatchResult]:
    decl = Declaration(
        name="List.mergeSort",
        type_signature="list Nat -> list Nat",
        prover=Prover.LEAN4,
    )
    candidate = CandidateMatch(
        declaration=decl,
        score=0.95,
        retrieval_method="embedding",
    )
    verification = VerificationResult(
        candidate=candidate,
        verified=True,
        proof_term="@List.mergeSort",
    )
    return [
        MatchResult(
            pdg_node=PDGNode(predicate_id="leaf", statement="list Nat -> list Nat"),
            verified_match=verification,
            all_candidates=[candidate],
            all_verifications=[verification],
        )
    ]


@pytest.mark.asyncio
async def test_synthesizer_service_assemble_and_compile():
    service = SynthesizerService(prover=Prover.LEAN4)
    skeleton = service.assemble(
        SynthesizerAssembleRequest(cdg=_sample_cdg(), match_results=_sample_matches())
    ).skeleton
    env = AsyncMock()
    env.prover_name = "lean4"
    env._run = AsyncMock(
        return_value=CompilerFeedback(raw_output="ok", errors=[], warnings=[])
    )

    compile_result = await service.compile(
        SynthesizerCompileRequest(skeleton=skeleton, env=env)
    )

    assert skeleton.prover == "lean4"
    assert len(skeleton.units) == 1
    assert compile_result.result.compiled_ok is True


@pytest.mark.asyncio
async def test_synthesizer_service_compile_sanitizes_python_annotations():
    service = SynthesizerService(prover=Prover.PYTHON)
    skeleton = SkeletonFile(
        prover="python",
        source_code=(
            "def apply_filter(spec: filter specification) -> filter design targets:\n"
            "    return spec\n"
        ),
    )
    env = AsyncMock()
    env.prover_name = "python"
    env._run = AsyncMock(
        return_value=CompilerFeedback(raw_output="ok", errors=[], warnings=[])
    )

    compile_result = await service.compile(
        SynthesizerCompileRequest(skeleton=skeleton, env=env)
    )

    compiled_source = env._run.await_args_list[0].args[0]
    assert "spec: 'filter specification'" in compiled_source
    assert "-> 'filter design targets':" in compiled_source
    assert compile_result.result.compiled_ok is True


@pytest.mark.asyncio
async def test_synthesizer_service_assemble_and_check_delegates():
    fake_result = AssemblyResult(
        skeleton=SkeletonFile(prover="lean4", source_code="def x := 1"),
        compiled_ok=True,
    )

    async def _fake_assemble_and_check(cdg, match_results, env, *, skip_ghost_sim=False):
        assert skip_ghost_sim is True
        return fake_result

    service = SynthesizerService(
        prover=Prover.LEAN4,
        assemble_and_check_fn=_fake_assemble_and_check,
    )
    result = await service.assemble_and_check(
        SynthesizerAssembleAndCheckRequest(
            cdg=_sample_cdg(),
            match_results=_sample_matches(),
            env=SimpleNamespace(),
            skip_ghost_sim=True,
        )
    )

    assert result.result is fake_result


@pytest.mark.asyncio
async def test_synthesizer_service_repair_delegates_to_agent():
    synthesis = SynthesisResult(
        skeleton=SkeletonFile(prover="lean4", source_code="def x := 1"),
        compiled_ok=True,
        iterations_used=1,
    )
    repair_agent = AsyncMock()
    repair_agent.synthesize = AsyncMock(return_value=synthesis)
    service = SynthesizerService(prover=Prover.LEAN4, repair_agent=repair_agent)
    skeleton = SkeletonFile(prover="lean4", source_code="def x := 1")

    result = await service.repair(SynthesizerRepairRequest(skeleton=skeleton))

    assert result.result is synthesis
    repair_agent.synthesize.assert_awaited_once_with(skeleton)


@pytest.mark.asyncio
async def test_synthesizer_service_repair_requires_agent():
    service = SynthesizerService(prover=Prover.LEAN4)

    with pytest.raises(RuntimeError, match="repair_agent is required"):
        await service.repair(
            SynthesizerRepairRequest(
                skeleton=SkeletonFile(prover="lean4", source_code="def x := 1")
            )
        )
