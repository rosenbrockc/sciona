from __future__ import annotations

from types import SimpleNamespace

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.architect.planning_contract import build_planning_artifact
from sciona.principal.models import OptimizationMetric
from sciona.principal.proposal_helpers import (
    build_expansion_context,
    evaluate_proposal_candidate,
    summarize_expansion_context,
)


@pytest.mark.asyncio
async def test_evaluate_proposal_candidate_returns_infinite_loss_on_synthesis_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="leaf",
                name="Leaf",
                description="Atomic leaf",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
            )
        ],
        edges=[],
        metadata={},
    )
    state = SimpleNamespace(
        metric=OptimizationMetric.PRECISION,
        dataset_path="dataset.yml",
    )

    async def _boom(_cdg: CDGExport, _match_results: list[object]) -> object:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(
        "sciona.principal.proposal_helpers.run_ghost_simulation",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    deps = SimpleNamespace(
        match_results_fn=lambda _cdg: [],
        synthesize_fn=_boom,
        sandbox=None,
        dataset_varset=None,
        evaluation_spec=None,
    )

    loss, bundle, benchmark, match_results, ghost_report = await evaluate_proposal_candidate(
        state,
        deps,
        cdg,
    )

    assert loss == float("inf")
    assert bundle is None
    assert benchmark is None
    assert match_results == []
    assert ghost_report is not None


def test_build_expansion_context_carries_planning_artifact() -> None:
    planning_artifact = build_planning_artifact(
        goal="Detect ECG peaks",
        thread_id="thread-1",
        paradigm="signal_detect_measure",
        variant_hint="peak_detection",
        root_inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        root_outputs=[IOSpec(name="events", type_desc="np.ndarray")],
    ).model_dump(mode="json")
    state = SimpleNamespace(
        benchmark=SimpleNamespace(
            runtime_artifacts={
                "stdout_payload": {"global_loss": 1.0},
                "intermediates": {"events": [1, 2, 3]},
                "runtime_inputs": {"features": [0.1, 0.2]},
                "signal_data": {"signal": [0.1, 0.2]},
                "runtime_evidence": {"core": {"input_count": 1}},
            }
        ),
        planning_artifact=planning_artifact,
    )

    ctx = build_expansion_context(state)

    assert ctx.planning_artifact is not None
    assert ctx.planning_artifact["artifact_version"] == "phase1.v1"
    assert ctx.runtime_inputs == {"features": [0.1, 0.2]}
    assert ctx.runtime_evidence == {"core": {"input_count": 1}}
    summary = summarize_expansion_context(ctx)
    assert summary["runtime_input_keys"] == ["features"]
    assert summary["runtime_evidence_keys"] == ["core"]
    assert summary["planning_artifact"]["paradigm"] == "signal_detect_measure"
