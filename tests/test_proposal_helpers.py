from __future__ import annotations

from types import SimpleNamespace

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.principal.models import OptimizationMetric
from sciona.principal.proposal_helpers import evaluate_proposal_candidate


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
