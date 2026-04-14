from __future__ import annotations

import pytest

from sciona.services.skeleton_artifacts import (
    build_local_skeleton_macro_retriever,
    load_local_skeleton_macro_candidates,
)
from sciona.services.models import MacroMatchRequest


def test_local_skeleton_macro_candidates_are_deterministic() -> None:
    candidates = load_local_skeleton_macro_candidates()

    assert candidates
    assert [candidate.fqdn for candidate in candidates] == sorted(
        candidate.fqdn for candidate in candidates
    )
    signal_candidate = next(
        candidate
        for candidate in candidates
        if candidate.fqdn == "cdg.skeleton.signal_detect_measure"
    )
    assert signal_candidate.terminal_on_match is False
    assert signal_candidate.artifact_kind == "cdg"
    assert "event_rate_estimation" in signal_candidate.domain_tags
    assert signal_candidate.cdg is not None
    assert signal_candidate.cdg.metadata["artifact_fqdn"] == signal_candidate.fqdn
    assert signal_candidate.cdg.metadata["artifact_source"] == "local_skeleton_asset"


@pytest.mark.asyncio
async def test_local_skeleton_macro_retriever_matches_signal_detect_measure() -> None:
    retriever = build_local_skeleton_macro_retriever(min_score=0.3)

    result = await retriever.match_goal(
        MacroMatchRequest(goal="Detect heart rate from ECG signal")
    )

    assert result.success is True
    assert result.candidate is not None
    assert result.candidate.fqdn == "cdg.skeleton.signal_detect_measure"
    assert result.candidate.terminal_on_match is False
