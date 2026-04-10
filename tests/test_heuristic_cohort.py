from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from sciona.principal.heuristic_cohort import (
    HeuristicCohortMember,
    MaterializedHeuristicCohort,
    build_adapter_heuristic_cohort,
    materialize_heuristic_tracker_cohort,
    summarize_heuristic_cohort,
)
from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.principal.runtime_usability import build_runtime_usability_assessment
from sciona.synthesizer.models import ExportBundle


def test_materialize_heuristic_tracker_cohort_creates_subset_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "ageom.yml").write_text(
        "\n".join(
            [
                "name: Demo",
                "groups: {}",
                "meta:",
                "  source: tracker_$(tracker).csv",
                "  folder:",
                "    source: trial_name",
            ]
        )
        + "\n"
    )
    pd.DataFrame([{"trial_name": "night_1"}]).to_csv(
        dataset_root / "tracker_single.csv", index=False
    )
    pd.DataFrame(
        [{"trial_name": f"night_{idx}"} for idx in range(1, 6)]
    ).to_csv(dataset_root / "tracker_full.csv", index=False)
    for idx in range(1, 6):
        (dataset_root / f"night_{idx}").mkdir()

    cohort = materialize_heuristic_tracker_cohort(
        adapter_path=str(dataset_root / "ageom.yml"),
        varset={"tracker": "single"},
        cohort_size=3,
        output_root=tmp_path / "out",
    )

    assert cohort is not None
    assert cohort.adapter_path.exists()
    assert cohort.combined_tracker_csv.exists()
    assert len(cohort.members) == 5
    assert cohort.members[0].tracker_value == "heuristic_cohort_3_001"
    assert (cohort.dataset_root / "night_1").exists()
    assert (cohort.dataset_root / "night_2").exists()


def _runtime_usability_assessment(heuristic_id: str) -> dict[str, object]:
    evidence = {
        "runtime_context": {"tracker": heuristic_id},
        "telemetry_summary": {
            "signal": {"count": 10.0, "mean": 0.5, "std": 0.1},
            "rate": {"count": 3.0, "mean": 70.0, "std": 1.0},
        },
        "heuristics": [
            {
                "heuristic": {"heuristic_id": heuristic_id},
                "confidence": 0.8,
                "source_section": "events",
            }
        ],
        "heuristic_summary": {
            "heuristic_count": 1,
            "heuristic_ids": [heuristic_id],
            "max_confidence": 0.8,
        },
    }
    return build_runtime_usability_assessment(evidence).model_dump(mode="json")


def _runtime_usability_assessment_blocked(heuristic_id: str) -> dict[str, object]:
    evidence = {
        "runtime_context": {"tracker": heuristic_id},
        "heuristics": [
            {
                "heuristic": {"heuristic_id": heuristic_id},
                "confidence": 0.8,
                "source_section": "events",
            }
        ],
        "heuristic_summary": {
            "heuristic_count": 1,
            "heuristic_ids": [heuristic_id],
            "max_confidence": 0.8,
        },
    }
    return build_runtime_usability_assessment(evidence).model_dump(mode="json")


def _runtime_usability_assessment_unscoreable(heuristic_id: str) -> dict[str, object]:
    evidence = {
        "runtime_context": {"tracker": heuristic_id},
        "telemetry_summary": {
            "signal": {"count": 10.0, "mean": 0.5, "std": 0.1},
        },
        "heuristics": [
            {
                "heuristic": {"heuristic_id": heuristic_id},
                "confidence": 0.8,
                "source_section": "events",
            }
        ],
        "heuristic_summary": {
            "heuristic_count": 1,
            "heuristic_ids": [heuristic_id],
            "max_confidence": 0.8,
        },
    }
    return build_runtime_usability_assessment(evidence).model_dump(mode="json")


def test_summarize_heuristic_cohort_tracks_member_coverage() -> None:
    summary = summarize_heuristic_cohort(
        [
            {
                "member_label": "night_1",
                "heuristics": [
                    {
                        "heuristic": {"heuristic_id": "interval_instability"},
                        "source_section": "events",
                        "confidence": 0.7,
                    }
                ],
                "usability": _runtime_usability_assessment("interval_instability"),
            },
            {
                "member_label": "night_2",
                "heuristics": [
                    {
                        "heuristic": {"heuristic_id": "interval_instability"},
                        "source_section": "events",
                        "confidence": 0.8,
                    },
                    {
                        "heuristic": {"heuristic_id": "quality_instability"},
                        "source_section": "signal",
                        "confidence": 0.9,
                    },
                ],
                "usability": _runtime_usability_assessment("quality_instability"),
            },
        ],
        cohort_size=2,
        source_tracker_path="tracker_full.csv",
        combined_tracker_csv="tracker_heuristic_cohort_2.csv",
    )

    interval = summary["heuristics"]["interval_instability"]
    quality = summary["heuristics"]["quality_instability"]

    assert interval["member_count"] == 2
    assert interval["coverage_fraction"] == 1.0
    assert round(interval["mean_confidence"], 3) == 0.75
    assert quality["member_count"] == 1
    assert quality["coverage_fraction"] == 0.5


def test_summarize_heuristic_cohort_separates_unscoreable_gating_heuristics() -> None:
    summary = summarize_heuristic_cohort(
        [
            {
                "member_label": "night_scoreable",
                "heuristics": [
                    {
                        "heuristic": {"heuristic_id": "interval_instability"},
                        "source_section": "events",
                        "confidence": 0.7,
                    }
                ],
                "usability": _runtime_usability_assessment("interval_instability"),
            }
        ],
        cohort_size=2,
        source_tracker_path="tracker_full.csv",
        combined_tracker_csv="tracker_heuristic_cohort_2.csv",
        excluded_members=[
            {
                "member_label": "night_unscoreable",
                "heuristics": [
                    {
                        "heuristic": {"heuristic_id": "quality_instability"},
                        "source_section": "signal",
                        "confidence": 0.9,
                    }
                ],
                "usability": _runtime_usability_assessment_unscoreable(
                    "quality_instability"
                ),
            }
        ],
    )

    assert "interval_instability" in summary["heuristics"]
    assert "quality_instability" not in summary["heuristics"]
    assert "quality_instability" in summary["gating_heuristics"]
    assert summary["usability"]["proposal_basis"] == "scoring_usable_members"
    assert summary["usability"]["proposal_member_count"] == 1
    assert summary["usability"]["unscoreable_member_count"] == 1


@pytest.mark.asyncio
async def test_build_adapter_heuristic_cohort_stops_after_enough_usable_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "verified.py"
    source_path.write_text("def run_pipeline(**kwargs):\n    return kwargs\n")
    bundle = ExportBundle(target="python", output_dir=tmp_path / "out", source_path=source_path)
    cohort = MaterializedHeuristicCohort(
        dataset_root=tmp_path / "dataset",
        adapter_path=tmp_path / "dataset" / "ageom.yml",
        source_tracker_path=tmp_path / "dataset" / "tracker.csv",
        combined_tracker_csv=tmp_path / "dataset" / "tracker_heuristic_cohort_2.csv",
        members=tuple(
            HeuristicCohortMember(
                tracker_value=f"heuristic_cohort_2_{index:03d}",
                member_label=f"night_{index}",
                folder_name=f"night_{index}",
                tracker_csv=tmp_path / "dataset" / f"tracker_{index:03d}.csv",
            )
            for index in range(1, 7)
        ),
    )

    monkeypatch.setattr(
        "sciona.principal.heuristic_cohort.materialize_heuristic_tracker_cohort",
        lambda **_kwargs: cohort,
    )

    calls: list[str] = []

    class Sandbox:
        async def evaluate_adapter(self, _bundle, _adapter_path, _metric, *, varset, evaluation_spec):
            _ = evaluation_spec
            tracker = str(varset["tracker"])
            calls.append(tracker)
            return BenchmarkResult(
                global_loss=0.1,
                runtime_artifacts={
                    "heuristics": [
                        {
                            "heuristic": {"heuristic_id": f"h_{tracker}"},
                            "confidence": 0.9,
                            "source_section": "events",
                        }
                    ],
                    "usability_assessment": _runtime_usability_assessment(f"h_{tracker}"),
                    "runtime_context": {"tracker": tracker},
                    "telemetry_summary": {"signal": {"count": 10.0, "mean": 0.5}},
                },
            )

    summary = await build_adapter_heuristic_cohort(
        bundle=bundle,
        sandbox=Sandbox(),
        adapter_path=str(tmp_path / "dataset" / "ageom.yml"),
        metric=OptimizationMetric.PRECISION,
        dataset_varset={"tracker": "single"},
        evaluation_spec=None,
        cohort_size=2,
        max_concurrency=2,
    )

    assert summary is not None
    assert summary["evaluated_member_count"] == 2
    assert summary["attempted_member_count"] == 2
    assert calls == ["heuristic_cohort_2_001", "heuristic_cohort_2_002"]
    assert summary["members"][0]["usability_assessment"]["assessment_id"] == "runtime_usability_assessment"
    assert summary["usability"]["guidance_usable_member_count"] == 2
    assert summary["usability"]["excluded_member_count"] == 0
    assert summary["usability"]["final_benchmark_usable_member_count"] == 2


@pytest.mark.asyncio
async def test_build_adapter_heuristic_cohort_requires_scoreable_members_for_proposals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "verified.py"
    source_path.write_text("def run_pipeline(**kwargs):\n    return kwargs\n")
    bundle = ExportBundle(target="python", output_dir=tmp_path / "out", source_path=source_path)
    cohort = MaterializedHeuristicCohort(
        dataset_root=tmp_path / "dataset",
        adapter_path=tmp_path / "dataset" / "ageom.yml",
        source_tracker_path=tmp_path / "dataset" / "tracker.csv",
        combined_tracker_csv=tmp_path / "dataset" / "tracker_heuristic_cohort_2.csv",
        members=tuple(
            HeuristicCohortMember(
                tracker_value=f"heuristic_cohort_2_{index:03d}",
                member_label=f"night_{index}",
                folder_name=f"night_{index}",
                tracker_csv=tmp_path / "dataset" / f"tracker_{index:03d}.csv",
            )
            for index in range(1, 4)
        ),
    )

    monkeypatch.setattr(
        "sciona.principal.heuristic_cohort.materialize_heuristic_tracker_cohort",
        lambda **_kwargs: cohort,
    )

    calls: list[str] = []

    class Sandbox:
        async def evaluate_adapter(self, _bundle, _adapter_path, _metric, *, varset, evaluation_spec):
            _ = evaluation_spec
            tracker = str(varset["tracker"])
            calls.append(tracker)
            usability = (
                _runtime_usability_assessment_unscoreable(f"h_{tracker}")
                if tracker.endswith("001")
                else _runtime_usability_assessment(f"h_{tracker}")
            )
            return BenchmarkResult(
                global_loss=0.1,
                runtime_artifacts={
                    "heuristics": [
                        {
                            "heuristic": {"heuristic_id": f"h_{tracker}"},
                            "confidence": 0.9,
                            "source_section": "events",
                        }
                    ],
                    "usability_assessment": usability,
                    "runtime_context": {"tracker": tracker},
                    "telemetry_summary": {"signal": {"count": 10.0, "mean": 0.5}},
                },
            )

    summary = await build_adapter_heuristic_cohort(
        bundle=bundle,
        sandbox=Sandbox(),
        adapter_path=str(tmp_path / "dataset" / "ageom.yml"),
        metric=OptimizationMetric.PRECISION,
        dataset_varset={"tracker": "single"},
        evaluation_spec=None,
        cohort_size=2,
        max_concurrency=2,
    )

    assert summary is not None
    assert summary["evaluated_member_count"] == 2
    assert summary["attempted_member_count"] == 3
    assert calls == [
        "heuristic_cohort_2_001",
        "heuristic_cohort_2_002",
        "heuristic_cohort_2_003",
    ]
    assert summary["members"][0]["tracker_value"] == "heuristic_cohort_2_002"
    assert summary["excluded_members"][0]["tracker_value"] == "heuristic_cohort_2_001"
    assert summary["heuristics"] == {
        "h_heuristic_cohort_2_002": summary["heuristics"]["h_heuristic_cohort_2_002"],
        "h_heuristic_cohort_2_003": summary["heuristics"]["h_heuristic_cohort_2_003"],
    }
    assert "h_heuristic_cohort_2_001" in summary["gating_heuristics"]


@pytest.mark.asyncio
async def test_build_adapter_heuristic_cohort_records_usability_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "verified.py"
    source_path.write_text("def run_pipeline(**kwargs):\n    return kwargs\n")
    bundle = ExportBundle(target="python", output_dir=tmp_path / "out", source_path=source_path)
    cohort = MaterializedHeuristicCohort(
        dataset_root=tmp_path / "dataset",
        adapter_path=tmp_path / "dataset" / "ageom.yml",
        source_tracker_path=tmp_path / "dataset" / "tracker.csv",
        combined_tracker_csv=tmp_path / "dataset" / "tracker_heuristic_cohort_3.csv",
        members=tuple(
            HeuristicCohortMember(
                tracker_value=f"heuristic_cohort_3_{index:03d}",
                member_label=f"night_{index}",
                folder_name=f"night_{index}",
                tracker_csv=tmp_path / "dataset" / f"tracker_{index:03d}.csv",
            )
            for index in range(1, 4)
        ),
    )

    monkeypatch.setattr(
        "sciona.principal.heuristic_cohort.materialize_heuristic_tracker_cohort",
        lambda **_kwargs: cohort,
    )

    class Sandbox:
        async def evaluate_adapter(self, _bundle, _adapter_path, _metric, *, varset, evaluation_spec):
            _ = evaluation_spec
            tracker = str(varset["tracker"])
            if tracker.endswith("001"):
                usability = _runtime_usability_assessment_blocked(f"h_{tracker}")
            else:
                usability = _runtime_usability_assessment(f"h_{tracker}")
            return BenchmarkResult(
                global_loss=0.1,
                runtime_artifacts={
                    "heuristics": [
                        {
                            "heuristic": {"heuristic_id": f"h_{tracker}"},
                            "confidence": 0.9,
                            "source_section": "events",
                        }
                    ],
                    "usability_assessment": usability,
                    "runtime_context": {"tracker": tracker},
                    "telemetry_summary": {"signal": {"count": 10.0, "mean": 0.5}},
                },
            )

    summary = await build_adapter_heuristic_cohort(
        bundle=bundle,
        sandbox=Sandbox(),
        adapter_path=str(tmp_path / "dataset" / "ageom.yml"),
        metric=OptimizationMetric.PRECISION,
        dataset_varset={"tracker": "single"},
        evaluation_spec=None,
        cohort_size=2,
        max_concurrency=2,
    )

    assert summary is not None
    assert summary["evaluated_member_count"] == 2
    assert summary["attempted_member_count"] == 3
    assert summary["usability"]["excluded_member_count"] == 1
    assert summary["usability"]["excluded_member_labels"] == ["night_1"]
    assert summary["usability"]["guidance_usable_member_count"] == 2
    assert summary["usability"]["scoring_usable_member_count"] == 2
    assert summary["usability"]["final_benchmark_usable_member_count"] == 2
    assert summary["excluded_members"][0]["usability"]["usable_for_guidance"] is False


@pytest.mark.asyncio
async def test_build_adapter_heuristic_cohort_respects_max_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "verified.py"
    source_path.write_text("def run_pipeline(**kwargs):\n    return kwargs\n")
    bundle = ExportBundle(target="python", output_dir=tmp_path / "out", source_path=source_path)
    cohort = MaterializedHeuristicCohort(
        dataset_root=tmp_path / "dataset",
        adapter_path=tmp_path / "dataset" / "ageom.yml",
        source_tracker_path=tmp_path / "dataset" / "tracker.csv",
        combined_tracker_csv=tmp_path / "dataset" / "tracker_heuristic_cohort_4.csv",
        members=tuple(
            HeuristicCohortMember(
                tracker_value=f"heuristic_cohort_4_{index:03d}",
                member_label=f"night_{index}",
                folder_name=f"night_{index}",
                tracker_csv=tmp_path / "dataset" / f"tracker_{index:03d}.csv",
            )
            for index in range(1, 5)
        ),
    )

    monkeypatch.setattr(
        "sciona.principal.heuristic_cohort.materialize_heuristic_tracker_cohort",
        lambda **_kwargs: cohort,
    )

    active = 0
    max_active = 0

    class Sandbox:
        async def evaluate_adapter(self, _bundle, _adapter_path, _metric, *, varset, evaluation_spec):
            _ = (varset, evaluation_spec)
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return BenchmarkResult(
                global_loss=0.1,
                runtime_artifacts={
                    "heuristics": [
                        {
                            "heuristic": {"heuristic_id": "interval_instability"},
                            "confidence": 0.8,
                            "source_section": "events",
                        }
                    ],
                    "usability_assessment": _runtime_usability_assessment(
                        "interval_instability"
                    ),
                    "runtime_context": {"tracker": "single"},
                    "telemetry_summary": {"signal": {"count": 10.0, "mean": 0.5}},
                },
            )

    summary = await build_adapter_heuristic_cohort(
        bundle=bundle,
        sandbox=Sandbox(),
        adapter_path=str(tmp_path / "dataset" / "ageom.yml"),
        metric=OptimizationMetric.PRECISION,
        dataset_varset={"tracker": "single"},
        evaluation_spec=None,
        cohort_size=4,
        max_concurrency=2,
    )

    assert summary is not None
    assert summary["evaluated_member_count"] == 4
    assert max_active == 2
