from __future__ import annotations

from pathlib import Path

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from ageom.commands.optimize_cmds import _parse_dataset_vars
from ageom.principal.models import BenchmarkResult, OptimizationMetric
from ageom.principal.profiler import profile_algorithm_error
from ageom.synthesizer.models import ExportBundle


def test_parse_dataset_vars_accepts_repeated_key_value_entries():
    assert _parse_dataset_vars(["tracker=full", "subset=night1"]) == {
        "tracker": "full",
        "subset": "night1",
    }


def test_parse_dataset_vars_rejects_missing_equals():
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        _parse_dataset_vars(["tracker"])


@pytest.mark.asyncio
async def test_profile_algorithm_error_passes_dataset_varset_to_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class DummySandbox:
        async def evaluate_adapter(
            self,
            bundle: ExportBundle,
            dataset_path: str,
            metric: OptimizationMetric,
            *,
            varset: dict[str, str] | None = None,
            user: str | None = None,
            serial: str | None = None,
        ) -> BenchmarkResult:
            calls["dataset_path"] = dataset_path
            calls["metric"] = metric
            calls["varset"] = varset
            calls["user"] = user
            calls["serial"] = serial
            return BenchmarkResult(global_loss=1.0)

        async def evaluate(
            self,
            bundle: ExportBundle,
            dataset_path: str,
            metric: OptimizationMetric,
        ) -> BenchmarkResult:
            raise AssertionError("YAML datasets should use evaluate_adapter")

    monkeypatch.setattr(
        "ageom.principal.profiler.ExecutionSandbox",
        lambda: DummySandbox(),
    )

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
    bundle = ExportBundle(
        target="python-pkg",
        output_dir=tmp_path,
        source_path=tmp_path / "verified.py",
        compiled_artifact=tmp_path / "verified.py",
    )

    gradients = await profile_algorithm_error(
        cdg=cdg,
        bundle=bundle,
        dataset_path=str(tmp_path / "adapter.yml"),
        metric=OptimizationMetric.PRECISION,
        dataset_varset={"tracker": "full"},
    )

    assert gradients == []
    assert calls == {
        "dataset_path": str(tmp_path / "adapter.yml"),
        "metric": OptimizationMetric.PRECISION,
        "varset": {"tracker": "full"},
        "user": None,
        "serial": None,
    }
