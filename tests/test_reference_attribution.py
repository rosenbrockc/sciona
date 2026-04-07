from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.principal.reference_attribution import compute_reference_loss_gradients
from sciona.synthesizer.models import ExportBundle


def _make_cdg() -> CDGExport:
    return CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="node_a",
                name="Stage A",
                description="first stage",
                concept_type=ConceptType.ARITHMETIC,
                status=NodeStatus.ATOMIC,
            ),
            AlgorithmicNode(
                node_id="node_b",
                name="Stage B",
                description="second stage",
                concept_type=ConceptType.ARITHMETIC,
                status=NodeStatus.ATOMIC,
            ),
        ],
        edges=[],
    )


def test_reference_loss_gradients_use_counterfactual_rmse(tmp_path: Path) -> None:
    export_dir = tmp_path / "export_python_pkg"
    package_dir = export_dir / "src" / "demo_pkg"
    package_dir.mkdir(parents=True)
    (export_dir / "runner.py").write_text("# runner placeholder\n")
    (package_dir / "__init__.py").write_text("")
    (package_dir / "atoms.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def _sciona_probe(node_id, fn):",
                "    return fn()",
                "",
                "def _stage_a_inner(signal):",
                "    return signal",
                "",
                "def stage_a(signal):",
                "    return _sciona_probe('node_a', lambda: _stage_a_inner(signal))",
                "",
                "def _stage_b_inner(x):",
                "    return x * 10.0",
                "",
                "def stage_b(x):",
                "    return _sciona_probe('node_b', lambda: _stage_b_inner(x))",
                "",
                "def toy_pipeline(signal):",
                "    return stage_b(stage_a(signal))",
            ]
        )
        + "\n"
    )
    (package_dir / "pipeline.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import numpy as np",
                "from . import atoms",
                "",
                "DEFAULT_ENTRYPOINT = 'toy_pipeline'",
                "",
                "def load_dataset(dataset_root, dataset_vars=None, user=None, serial=None, entrypoint=None, eval_spec=None):",
                "    return {'signal': np.array([1.0, 2.0, 3.0]), 'reference': np.array([10.0, 20.0, 30.0])}",
                "",
                "def _flatten_inputs(kwargs):",
                "    return dict(kwargs)",
                "",
                "def run_pipeline(**kwargs):",
                "    return atoms.toy_pipeline(kwargs['signal'])",
            ]
        )
        + "\n"
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "sciona.yml").write_text("name: demo\n")

    bundle = ExportBundle(
        target="python-pkg",
        output_dir=export_dir,
        source_path=export_dir / "runner.py",
        compiled_artifact=export_dir / "runner.py",
        executable_artifact=export_dir / "runner.py",
    )

    gradients = asyncio.run(
        compute_reference_loss_gradients(
            _make_cdg(),
            bundle,
            str(dataset_root / "sciona.yml"),
            {
                "loss": "rmse",
                "reference": {"value_source": "reference"},
            },
        )
    )

    assert gradients
    assert {gradient.node_id for gradient in gradients} == {"node_a", "node_b"}
    assert all("rmse" in gradient.bottleneck_reason for gradient in gradients)
    runtime_artifacts = json.loads(
        (export_dir / "profile_runtime_artifacts.json").read_text()
    )
    assert "runtime_context" in runtime_artifacts
    assert "canonical_runtime_context" in runtime_artifacts
    assert "telemetry_summary" in runtime_artifacts
    assert (
        runtime_artifacts["canonical_runtime_context"]["canonical_inputs"]["signal"][
            "raw_key"
        ]
        == "signal"
    )
