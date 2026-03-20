"""Tests for Phase 2 — parameter flow through Principal → Assembler → Evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciona.principal.graph import PrincipalState
from sciona.principal.models import OptimizationMetric
from sciona.synthesizer.models import ExportBundle


class TestExportBundleParameterAssignments:
    def test_default_empty(self, tmp_path):
        bundle = ExportBundle(
            target="python",
            output_dir=tmp_path,
            source_path=tmp_path / "source.py",
        )
        assert bundle.parameter_assignments == {}

    def test_roundtrip(self, tmp_path):
        assignments = {
            "node_a": {"filter_order": 2, "low_cutoff_hz": 1.0},
            "node_b": {"prominence_scale": 3.0},
        }
        bundle = ExportBundle(
            target="python",
            output_dir=tmp_path,
            source_path=tmp_path / "source.py",
            parameter_assignments=assignments,
        )
        data = bundle.model_dump()
        restored = ExportBundle.model_validate(data)
        assert restored.parameter_assignments == assignments

    def test_json_serialization(self, tmp_path):
        assignments = {"n1": {"x": 1.5, "y": True}}
        bundle = ExportBundle(
            target="python",
            output_dir=tmp_path,
            source_path=tmp_path / "source.py",
            parameter_assignments=assignments,
        )
        text = bundle.model_dump_json()
        restored = ExportBundle.model_validate_json(text)
        assert restored.parameter_assignments["n1"]["x"] == 1.5


class TestPrincipalStateParams:
    def test_default_empty(self):
        state = PrincipalState()
        assert state.node_params == {}
        assert state.best_node_params == {}

    def test_node_params_stored(self):
        state = PrincipalState()
        state.node_params = {"node_a": {"filter_order": 2}}
        assert state.node_params["node_a"]["filter_order"] == 2


class TestEvaluatorParamsFlag:
    @pytest.mark.asyncio
    async def test_passes_params_flag(self, tmp_path):
        """Verify that --params is added to subprocess command when assignments present."""
        from unittest.mock import AsyncMock, patch

        from sciona.principal.evaluator import ExecutionSandbox
        from sciona.principal.models import BenchmarkResult

        # Create a minimal artifact
        artifact = tmp_path / "artifact.py"
        artifact.write_text("print('ok')")

        bundle = ExportBundle(
            target="python",
            output_dir=tmp_path,
            source_path=artifact,
            executable_artifact=artifact,
            parameter_assignments={"node_a": {"x": 1.0}},
        )

        sandbox = ExecutionSandbox(timeout_s=5.0)

        captured_cmd = []

        async def mock_exec(*args, **kwargs):
            captured_cmd.extend(args)
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b'{"loss": 0.5}', b""))
            proc.returncode = 0
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        # Write a trace file so parsing doesn't fail
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text('{"node_id": "node_a", "execution_time_ms": 10, "peak_memory_bytes": 100}\n')

        async def await_coro(coro, **kw):
            return await coro

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            with patch("asyncio.wait_for", side_effect=await_coro):
                result = await sandbox.evaluate(
                    bundle, str(tmp_path / "data.json"), OptimizationMetric.PRECISION
                )

        # Check that --params was in the command
        cmd_str = " ".join(str(c) for c in captured_cmd)
        assert "--params" in cmd_str
        # Check that params.json was written
        params_path = tmp_path / "params.json"
        assert params_path.exists()
        params_data = json.loads(params_path.read_text())
        assert params_data["node_a"]["x"] == 1.0

    @pytest.mark.asyncio
    async def test_no_params_flag_when_empty(self, tmp_path):
        """No --params flag when parameter_assignments is empty."""
        from unittest.mock import AsyncMock, patch

        from sciona.principal.evaluator import ExecutionSandbox

        artifact = tmp_path / "artifact.py"
        artifact.write_text("print('ok')")

        bundle = ExportBundle(
            target="python",
            output_dir=tmp_path,
            source_path=artifact,
            executable_artifact=artifact,
            # Empty assignments
        )

        sandbox = ExecutionSandbox(timeout_s=5.0)

        captured_cmd = []

        async def mock_exec(*args, **kwargs):
            captured_cmd.extend(args)
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b'{"loss": 0.5}', b""))
            proc.returncode = 0
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text('{"node_id": "n", "execution_time_ms": 10, "peak_memory_bytes": 100}\n')

        async def await_coro(coro, **kw):
            return await coro

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            with patch("asyncio.wait_for", side_effect=await_coro):
                await sandbox.evaluate(
                    bundle, str(tmp_path / "data.json"), OptimizationMetric.PRECISION
                )

        cmd_str = " ".join(str(c) for c in captured_cmd)
        assert "--params" not in cmd_str
