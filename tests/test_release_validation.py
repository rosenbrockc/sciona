from __future__ import annotations

import json

import pytest

from ageom.release_validation import run_release_validation


@pytest.mark.asyncio
async def test_run_release_validation_writes_manifest_and_benchmark_bundle(tmp_path):
    summary = await run_release_validation(tmp_path)

    manifest_path = tmp_path / "release_validation.json"
    assert summary["manifest"] == str(manifest_path)
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    bench = manifest["checks"]["benchmark_validation"]
    assert bench["prompt_results"] > 0
    assert bench["flow_results"] > 0
    assert (tmp_path / "benchmarks" / "summary.json").exists()
