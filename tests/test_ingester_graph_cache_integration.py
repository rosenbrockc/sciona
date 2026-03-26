"""Integration test for IngesterAgent cache miss/hit behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sciona.ingester.cache import compute_ingest_cache_key
from sciona.ingester.graph import IngesterAgent
from sciona.ingester.models import IngestionBundle
from sciona.ingester.monitor import IngestMonitor, TRACE_FILE


@pytest.mark.asyncio
async def test_ingester_agent_cache_miss_then_hit_preserves_output_and_emits_trace(
    tmp_path,
):
    source = tmp_path / "example.py"
    source.write_text("class Example:\n    pass\n")

    cache_dir = tmp_path / "cache"
    monitor_dir = tmp_path / "monitor"
    monitor = IngestMonitor(monitor_dir, enable_trace=True)
    agent = IngesterAgent(
        llm=AsyncMock(),
        monitor=monitor,
        enable_cache=True,
        cache_dir=str(cache_dir),
    )

    expected_bundle = IngestionBundle.model_validate(
        {
            "cdg": {"nodes": [], "edges": []},
            "generated_atoms": "class ExampleAtom:\n    pass\n",
            "generated_state_models": "class ExampleState:\n    pass\n",
            "generated_witnesses": "def witness_example_atom() -> bool:\n    return True\n",
            "match_results": [],
            "mypy_passed": True,
            "ghost_sim_passed": True,
            "ghost_sim_report": {"ran": False, "passed": True},
        }
    )

    ingest_calls = 0

    async def fake_ingest_state(source_path: str, class_name: str) -> dict[str, object]:
        nonlocal ingest_calls
        assert source_path == str(source)
        assert class_name == "Example"
        ingest_calls += 1
        return {"bundle": expected_bundle.model_copy(deep=True), "error": ""}

    setattr(agent, "ingest_state", fake_ingest_state)

    miss_bundle = await agent.ingest(str(source), "Example")
    assert ingest_calls == 1

    cache_key = compute_ingest_cache_key(
        source_path=str(source),
        class_name="Example",
        max_depth=agent._deps.max_depth,
        line_threshold=agent._deps.line_threshold,
    )
    assert (cache_dir / f"{cache_key}.json").exists()

    hit_bundle = await agent.ingest(str(source), "Example")
    assert ingest_calls == 1

    assert miss_bundle.generated_atoms == hit_bundle.generated_atoms
    assert miss_bundle.generated_state_models == hit_bundle.generated_state_models
    assert miss_bundle.generated_witnesses == hit_bundle.generated_witnesses
    assert miss_bundle.cdg.model_dump(mode="json") == hit_bundle.cdg.model_dump(
        mode="json"
    )
    assert [item.to_dict() for item in miss_bundle.match_results] == [
        item.to_dict() for item in hit_bundle.match_results
    ]

    trace_path = monitor_dir / TRACE_FILE
    assert trace_path.exists()
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]

    lookup_states = [
        str((((event.get("payload") or {}).get("cache") or {}).get("state") or ""))
        for event in events
        if event.get("event_type") == "CACHE_LOOKUP"
    ]
    store_states = [
        str((((event.get("payload") or {}).get("cache") or {}).get("state") or ""))
        for event in events
        if event.get("event_type") == "CACHE_STORE"
    ]
    assert lookup_states == ["miss", "hit"]
    assert store_states == ["miss"]
