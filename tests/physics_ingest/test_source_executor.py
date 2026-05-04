from __future__ import annotations

import json

from sciona.physics_ingest.sources import (
    execute_source_retrieval_plan,
    execute_source_retrieval_plan_dict,
)
from sciona.physics_ingest.sources._manifest import stable_payload_sha256
from sciona.physics_ingest.sources.retrieval_plan import (
    build_physics_source_retrieval_run_plan,
)


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> object:
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        return self.response


class FakeResponse:
    status_code = 202
    headers = {"content-type": "application/json"}

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeSink:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: object,
        metadata: object,
    ) -> dict[str, object]:
        self.writes.append(
            {
                "snapshot_key": snapshot_key,
                "replay_key": replay_key,
                "payload": payload,
                "metadata": metadata,
            }
        )
        return {"stored": True, "snapshot_key": snapshot_key}


def test_source_executor_dry_run_is_json_safe_and_side_effect_free() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="nist_codata_constants.backfill",
    )
    client = FakeClient(FakeResponse({"should_not": "be called"}))
    sink = FakeSink()

    report_dict = execute_source_retrieval_plan_dict(
        plan.to_dict(),
        http_client=client,
        snapshot_sink=sink,
    )
    encoded = json.dumps(report_dict, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert report_dict["summary"] == {
        "total_steps": 1,
        "dry_run": 1,
        "manual_offline": 0,
        "blocked_missing_client": 0,
        "executed": 0,
        "with_payload": 0,
        "persisted": 0,
        "by_status": {"dry_run": 1},
        "by_source_system": {"nist_codata": 1},
    }
    result = report_dict["results"][0]
    assert result["status"] == "dry_run"
    assert result["request"]["status"] == "planned"
    assert result["snapshot_key"] == plan.steps[0].snapshot_key
    assert result["replay_key"] == plan.steps[0].replay_key
    assert result["adapter_target"] == plan.steps[0].request_envelope["adapter_target"]
    assert result["execution"]["mode"] == "dry_run"
    assert client.calls == []
    assert sink.writes == []


def test_source_executor_marks_manual_steps_offline_without_io() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="foundational_manual_seed.backfill",
        dry_run=False,
    )
    client = FakeClient(FakeResponse({"should_not": "be called"}))
    sink = FakeSink()

    report = execute_source_retrieval_plan(
        plan,
        http_client=client,
        snapshot_sink=sink,
    )

    assert report.summary["manual_offline"] == 1
    result = report.results[0]
    assert result.status == "manual_offline"
    assert result.execution["mode"] == "manual/offline"
    assert result.execution["io_performed"] is False
    assert result.request["status"] == "offline"
    assert result.storage["status"] == "not_required"
    assert client.calls == []
    assert sink.writes == []


def test_source_executor_blocks_network_steps_without_client() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="nist_codata_constants.backfill",
        dry_run=False,
    )

    report = execute_source_retrieval_plan(plan)

    assert report.summary["blocked_missing_client"] == 1
    result = report.results[0]
    assert result.status == "blocked_missing_client"
    assert result.status_reason == "network step requires an injected HTTP client"
    assert result.execution["network_io_performed"] is False
    assert result.request["status"] == "blocked"
    assert result.storage["status"] == "not_attempted"
    assert result.payload["available"] is False


def test_source_executor_uses_fake_client_and_sink_for_network_execution() -> None:
    payload = {"records": [{"id": "Q1", "formula": "E = mc^2"}], "next": None}
    response = FakeResponse(payload)
    client = FakeClient(response)
    sink = FakeSink()
    plan = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        dry_run=False,
        limit=7,
    )

    report = execute_source_retrieval_plan(
        plan,
        http_client=client,
        snapshot_sink=sink,
    )

    assert report.summary["executed"] == 1
    assert report.summary["with_payload"] == 1
    assert report.summary["persisted"] == 1
    result = report.results[0]
    expected_hash = stable_payload_sha256(payload)
    assert result.status == "executed"
    assert result.response["status_code"] == 202
    assert result.payload == {
        "available": True,
        "payload_sha256": expected_hash,
        "status_metadata": {
            "payload_sha256": expected_hash,
            "response_status": 202,
            "content_type": "application/json",
        },
    }
    assert result.storage["status"] == "written"
    assert result.storage["receipt"] == {
        "stored": True,
        "snapshot_key": plan.steps[0].snapshot_key,
    }

    assert client.calls == [
        {
            "method": "POST",
            "url": "https://query.wikidata.org/sparql",
            "kwargs": {
                "headers": {"Accept": "application/sparql-results+json"},
                "params": {"LIMIT": 7},
                "data": {
                    "format": "json",
                    "query": (
                        "SELECT ?item ?itemLabel ?itemDescription ?formulaProperty "
                        "?formula ?alias ?use ?useLabel ?useDescription WHERE {\n"
                        "  VALUES ?formulaProperty { wdt:P2534 }\n"
                        "  ?item ?formulaProperty ?formula .\n"
                        '  OPTIONAL { ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }\n'
                        "  OPTIONAL { ?item wdt:P366 ?use . }\n"
                        "  SERVICE wikibase:label {\n"
                        '    bd:serviceParam wikibase:language "en,mul,en".\n'
                        "  }\n"
                        "} LIMIT 7"
                    ),
                },
            },
        }
    ]
    assert sink.writes[0]["snapshot_key"] == plan.steps[0].snapshot_key
    assert sink.writes[0]["replay_key"] == plan.steps[0].replay_key
    assert sink.writes[0]["payload"] == payload
    assert sink.writes[0]["metadata"]["payload_sha256"] == expected_hash
