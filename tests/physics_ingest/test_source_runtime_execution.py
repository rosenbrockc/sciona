from __future__ import annotations

import json

from sciona.physics_ingest.sources import (
    build_physics_source_retrieval_run_plan,
    build_source_retrieval_runtime_execution_report,
    build_source_retrieval_runtime_execution_report_dict,
)
from sciona.physics_ingest.sources._manifest import stable_payload_sha256


class FakeSession:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> object:
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        return self.response


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}

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


def test_runtime_execution_preflight_is_side_effect_free() -> None:
    payload = {"should_not": "be requested"}
    session = FakeSession(FakeResponse(payload))
    sink = FakeSink()
    plan = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        dry_run=False,
        limit=2,
    )

    report = build_source_retrieval_runtime_execution_report(
        plan,
        http_client=session,
        snapshot_sink=sink,
        headers={"User-Agent": "sciona-runtime-test"},
        auth_headers={"Authorization": "Bearer token"},
        execute=True,
        preflight=True,
    )
    report_dict = report.to_dict()

    assert json.loads(json.dumps(report_dict, sort_keys=True)) == report_dict
    assert report.execution_requested is True
    assert report.execution_performed is False
    assert report.execution_skipped_reason == "preflight_only"
    assert report.side_effect_free is True
    assert report_dict["execution_report"] is None
    assert report_dict["summary"]["total_steps"] == 1
    assert report_dict["summary"]["requires_http_client_count"] == 1
    assert report_dict["summary"]["requires_snapshot_sink_count"] == 1
    assert report_dict["steps"][0]["executor_kwargs_required"] == [
        "http_client",
        "snapshot_sink",
    ]
    assert report_dict["executor_kwargs"] == {
        "http_client": {
            "kwarg": "http_client",
            "supplied": True,
            "adapter": "SourceRetrievalHTTPAdapter",
        },
        "snapshot_sink": {
            "kwarg": "snapshot_sink",
            "supplied": True,
            "adapter": "SourceRetrievalSnapshotSinkAdapter",
        },
    }
    assert session.calls == []
    assert sink.writes == []


def test_runtime_execution_uses_injected_fake_session_and_sink() -> None:
    payload = {"head": {"vars": ["item"]}, "results": {"bindings": []}}
    session = FakeSession(FakeResponse(payload))
    sink = FakeSink()
    plan = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        dry_run=False,
        limit=5,
    )

    report = build_source_retrieval_runtime_execution_report(
        plan.to_dict(),
        http_client=session,
        snapshot_sink=sink,
        headers={"User-Agent": "sciona-runtime-test"},
        auth_headers={"Authorization": "Bearer token"},
        execute=True,
        preflight=False,
    )
    report_dict = report.to_dict()

    assert report.execution_requested is True
    assert report.execution_performed is True
    assert report.execution_skipped_reason == ""
    assert report.side_effect_free is False
    assert report_dict["summary"]["execution_result_count"] == 1
    assert report_dict["execution_report"]["summary"]["executed"] == 1
    result = report_dict["execution_report"]["results"][0]
    assert result["status"] == "executed"
    assert result["payload"]["payload_sha256"] == stable_payload_sha256(payload)
    assert result["storage"]["status"] == "written"

    assert session.calls == [
        {
            "method": "POST",
            "url": "https://query.wikidata.org/sparql",
            "kwargs": {
                "headers": {
                    "User-Agent": "sciona-runtime-test",
                    "Authorization": "Bearer token",
                    "Accept": "application/sparql-results+json",
                },
                "params": {"LIMIT": 5},
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
                        "} LIMIT 5"
                    ),
                },
            },
        }
    ]
    assert sink.writes[0]["snapshot_key"] == plan.steps[0].snapshot_key
    assert sink.writes[0]["replay_key"] == plan.steps[0].replay_key
    assert sink.writes[0]["payload"] == payload


def test_runtime_execution_reports_missing_dependencies_without_execution() -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="nist_codata_constants.backfill",
        dry_run=False,
    )

    report_dict = build_source_retrieval_runtime_execution_report_dict(
        plan,
        execute=True,
        preflight=False,
    )

    assert json.loads(json.dumps(report_dict, sort_keys=True)) == report_dict
    assert report_dict["execution_requested"] is True
    assert report_dict["execution_performed"] is False
    assert report_dict["execution_skipped_reason"] == "blocking_diagnostics"
    assert report_dict["execution_report"] is None
    assert report_dict["side_effect_free"] is True
    assert report_dict["summary"]["blocking_diagnostic_count"] == 2
    assert report_dict["summary"]["by_diagnostic_code"] == {
        "missing_http_client": 1,
        "missing_snapshot_sink": 1,
    }
    assert [diagnostic["code"] for diagnostic in report_dict["diagnostics"]] == [
        "missing_http_client",
        "missing_snapshot_sink",
    ]
    assert report_dict["steps"][0]["diagnostics"] == report_dict["diagnostics"]
