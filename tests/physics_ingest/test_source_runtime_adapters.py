from __future__ import annotations

import json

from sciona.physics_ingest.sources import (
    SourceRetrievalHTTPAdapter,
    SourceRetrievalSnapshotSinkAdapter,
    build_physics_source_retrieval_run_plan,
    build_source_retrieval_runtime_adapter_report,
    build_source_retrieval_runtime_adapters,
    execute_source_retrieval_plan,
)
from sciona.physics_ingest.sources._manifest import stable_payload_sha256


class FakeSession:
    def __init__(self, response: object = "ok") -> None:
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


class FakeWriteSink:
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


class FakeStoreSink:
    def __init__(self) -> None:
        self.stores: list[dict[str, object]] = []

    def store(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: object,
        metadata: object,
    ) -> dict[str, object]:
        self.stores.append(
            {
                "snapshot_key": snapshot_key,
                "replay_key": replay_key,
                "payload": payload,
                "metadata": metadata,
            }
        )
        return {"stored": True, "path": ("snapshots", snapshot_key)}


def test_http_adapter_merges_headers_without_mutating_inputs() -> None:
    session = FakeSession()
    injected_headers = {"User-Agent": "sciona-test", "Accept": "application/json"}
    auth_headers = {"Authorization": "Bearer injected"}
    request_headers = {
        "Accept": "application/sparql-results+json",
        "X-Request": "step-1",
    }

    adapter = SourceRetrievalHTTPAdapter(
        session,
        headers=injected_headers,
        auth_headers=auth_headers,
    )
    result = adapter.request("GET", "https://example.test", headers=request_headers)

    assert result == "ok"
    assert session.calls == [
        {
            "method": "GET",
            "url": "https://example.test",
            "kwargs": {
                "headers": {
                    "User-Agent": "sciona-test",
                    "Accept": "application/sparql-results+json",
                    "Authorization": "Bearer injected",
                    "X-Request": "step-1",
                }
            },
        }
    ]
    assert injected_headers == {
        "User-Agent": "sciona-test",
        "Accept": "application/json",
    }
    assert request_headers == {
        "Accept": "application/sparql-results+json",
        "X-Request": "step-1",
    }


def test_http_adapter_delegates_to_session_or_callable() -> None:
    session = FakeSession(response={"session": True})
    session_adapter = SourceRetrievalHTTPAdapter(session, auth=("user", "pass"))

    assert session_adapter.request("POST", "https://example.test", json={"a": 1}) == {
        "session": True
    }
    assert session.calls[0]["kwargs"] == {"json": {"a": 1}, "auth": ("user", "pass")}

    calls: list[dict[str, object]] = []

    def request_callable(method: str, url: str, **kwargs: object) -> dict[str, object]:
        calls.append({"method": method, "url": url, "kwargs": kwargs})
        return {"callable": True}

    callable_adapter = SourceRetrievalHTTPAdapter(request_callable)

    assert callable_adapter.request("GET", "https://callable.test") == {
        "callable": True
    }
    assert calls == [
        {"method": "GET", "url": "https://callable.test", "kwargs": {}}
    ]


def test_snapshot_sink_dry_run_returns_receipt_without_delegating() -> None:
    sink = FakeWriteSink()
    adapter = SourceRetrievalSnapshotSinkAdapter(sink, dry_run=True)
    payload = {"records": [{"id": "A"}]}
    metadata = {"source": "unit-test"}

    receipt = adapter.write(
        snapshot_key="physics-ingest/test",
        replay_key="replay:test",
        payload=payload,
        metadata=metadata,
    )

    assert sink.writes == []
    assert receipt["status"] == "dry_run"
    assert receipt["delegated_write_performed"] is False
    assert receipt["sink_method"] == "none"
    assert receipt["snapshot_key"] == "physics-ingest/test"
    assert receipt["replay_key"] == "replay:test"
    assert receipt["payload_sha256"] == stable_payload_sha256(payload)
    assert receipt["metadata_sha256"] == stable_payload_sha256(metadata)


def test_snapshot_sink_delegates_and_normalizes_receipt_fields() -> None:
    sink = FakeStoreSink()
    adapter = SourceRetrievalSnapshotSinkAdapter(sink)
    payload = {"records": [{"id": "B"}]}
    metadata = {"response_status": 200}

    receipt = adapter.write(
        snapshot_key="physics-ingest/delegated",
        replay_key="replay:delegated",
        payload=payload,
        metadata=metadata,
    )

    assert sink.stores == [
        {
            "snapshot_key": "physics-ingest/delegated",
            "replay_key": "replay:delegated",
            "payload": payload,
            "metadata": metadata,
        }
    ]
    assert receipt["receipt_version"] == "physics-source-snapshot-receipt.v1"
    assert receipt["status"] == "written"
    assert receipt["delegated_write_performed"] is True
    assert receipt["sink_method"] == "store"
    assert receipt["payload_sha256"] == stable_payload_sha256(payload)
    assert receipt["metadata_sha256"] == stable_payload_sha256(metadata)
    assert receipt["snapshot_manifest"] == {
        "manifest_version": "physics-source-snapshot-manifest.v1",
        "snapshot_key": "physics-ingest/delegated",
        "replay_key": "replay:delegated",
        "payload_sha256": stable_payload_sha256(payload),
        "metadata_sha256": stable_payload_sha256(metadata),
    }
    assert receipt["sink_receipt"] == {
        "stored": True,
        "path": ["snapshots", "physics-ingest/delegated"],
    }
    assert json.loads(json.dumps(receipt, sort_keys=True)) == receipt


def test_runtime_adapter_report_is_json_safe_and_preflight_only() -> None:
    report = build_source_retrieval_runtime_adapter_report(
        http_client=lambda method, url, **kwargs: {"ok": True},
        snapshot_sink=None,
        headers={"User-Agent": "sciona"},
        auth=object(),
        dry_run=True,
        preflight=True,
    ).to_dict()

    assert json.loads(json.dumps(report, sort_keys=True)) == report
    assert report["report_version"] == "physics-source-runtime-adapters.v1"
    assert report["capabilities"]["http"]["target_supplied"] is True
    assert report["capabilities"]["http"]["supports_callable"] is True
    assert report["capabilities"]["http"]["injects_headers"] is True
    assert report["capabilities"]["http"]["injects_auth"] is True
    assert report["capabilities"]["snapshot_sink"]["target_supplied"] is False
    assert report["preflight_metadata"]["network_calls_during_preflight"] is False
    assert report["preflight_metadata"]["snapshot_writes_during_preflight"] is False


def test_runtime_adapter_bundle_integrates_with_source_executor() -> None:
    payload = {"head": {"vars": ["item"]}, "results": {"bindings": []}}
    session = FakeSession(FakeResponse(payload))
    sink = FakeWriteSink()
    bundle = build_source_retrieval_runtime_adapters(
        http_client=session,
        snapshot_sink=sink,
        headers={"User-Agent": "sciona-runtime-test"},
        auth_headers={"Authorization": "Bearer token"},
    )
    plan = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        dry_run=False,
        limit=3,
    )

    report = execute_source_retrieval_plan(plan, **bundle.execute_kwargs())
    result = report.results[0]

    assert result.status == "executed"
    assert result.response["status_code"] == 200
    assert result.storage["status"] == "written"
    receipt = result.storage["receipt"]
    assert receipt["status"] == "written"
    assert receipt["snapshot_key"] == plan.steps[0].snapshot_key
    assert receipt["replay_key"] == plan.steps[0].replay_key
    assert receipt["payload_sha256"] == stable_payload_sha256(payload)
    assert sink.writes[0]["snapshot_key"] == plan.steps[0].snapshot_key

    request_kwargs = session.calls[0]["kwargs"]
    assert request_kwargs["headers"] == {
        "User-Agent": "sciona-runtime-test",
        "Authorization": "Bearer token",
        "Accept": "application/sparql-results+json",
    }
    assert request_kwargs["params"] == {"LIMIT": 3}
    assert json.loads(json.dumps(bundle.report.to_dict(), sort_keys=True)) == (
        bundle.report.to_dict()
    )
