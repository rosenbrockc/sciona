from __future__ import annotations

import json

from sciona.physics_ingest.sources import (
    LocalFilesystemSnapshotSink,
    UrllibSourceHTTPClient,
    build_physics_source_retrieval_run_plan,
    build_source_retrieval_runtime_execution_report_dict,
)
from sciona.physics_ingest.sources._manifest import stable_payload_sha256
import sciona.physics_ingest.sources.local_runtime as local_runtime


class FakeHTTPResponse:
    status = 200
    url = "https://example.test/data?limit=2"
    headers = {"Content-Type": "application/json; charset=utf-8"}

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"ok": true, "records": [1, 2]}'


def test_urllib_source_http_client_uses_stdlib_request_boundary(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        calls.append(
            {
                "full_url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "data": request.data,
                "timeout": timeout,
            }
        )
        return FakeHTTPResponse()

    monkeypatch.setattr(local_runtime, "urlopen", fake_urlopen)
    client = UrllibSourceHTTPClient(
        timeout_seconds=12.5,
        user_agent="sciona-test-agent",
    )

    response = client.request(
        "GET",
        "https://example.test/data",
        params={"limit": 2},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "records": [1, 2]}
    assert calls == [
        {
            "full_url": "https://example.test/data?limit=2",
            "method": "GET",
            "headers": {
                "User-agent": "sciona-test-agent",
                "Accept": "application/json",
            },
            "data": None,
            "timeout": 12.5,
        }
    ]


def test_local_filesystem_snapshot_sink_writes_manifest_payload_and_metadata(
    tmp_path,
) -> None:
    sink = LocalFilesystemSnapshotSink(tmp_path)
    payload = {"records": [{"id": "codata-speed-of-light"}]}
    metadata = {"source": "nist_codata", "status": 200}

    receipt = sink.write(
        snapshot_key="physics-ingest/nist-codata",
        replay_key="physics-source-retrieval:test",
        payload=payload,
        metadata=metadata,
    )

    assert receipt["manifest_version"] == "physics-source-local-snapshot.v1"
    assert receipt["payload_sha256"] == stable_payload_sha256(payload)
    payload_path = tmp_path / "physics-ingest" / "nist-codata" / receipt[
        "payload_sha256"
    ] / "payload.json"
    metadata_path = payload_path.with_name("metadata.json")
    manifest_path = payload_path.with_name("manifest.json")
    assert json.loads(payload_path.read_text(encoding="utf-8")) == payload
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == receipt


def test_local_runtime_helpers_clear_nist_activation_preflight_blockers(tmp_path) -> None:
    plan = build_physics_source_retrieval_run_plan(
        job_id="nist_codata_constants.backfill",
        dry_run=False,
        limit=1,
    )
    report = build_source_retrieval_runtime_execution_report_dict(
        plan,
        http_client=UrllibSourceHTTPClient(),
        snapshot_sink=LocalFilesystemSnapshotSink(tmp_path),
        execute=False,
        preflight=True,
    )

    assert report["side_effect_free"] is True
    assert report["summary"]["blocking_diagnostic_count"] == 0
    assert report["summary"]["requires_http_client_count"] == 1
    assert report["summary"]["requires_snapshot_sink_count"] == 1
    assert report["executor_kwargs"]["http_client"]["supplied"] is True
    assert report["executor_kwargs"]["snapshot_sink"]["supplied"] is True
