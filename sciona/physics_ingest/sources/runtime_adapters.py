"""Production-boundary adapters for physics source retrieval execution.

These adapters deliberately avoid importing concrete HTTP or storage clients.
Callers supply already-constructed client/session/sink objects; construction is
side-effect-free and actual IO can only happen when the executor invokes the
wrapper methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from sciona.physics_ingest.sources._manifest import jsonable, stable_payload_sha256


JSONDict = dict[str, Any]

HTTP_ADAPTER_VERSION = "physics-source-runtime-http-adapter.v1"
SNAPSHOT_SINK_ADAPTER_VERSION = "physics-source-runtime-snapshot-sink-adapter.v1"
SNAPSHOT_MANIFEST_VERSION = "physics-source-snapshot-manifest.v1"
SNAPSHOT_RECEIPT_VERSION = "physics-source-snapshot-receipt.v1"
RUNTIME_ADAPTER_REPORT_VERSION = "physics-source-runtime-adapters.v1"


@dataclass(frozen=True)
class SourceRetrievalRuntimeAdapterReport:
    """JSON-safe capability/preflight report for retrieval runtime adapters."""

    report_version: str = RUNTIME_ADAPTER_REPORT_VERSION
    dry_run: bool = False
    preflight: bool = False
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    preflight_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return _json_safe(
            {
                "report_version": self.report_version,
                "dry_run": self.dry_run,
                "preflight": self.preflight,
                "capabilities": self.capabilities,
                "preflight_metadata": self.preflight_metadata,
            }
        )


@dataclass(frozen=True)
class SourceRetrievalRuntimeAdapterBundle:
    """Executor-ready adapter bundle plus a side-effect-free report."""

    http_client: Any | None
    snapshot_sink: Any | None
    report: SourceRetrievalRuntimeAdapterReport

    def execute_kwargs(self) -> JSONDict:
        return {
            "http_client": self.http_client,
            "snapshot_sink": self.snapshot_sink,
        }

    def to_dict(self) -> JSONDict:
        return self.report.to_dict()


class SourceRetrievalHTTPAdapter:
    """Expose ``request(method, url, **kwargs)`` for injected HTTP objects."""

    def __init__(
        self,
        target: Any,
        *,
        headers: Mapping[str, Any] | None = None,
        auth_headers: Mapping[str, Any] | None = None,
        auth: Any | None = None,
    ) -> None:
        self._target = target
        self._headers = _copy_headers(headers)
        self._auth_headers = _copy_headers(auth_headers)
        self._auth = auth

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Delegate an HTTP request after copying and merging injected headers."""

        request_kwargs = dict(kwargs)
        merged_headers = {
            **self._headers,
            **self._auth_headers,
            **_copy_headers(request_kwargs.get("headers")),
        }
        if merged_headers or "headers" in request_kwargs:
            request_kwargs["headers"] = merged_headers
        if self._auth is not None and "auth" not in request_kwargs:
            request_kwargs["auth"] = self._auth

        request_method = getattr(self._target, "request", None)
        if callable(request_method):
            return request_method(method, url, **request_kwargs)
        if callable(self._target):
            return self._target(method, url, **request_kwargs)
        raise TypeError("http target must expose request() or be callable")

    def capability_report(self) -> JSONDict:
        return _http_capabilities(
            self._target,
            headers=self._headers,
            auth_headers=self._auth_headers,
            auth=self._auth,
        )


class SourceRetrievalSnapshotSinkAdapter:
    """Normalize source snapshot writes for an injected sink object."""

    def __init__(
        self,
        target: Any | None = None,
        *,
        dry_run: bool = False,
        preflight: bool = False,
    ) -> None:
        self._target = target
        self._dry_run = dry_run
        self._preflight = preflight

    def write(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: Any,
        metadata: Mapping[str, Any],
    ) -> JSONDict:
        return self._write(
            snapshot_key=snapshot_key,
            replay_key=replay_key,
            payload=payload,
            metadata=metadata,
        )

    def store(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: Any,
        metadata: Mapping[str, Any],
    ) -> JSONDict:
        return self._write(
            snapshot_key=snapshot_key,
            replay_key=replay_key,
            payload=payload,
            metadata=metadata,
        )

    def put(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: Any,
        metadata: Mapping[str, Any],
    ) -> JSONDict:
        return self._write(
            snapshot_key=snapshot_key,
            replay_key=replay_key,
            payload=payload,
            metadata=metadata,
        )

    def capability_report(self) -> JSONDict:
        return _snapshot_sink_capabilities(
            self._target,
            dry_run=self._dry_run,
            preflight=self._preflight,
        )

    def _write(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: Any,
        metadata: Mapping[str, Any],
    ) -> JSONDict:
        safe_metadata = _json_safe(metadata)
        manifest = _snapshot_manifest(
            snapshot_key=snapshot_key,
            replay_key=replay_key,
            payload=payload,
            metadata=safe_metadata,
        )
        if self._dry_run or self._preflight:
            return _snapshot_receipt(
                manifest=manifest,
                status="preflight" if self._preflight else "dry_run",
                delegated_write_performed=False,
                sink_method="none",
                sink_receipt={},
            )

        writer, sink_method = _snapshot_writer(self._target)
        sink_receipt = writer(
            snapshot_key=snapshot_key,
            replay_key=replay_key,
            payload=payload,
            metadata=safe_metadata,
        )
        return _snapshot_receipt(
            manifest=manifest,
            status="written",
            delegated_write_performed=True,
            sink_method=sink_method,
            sink_receipt=_json_safe(sink_receipt) if sink_receipt is not None else {},
        )


def build_source_retrieval_runtime_adapters(
    *,
    http_client: Any | None = None,
    snapshot_sink: Any | None = None,
    headers: Mapping[str, Any] | None = None,
    auth_headers: Mapping[str, Any] | None = None,
    auth: Any | None = None,
    dry_run: bool = False,
    preflight: bool = False,
) -> SourceRetrievalRuntimeAdapterBundle:
    """Build executor-ready adapters and a JSON-safe capability report."""

    adapted_http = (
        SourceRetrievalHTTPAdapter(
            http_client,
            headers=headers,
            auth_headers=auth_headers,
            auth=auth,
        )
        if http_client is not None
        else None
    )
    adapted_sink = (
        SourceRetrievalSnapshotSinkAdapter(
            snapshot_sink,
            dry_run=dry_run,
            preflight=preflight,
        )
        if snapshot_sink is not None or dry_run or preflight
        else None
    )
    report = SourceRetrievalRuntimeAdapterReport(
        dry_run=dry_run,
        preflight=preflight,
        capabilities={
            "http": _http_capabilities(
                http_client,
                headers=_copy_headers(headers),
                auth_headers=_copy_headers(auth_headers),
                auth=auth,
            ),
            "snapshot_sink": _snapshot_sink_capabilities(
                snapshot_sink,
                dry_run=dry_run,
                preflight=preflight,
            ),
        },
        preflight_metadata={
            "side_effect_free": True,
            "network_calls_during_preflight": False,
            "snapshot_writes_during_preflight": False,
            "executor_kwargs": {
                "http_client_supplied": adapted_http is not None,
                "snapshot_sink_supplied": adapted_sink is not None,
            },
        },
    )
    return SourceRetrievalRuntimeAdapterBundle(
        http_client=adapted_http,
        snapshot_sink=adapted_sink,
        report=report,
    )


def build_source_retrieval_runtime_adapter_report(
    **kwargs: Any,
) -> SourceRetrievalRuntimeAdapterReport:
    """Return only the side-effect-free runtime adapter report."""

    return build_source_retrieval_runtime_adapters(**kwargs).report


def _snapshot_manifest(
    *,
    snapshot_key: str,
    replay_key: str,
    payload: Any,
    metadata: Mapping[str, Any],
) -> JSONDict:
    return {
        "manifest_version": SNAPSHOT_MANIFEST_VERSION,
        "snapshot_key": snapshot_key,
        "replay_key": replay_key,
        "payload_sha256": stable_payload_sha256(payload),
        "metadata_sha256": stable_payload_sha256(metadata),
    }


def _snapshot_receipt(
    *,
    manifest: Mapping[str, Any],
    status: str,
    delegated_write_performed: bool,
    sink_method: str,
    sink_receipt: Mapping[str, Any],
) -> JSONDict:
    return _json_safe(
        {
            "receipt_version": SNAPSHOT_RECEIPT_VERSION,
            "status": status,
            "snapshot_key": manifest["snapshot_key"],
            "replay_key": manifest["replay_key"],
            "payload_sha256": manifest["payload_sha256"],
            "metadata_sha256": manifest["metadata_sha256"],
            "snapshot_manifest": manifest,
            "delegated_write_performed": delegated_write_performed,
            "sink_method": sink_method,
            "sink_receipt": sink_receipt,
            "side_effect_free": not delegated_write_performed,
        }
    )


def _snapshot_writer(target: Any | None) -> tuple[Any, str]:
    if target is None:
        raise TypeError(
            "snapshot sink target is required unless dry_run/preflight is enabled"
        )
    for method_name in ("write", "store", "put"):
        writer = getattr(target, method_name, None)
        if callable(writer):
            return writer, method_name
    if callable(target):
        return target, "callable"
    raise TypeError("snapshot sink target must expose write/store/put or be callable")


def _copy_headers(headers: Any) -> dict[str, Any]:
    if headers is None:
        return {}
    if isinstance(headers, Mapping):
        return {str(key): value for key, value in headers.items()}
    return {str(key): value for key, value in dict(headers).items()}


def _http_capabilities(
    target: Any | None,
    *,
    headers: Mapping[str, Any],
    auth_headers: Mapping[str, Any],
    auth: Any | None,
) -> JSONDict:
    return _json_safe(
        {
            "adapter_version": HTTP_ADAPTER_VERSION,
            "imports_requests": False,
            "imports_httpx": False,
            "requires_injected_client": True,
            "target_supplied": target is not None,
            "target_kind": _target_kind(target),
            "supports_request_method": callable(getattr(target, "request", None)),
            "supports_callable": callable(target),
            "injects_headers": bool(headers),
            "injects_auth_headers": bool(auth_headers),
            "injects_auth": auth is not None,
            "mutates_input_headers": False,
            "network_io_at_construction": False,
        }
    )


def _snapshot_sink_capabilities(
    target: Any | None,
    *,
    dry_run: bool,
    preflight: bool,
) -> JSONDict:
    return _json_safe(
        {
            "adapter_version": SNAPSHOT_SINK_ADAPTER_VERSION,
            "imports_boto": False,
            "imports_supabase": False,
            "requires_injected_sink_for_writes": True,
            "target_supplied": target is not None,
            "target_kind": _target_kind(target),
            "supports_write": callable(getattr(target, "write", None)),
            "supports_store": callable(getattr(target, "store", None)),
            "supports_put": callable(getattr(target, "put", None)),
            "supports_callable": callable(target),
            "dry_run": dry_run,
            "preflight": preflight,
            "writes_during_preflight": False,
            "writes_during_dry_run": False,
            "storage_io_at_construction": False,
            "normalizes_receipts": True,
        }
    )


def _target_kind(target: Any | None) -> str:
    if target is None:
        return "none"
    return type(target).__name__


def _json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(
            jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    )


__all__ = [
    "SourceRetrievalHTTPAdapter",
    "SourceRetrievalRuntimeAdapterBundle",
    "SourceRetrievalRuntimeAdapterReport",
    "SourceRetrievalSnapshotSinkAdapter",
    "build_source_retrieval_runtime_adapter_report",
    "build_source_retrieval_runtime_adapters",
]
