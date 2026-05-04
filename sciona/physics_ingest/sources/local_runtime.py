"""Local runtime helpers for source retrieval activation.

These helpers are intentionally small and dependency-free. They satisfy the
injected HTTP client and snapshot sink contracts used by the source retrieval
executor without constructing global clients inside ingestion code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sciona.physics_ingest.sources._manifest import jsonable, stable_payload_sha256


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class UrllibSourceHTTPResponse:
    """Small response object exposing the methods expected by the executor."""

    status_code: int
    headers: Mapping[str, Any]
    content: bytes
    url: str

    @property
    def text(self) -> str:
        return self.content.decode(_charset(self.headers), errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class UrllibSourceHTTPClient:
    """Conservative stdlib HTTP client for source retrieval activation."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        default_headers: Mapping[str, Any] | None = None,
        user_agent: str = "sciona-physics-ingest/0.1 (+https://github.com/sciona/sciona)",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.default_headers = {
            "User-Agent": user_agent,
            **{str(key): str(value) for key, value in (default_headers or {}).items()},
        }

    def request(self, method: str, url: str, **kwargs: Any) -> UrllibSourceHTTPResponse:
        request_url = _url_with_params(url, kwargs.get("params"))
        headers = {
            **self.default_headers,
            **{str(key): str(value) for key, value in (kwargs.get("headers") or {}).items()},
        }
        body = _request_body(kwargs, headers)
        request = Request(
            request_url,
            data=body,
            headers=headers,
            method=str(method).upper(),
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return UrllibSourceHTTPResponse(
                    status_code=int(getattr(response, "status", 0) or 0),
                    headers=dict(response.headers.items()),
                    content=response.read(),
                    url=str(getattr(response, "url", request_url) or request_url),
                )
        except HTTPError as exc:
            return UrllibSourceHTTPResponse(
                status_code=int(exc.code),
                headers=dict(exc.headers.items()),
                content=exc.read(),
                url=request_url,
            )


class LocalFilesystemSnapshotSink:
    """Write immutable source payload snapshots under a local directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(
        self,
        *,
        snapshot_key: str,
        replay_key: str,
        payload: Any,
        metadata: Mapping[str, Any],
    ) -> JSONDict:
        safe_payload = jsonable(payload)
        safe_metadata = jsonable(metadata)
        payload_sha256 = stable_payload_sha256(safe_payload)
        metadata_sha256 = stable_payload_sha256(safe_metadata)
        snapshot_dir = self.root / _safe_relative_path(snapshot_key) / payload_sha256
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        payload_path = snapshot_dir / _payload_filename(safe_payload)
        metadata_path = snapshot_dir / "metadata.json"
        manifest_path = snapshot_dir / "manifest.json"
        _write_payload(payload_path, safe_payload)
        metadata_path.write_text(
            json.dumps(safe_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest = {
            "manifest_version": "physics-source-local-snapshot.v1",
            "snapshot_key": snapshot_key,
            "replay_key": replay_key,
            "payload_sha256": payload_sha256,
            "metadata_sha256": metadata_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload_path": str(payload_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "payload_size_bytes": payload_path.stat().st_size,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return jsonable(manifest)

    store = write
    put = write


def _url_with_params(url: str, params: Any) -> str:
    if not isinstance(params, Mapping) or not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params, doseq=True)}"


def _request_body(kwargs: Mapping[str, Any], headers: dict[str, str]) -> bytes | None:
    if "json" in kwargs:
        headers.setdefault("Content-Type", "application/json")
        return json.dumps(jsonable(kwargs["json"]), sort_keys=True).encode("utf-8")
    if "data" not in kwargs:
        return None
    data = kwargs["data"]
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, Mapping):
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        return urlencode(data, doseq=True).encode("utf-8")
    return str(data).encode("utf-8")


def _charset(headers: Mapping[str, Any]) -> str:
    content_type = str(
        headers.get("content-type") or headers.get("Content-Type") or ""
    ).lower()
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("charset="):
            return part.split("=", 1)[1].strip() or "utf-8"
    return "utf-8"


def _safe_relative_path(value: str) -> Path:
    parts = [
        _safe_segment(part)
        for part in str(value).replace("\\", "/").split("/")
        if part not in {"", ".", ".."}
    ]
    return Path(*parts) if parts else Path("snapshot")


def _safe_segment(value: str) -> str:
    segment = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    )
    return segment.strip("._") or "segment"


def _payload_filename(payload: Any) -> str:
    return "payload.txt" if isinstance(payload, str) else "payload.json"


def _write_payload(path: Path, payload: Any) -> None:
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
