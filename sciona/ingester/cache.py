"""Content-addressed cache for ingester outputs."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from sciona.architect.handoff import CDGExport
from sciona.ingester.models import IngestionBundle
from sciona.types import MatchResult

_CACHE_VERSION = "ingest-cache-v3"
_CACHE_SCHEMA = "sciona.ingester.cache-envelope"
_CACHE_SCHEMA_VERSION = 1
_CACHE_RUNTIME_MODE = "canonical-first"
_CACHE_PAYLOAD_KIND = "ingestion_bundle"


def compute_ingest_cache_key(
    *,
    source_path: str,
    class_name: str,
    max_depth: int,
    line_threshold: int,
) -> str:
    """Compute a deterministic cache key for an ingest request."""
    text = Path(source_path).read_text(errors="replace")
    payload = {
        "version": _CACHE_VERSION,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "class_name": class_name,
        "max_depth": int(max_depth),
        "line_threshold": int(line_threshold),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_ingest_cache(cache_dir: Path, key: str) -> IngestionBundle | None:
    """Load an ingestion bundle from cache, or return None on miss/error."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        payload = _extract_bundle_payload(raw)
        if payload is None:
            return None
        return _bundle_from_payload(payload)
    except Exception:
        return None


def save_ingest_cache(cache_dir: Path, key: str, bundle: IngestionBundle) -> Path:
    """Persist an ingestion bundle cache entry and return the written path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    payload = _bundle_to_envelope(bundle, key=key)
    _write_json_atomic(path, payload)
    return path


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _bundle_to_envelope(bundle: IngestionBundle, *, key: str) -> dict[str, Any]:
    payload = _bundle_to_payload(bundle)
    cdg = payload.get("cdg", {}) if isinstance(payload.get("cdg"), dict) else {}
    nodes = cdg.get("nodes")
    edges = cdg.get("edges")
    sub_graphs = payload.get("sub_graphs")
    matches = payload.get("match_results")
    return {
        "schema": _CACHE_SCHEMA,
        "schema_version": _CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "cache_key_version": _CACHE_VERSION,
        "runtime_mode": _CACHE_RUNTIME_MODE,
        "payload_kind": _CACHE_PAYLOAD_KIND,
        "payload_summary": {
            "cdg_node_count": len(nodes) if isinstance(nodes, list) else 0,
            "cdg_edge_count": len(edges) if isinstance(edges, list) else 0,
            "sub_graph_count": len(sub_graphs) if isinstance(sub_graphs, dict) else 0,
            "match_result_count": len(matches) if isinstance(matches, list) else 0,
        },
        "payload": payload,
    }


def _extract_bundle_payload(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    if "schema" in raw or "schema_version" in raw or "payload" in raw:
        schema = str(raw.get("schema") or "")
        if schema != _CACHE_SCHEMA:
            return None
        try:
            schema_version = int(raw.get("schema_version"))
        except (TypeError, ValueError):
            return None
        if schema_version != _CACHE_SCHEMA_VERSION:
            return None
        payload_kind = str(raw.get("payload_kind") or _CACHE_PAYLOAD_KIND)
        if payload_kind != _CACHE_PAYLOAD_KIND:
            return None
        payload = raw.get("payload")
        return payload if isinstance(payload, dict) else None

    # Legacy shape: the bundle payload itself is the top-level dict.
    return raw


def _bundle_to_payload(bundle: IngestionBundle) -> dict[str, Any]:
    return {
        "cdg": bundle.cdg.model_dump(mode="json"),
        "sub_graphs": {k: v.model_dump(mode="json") for k, v in bundle.sub_graphs.items()},
        "generated_atoms": bundle.generated_atoms,
        "generated_state_models": bundle.generated_state_models,
        "generated_witnesses": bundle.generated_witnesses,
        "match_results": [mr.to_dict() for mr in bundle.match_results],
        "mypy_passed": bool(bundle.mypy_passed),
        "ghost_sim_passed": bool(bundle.ghost_sim_passed),
        "ghost_sim_report": dict(bundle.ghost_sim_report),
    }


def _bundle_from_payload(payload: dict[str, Any]) -> IngestionBundle:
    if not isinstance(payload, dict):
        payload = {}
    cdg = CDGExport.model_validate(payload.get("cdg", {"nodes": [], "edges": []}))
    sub_graphs_raw = payload.get("sub_graphs", {}) or {}
    sub_graphs: dict[str, CDGExport] = {}
    if isinstance(sub_graphs_raw, dict):
        for name, data in sub_graphs_raw.items():
            try:
                sub_graphs[str(name)] = CDGExport.model_validate(data)
            except Exception:
                continue
    matches_raw = payload.get("match_results", []) or []
    match_results: list[MatchResult] = []
    if isinstance(matches_raw, list):
        for item in matches_raw:
            if not isinstance(item, dict):
                continue
            try:
                match_results.append(MatchResult.from_dict(item))
            except Exception:
                continue
    ghost_sim_report = payload.get("ghost_sim_report", {}) or {}
    return IngestionBundle(
        cdg=cdg,
        sub_graphs=sub_graphs,
        generated_atoms=str(payload.get("generated_atoms", "")),
        generated_state_models=str(payload.get("generated_state_models", "")),
        generated_witnesses=str(payload.get("generated_witnesses", "")),
        match_results=match_results,
        mypy_passed=bool(payload.get("mypy_passed", False)),
        ghost_sim_passed=bool(payload.get("ghost_sim_passed", False)),
        ghost_sim_report=ghost_sim_report if isinstance(ghost_sim_report, dict) else {},
    )
