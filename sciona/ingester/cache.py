"""Content-addressed cache for ingester outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sciona.architect.handoff import CDGExport
from sciona.ingester.models import IngestionBundle
from sciona.types import MatchResult

_CACHE_VERSION = "ingest-cache-v1"


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
        payload = json.loads(path.read_text())
        return _bundle_from_payload(payload)
    except Exception:
        return None


def save_ingest_cache(cache_dir: Path, key: str, bundle: IngestionBundle) -> Path:
    """Persist an ingestion bundle cache entry and return the written path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    payload = _bundle_to_payload(bundle)
    path.write_text(json.dumps(payload, indent=2))
    return path


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
    cdg = CDGExport.model_validate(payload.get("cdg", {"nodes": [], "edges": []}))
    sub_graphs_raw = payload.get("sub_graphs", {}) or {}
    sub_graphs = {
        str(name): CDGExport.model_validate(data) for name, data in sub_graphs_raw.items()
    }
    matches_raw = payload.get("match_results", []) or []
    match_results = [MatchResult.from_dict(item) for item in matches_raw]
    return IngestionBundle(
        cdg=cdg,
        sub_graphs=sub_graphs,
        generated_atoms=str(payload.get("generated_atoms", "")),
        generated_state_models=str(payload.get("generated_state_models", "")),
        generated_witnesses=str(payload.get("generated_witnesses", "")),
        match_results=match_results,
        mypy_passed=bool(payload.get("mypy_passed", False)),
        ghost_sim_passed=bool(payload.get("ghost_sim_passed", False)),
        ghost_sim_report=payload.get("ghost_sim_report", {}) or {},
    )
