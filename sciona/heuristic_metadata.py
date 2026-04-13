"""Auditable atom-side metadata for heuristic-producing outputs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sciona.atom_identity import (
    ATOM_METADATA_GLOB_CANDIDATES,
    atom_provider_id_for_fqdn,
    candidate_atom_provider_roots,
    logical_atom_id_from_fqdn,
)
from sciona.heuristics import (
    CanonicalHeuristic,
    HeuristicProducerKind,
    canonical_heuristic_from_metric,
)


class AtomHeuristicReference(BaseModel):
    """Human-reviewable reference supporting a heuristic-producing output."""

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class HeuristicOutputContract(BaseModel):
    """One declared heuristic-producing output on an audited atom."""

    output_name: str
    output_path: str = ""
    role: Literal["advisory", "gating", "structural"] = "advisory"
    semantic_kind: str = ""
    heuristic: CanonicalHeuristic
    provenance_notes: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_producer_kind(self) -> "HeuristicOutputContract":
        if self.heuristic.producer_kind != HeuristicProducerKind.ATOM_OUTPUT:
            self.heuristic = self.heuristic.model_copy(
                update={"producer_kind": HeuristicProducerKind.ATOM_OUTPUT}
            )
        return self


class AtomHeuristicMetadata(BaseModel):
    """Audited metadata declaring which atom outputs are usable heuristics."""

    atom_fqdn: str
    summary: str
    dejargonized_summary: str
    heuristic_outputs: list[HeuristicOutputContract] = Field(default_factory=list)
    references: list[AtomHeuristicReference] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_uniqueness_and_docs(self) -> "AtomHeuristicMetadata":
        if not self.dejargonized_summary.strip():
            raise ValueError("Atom heuristic metadata must include a dejargonized summary")
        if not self.references:
            raise ValueError("Atom heuristic metadata must include at least one reference")
        output_names = [item.output_name for item in self.heuristic_outputs]
        duplicate_output_names = sorted(
            {name for name in output_names if output_names.count(name) > 1}
        )
        if duplicate_output_names:
            raise ValueError(
                "Duplicate heuristic output names are not allowed: "
                + ", ".join(duplicate_output_names)
            )
        heuristic_ids = [item.heuristic.heuristic_id for item in self.heuristic_outputs]
        duplicate_heuristics = sorted(
            {name for name in heuristic_ids if heuristic_ids.count(name) > 1}
        )
        if duplicate_heuristics:
            raise ValueError(
                "Duplicate canonical heuristic identifiers are not allowed: "
                + ", ".join(duplicate_heuristics)
            )
        return self


def atom_heuristic_metadata_summary(
    metadata: AtomHeuristicMetadata | dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a compact runtime-safe summary of atom heuristic metadata."""
    if metadata is None:
        return {}
    if isinstance(metadata, AtomHeuristicMetadata):
        return {
            "atom_fqdn": metadata.atom_fqdn,
            "logical_atom_id": logical_atom_id_from_fqdn(metadata.atom_fqdn),
            "provider_id": atom_provider_id_for_fqdn(metadata.atom_fqdn),
            "heuristic_output_count": len(metadata.heuristic_outputs),
            "heuristic_ids": [
                item.heuristic.heuristic_id for item in metadata.heuristic_outputs
            ],
            "roles": [item.role for item in metadata.heuristic_outputs],
            "maintainer_count": len(metadata.maintainers),
        }
    if isinstance(metadata, dict):
        outputs = metadata.get("heuristic_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        heuristic_ids: list[str] = []
        roles: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            heuristic = item.get("heuristic", {})
            if isinstance(heuristic, dict) and heuristic.get("heuristic_id"):
                heuristic_ids.append(str(heuristic["heuristic_id"]))
            if item.get("role"):
                roles.append(str(item["role"]))
        return {
            "atom_fqdn": str(metadata.get("atom_fqdn", "") or ""),
            "logical_atom_id": logical_atom_id_from_fqdn(
                str(metadata.get("atom_fqdn", "") or "")
            ),
            "provider_id": atom_provider_id_for_fqdn(
                str(metadata.get("atom_fqdn", "") or "")
            ),
            "heuristic_output_count": len(outputs),
            "heuristic_ids": heuristic_ids,
            "roles": roles,
            "maintainer_count": len(metadata.get("maintainers", []) or []),
        }
    return {}


def atom_heuristic_metadata_from_snapshot(
    atom_fqdn: str,
    snapshot: dict[str, Any],
) -> AtomHeuristicMetadata:
    """Normalize a flat snapshot payload into typed atom heuristic metadata."""
    outputs: list[HeuristicOutputContract] = []
    for raw in snapshot.get("heuristic_outputs", []) or []:
        if not isinstance(raw, dict):
            continue
        heuristic_payload = raw.get("heuristic")
        heuristic: CanonicalHeuristic | None = None
        if isinstance(heuristic_payload, dict):
            heuristic = CanonicalHeuristic.model_validate(heuristic_payload)
        else:
            metric_name = str(raw.get("metric_name", "") or "")
            heuristic = canonical_heuristic_from_metric(
                metric_name,
                source_domain=str(snapshot.get("source_domain", "") or ""),
            )
        if heuristic is None:
            continue
        outputs.append(
            HeuristicOutputContract(
                output_name=str(raw.get("output_name", "") or ""),
                output_path=str(raw.get("output_path", "") or ""),
                role=str(raw.get("role", "advisory") or "advisory"),
                semantic_kind=str(raw.get("semantic_kind", "") or ""),
                heuristic=heuristic,
                provenance_notes=[
                    str(item)
                    for item in raw.get("provenance_notes", []) or []
                    if str(item)
                ],
                uncertainty_notes=[
                    str(item)
                    for item in raw.get("uncertainty_notes", []) or []
                    if str(item)
                ],
            )
        )
    references = [
        AtomHeuristicReference.model_validate(item)
        for item in snapshot.get("references", []) or []
        if isinstance(item, dict)
    ]
    return AtomHeuristicMetadata(
        atom_fqdn=atom_fqdn,
        summary=str(snapshot.get("summary", "") or ""),
        dejargonized_summary=str(snapshot.get("dejargonized_summary", "") or ""),
        heuristic_outputs=outputs,
        references=references,
        uncertainty_notes=[
            str(item) for item in snapshot.get("uncertainty_notes", []) or [] if str(item)
        ],
        maintainers=[
            str(item) for item in snapshot.get("maintainers", []) or [] if str(item)
        ],
    )


def _metadata_snapshots_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _metadata_provider_roots(
    provider_roots: tuple[str | Path, ...] | None = None,
) -> tuple[Path, ...]:
    if provider_roots is None:
        return tuple(candidate_atom_provider_roots())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in provider_roots:
        resolved = Path(root).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


@lru_cache(maxsize=8)
def _load_external_atom_heuristic_metadata_cached(
    provider_roots: tuple[str, ...],
) -> tuple[AtomHeuristicMetadata, ...]:
    records: list[AtomHeuristicMetadata] = []
    seen_fqdns: set[str] = set()
    for root_text in provider_roots:
        root = Path(root_text).expanduser().resolve()
        for pattern in ATOM_METADATA_GLOB_CANDIDATES:
            for path in sorted(root.glob(pattern)):
                raw = json.loads(path.read_text())
                for snapshot in _metadata_snapshots_from_payload(raw):
                    record = AtomHeuristicMetadata.model_validate(snapshot)
                    if record.atom_fqdn in seen_fqdns:
                        continue
                    seen_fqdns.add(record.atom_fqdn)
                    records.append(record)
    return tuple(records)


def load_external_atom_heuristic_metadata(
    provider_roots: tuple[str | Path, ...] | None = None,
) -> tuple[AtomHeuristicMetadata, ...]:
    roots = _metadata_provider_roots(provider_roots)
    return _load_external_atom_heuristic_metadata_cached(
        tuple(str(root) for root in roots)
    )


def resolve_external_atom_heuristic_metadata(
    atom_fqdn: str,
    *,
    provider_roots: tuple[str | Path, ...] | None = None,
) -> AtomHeuristicMetadata | None:
    target = str(atom_fqdn or "").strip()
    if not target:
        return None
    for record in load_external_atom_heuristic_metadata(provider_roots):
        if record.atom_fqdn == target:
            return record
    return None


def clear_atom_heuristic_metadata_caches() -> None:
    """Clear metadata-loader caches used by tests and migration tooling."""
    _load_external_atom_heuristic_metadata_cached.cache_clear()
