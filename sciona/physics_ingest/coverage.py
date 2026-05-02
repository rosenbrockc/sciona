"""Offline Phase 7 coverage summaries for physics ingestion rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from sciona.physics_ingest.sources._manifest import jsonable


JSONDict = dict[str, Any]

REPORT_VERSION = "physics-phase7-coverage-summary.v1"

COVERAGE_COUNT_KEYS: tuple[str, ...] = (
    "discovered",
    "parsed",
    "dimensioned",
    "reviewed",
    "published",
    "blocked",
)

_PARSED_CANDIDATE_STATUSES = {
    "parsed",
    "dimension_resolved",
    "symbolically_validated",
    "source_verified",
    "human_reviewed",
    "published",
}
_DIMENSIONED_CANDIDATE_STATUSES = {
    "dimension_resolved",
    "symbolically_validated",
    "source_verified",
    "human_reviewed",
    "published",
}
_REVIEWED_CANDIDATE_STATUSES = {"human_reviewed", "published"}
_PUBLISHED_CANDIDATE_STATUSES = {"published"}
_BLOCKED_CANDIDATE_STATUSES = {"blocked", "parse_failed"}

_PARSED_PARSE_STATUSES = {"parsed", "normalized"}
_BLOCKED_PARSE_STATUSES = {"blocked", "parse_failed"}
_REVIEWED_REVIEW_STATUSES = {"automated_pass", "human_reviewed"}
_BLOCKED_REVIEW_STATUSES = {"blocked"}
_PASSED_VALUES = {"pass", "passed", "ok", "success", "succeeded", "true"}
_FAILED_VALUES = {"fail", "failed", "error", "blocked", "false"}

PHASE7_RING_LABELS: dict[str, str] = {
    "ring_1_foundational_physics": (
        "Foundational mechanics, thermodynamics, electromagnetism, waves, and transport"
    ),
    "ring_2_existing_sciona_domains": (
        "Existing Sciona domains: biosignals, imaging, particle tracking, "
        "astrophysics, and materials"
    ),
    "ring_3_wikidata_equations": "Full Wikidata physical equation corpus",
    "ring_4_pdg_derivations": (
        "Full Physics Derivation Graph equation and derivation corpus"
    ),
    "ring_5_reference_datasets": (
        "Constants, spectra, materials, and property datasets"
    ),
    "ring_6_long_tail": "Long-tail equations with lower metadata quality",
    "unknown": "Unknown or unassigned",
}

_PHASE7_RING_ALIASES: dict[str, str] = {
    "1": "ring_1_foundational_physics",
    "ring 1": "ring_1_foundational_physics",
    "ring 1 foundational physics": "ring_1_foundational_physics",
    "foundational": "ring_1_foundational_physics",
    "foundational physics": "ring_1_foundational_physics",
    "2": "ring_2_existing_sciona_domains",
    "ring 2": "ring_2_existing_sciona_domains",
    "ring 2 existing sciona domains": "ring_2_existing_sciona_domains",
    "existing sciona domains": "ring_2_existing_sciona_domains",
    "sciona domains": "ring_2_existing_sciona_domains",
    "3": "ring_3_wikidata_equations",
    "ring 3": "ring_3_wikidata_equations",
    "ring 3 wikidata equations": "ring_3_wikidata_equations",
    "wikidata": "ring_3_wikidata_equations",
    "wikidata equations": "ring_3_wikidata_equations",
    "4": "ring_4_pdg_derivations",
    "ring 4": "ring_4_pdg_derivations",
    "ring 4 pdg derivations": "ring_4_pdg_derivations",
    "pdg": "ring_4_pdg_derivations",
    "physics derivation graph": "ring_4_pdg_derivations",
    "5": "ring_5_reference_datasets",
    "ring 5": "ring_5_reference_datasets",
    "ring 5 reference datasets": "ring_5_reference_datasets",
    "reference datasets": "ring_5_reference_datasets",
    "property datasets": "ring_5_reference_datasets",
    "6": "ring_6_long_tail",
    "ring 6": "ring_6_long_tail",
    "ring 6 long tail": "ring_6_long_tail",
    "long tail": "ring_6_long_tail",
}

_FOUNDATIONAL_PHYSICS_FAMILIES = {
    "mechanics",
    "classical mechanics",
    "classical_mechanics",
    "thermodynamics",
    "electromagnetism",
    "waves",
    "wave propagation",
    "wave_propagation",
    "transport",
    "transport phenomena",
}
_EXISTING_SCIONA_PHYSICS_FAMILIES = {
    "biosignals",
    "bio signals",
    "imaging",
    "particle tracking",
    "particle_tracking",
    "astrophysics",
    "materials",
    "materials science",
}
_REFERENCE_DATA_SOURCE_SYSTEMS = {
    "nist codata",
    "nist_codata",
    "codata",
    "nist dlmf",
    "nist_dlmf",
    "dlmf",
    "hitran",
    "materials project",
    "materials_project",
}
_REFERENCE_DATA_SOURCE_FAMILIES = {
    "reference data",
    "reference_data",
    "constants",
    "spectra",
    "property data",
    "property_data",
}


@dataclass(frozen=True)
class Phase7CoverageBucket:
    """Coverage counts for one dashboard grouping bucket."""

    key: Mapping[str, str]
    counts: Mapping[str, int]
    metrics: Mapping[str, int | float]

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class Phase7CoverageSummary:
    """JSON-safe, side-effect-free Phase 7 coverage report."""

    report_version: str
    summary: Mapping[str, Any]
    by_source: tuple[Phase7CoverageBucket, ...]
    by_phase7_ring: tuple[Phase7CoverageBucket, ...]
    by_physics_family: tuple[Phase7CoverageBucket, ...]
    by_source_and_physics_family: tuple[Phase7CoverageBucket, ...]
    by_phase7_ring_and_physics_family: tuple[Phase7CoverageBucket, ...]

    def to_dict(self) -> JSONDict:
        return jsonable(self)


def build_phase7_coverage_summary(
    rows: Iterable[Mapping[str, Any] | Any],
) -> Phase7CoverageSummary:
    """Build deterministic coverage counts from already-local ingest rows.

    The input may contain dictionaries, Pydantic rows, dataclasses, or simple
    row-like objects from ``physics_equation_candidates``,
    ``artifact_symbolic_expressions``, or source adapters. The function never
    performs database or network IO.
    """

    total_counts = _empty_counts()
    source_counts: dict[tuple[str, str], dict[str, int]] = {}
    ring_counts: dict[str, dict[str, int]] = {}
    family_counts: dict[str, dict[str, int]] = {}
    source_family_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    ring_family_counts: dict[tuple[str, str], dict[str, int]] = {}
    total_rows = 0

    for raw_row in rows:
        row = _row_dict(raw_row)
        counts = _coverage_counts(row)
        source_system, source_family = _source_key(row)
        physics_families = _physics_families(row)
        phase7_ring = _phase7_ring(row, source_system, source_family, physics_families)

        total_rows += 1
        _add_counts(total_counts, counts)
        _add_counts(
            source_counts.setdefault((source_system, source_family), _empty_counts()),
            counts,
        )
        _add_counts(
            ring_counts.setdefault(phase7_ring, _empty_counts()),
            counts,
        )
        for physics_family in physics_families:
            _add_counts(
                family_counts.setdefault(physics_family, _empty_counts()),
                counts,
            )
            _add_counts(
                source_family_counts.setdefault(
                    (source_system, source_family, physics_family),
                    _empty_counts(),
                ),
                counts,
            )
            _add_counts(
                ring_family_counts.setdefault(
                    (phase7_ring, physics_family),
                    _empty_counts(),
                ),
                counts,
            )

    return Phase7CoverageSummary(
        report_version=REPORT_VERSION,
        summary=_summary_counts_and_metrics(total_rows, total_counts),
        by_source=tuple(
            Phase7CoverageBucket(
                key={"source_system": source_system, "source_family": source_family},
                counts=dict(counts),
                metrics=_coverage_metrics(counts),
            )
            for (source_system, source_family), counts in sorted(source_counts.items())
        ),
        by_phase7_ring=tuple(
            Phase7CoverageBucket(
                key={
                    "phase7_ring": phase7_ring,
                    "phase7_ring_label": PHASE7_RING_LABELS.get(
                        phase7_ring,
                        PHASE7_RING_LABELS["unknown"],
                    ),
                },
                counts=dict(counts),
                metrics=_coverage_metrics(counts),
            )
            for phase7_ring, counts in sorted(ring_counts.items())
        ),
        by_physics_family=tuple(
            Phase7CoverageBucket(
                key={"physics_family": physics_family},
                counts=dict(counts),
                metrics=_coverage_metrics(counts),
            )
            for physics_family, counts in sorted(family_counts.items())
        ),
        by_source_and_physics_family=tuple(
            Phase7CoverageBucket(
                key={
                    "source_system": source_system,
                    "source_family": source_family,
                    "physics_family": physics_family,
                },
                counts=dict(counts),
                metrics=_coverage_metrics(counts),
            )
            for (
                source_system,
                source_family,
                physics_family,
            ), counts in sorted(source_family_counts.items())
        ),
        by_phase7_ring_and_physics_family=tuple(
            Phase7CoverageBucket(
                key={
                    "phase7_ring": phase7_ring,
                    "phase7_ring_label": PHASE7_RING_LABELS.get(
                        phase7_ring,
                        PHASE7_RING_LABELS["unknown"],
                    ),
                    "physics_family": physics_family,
                },
                counts=dict(counts),
                metrics=_coverage_metrics(counts),
            )
            for (phase7_ring, physics_family), counts in sorted(
                ring_family_counts.items()
            )
        ),
    )


def build_phase7_coverage_summary_dict(
    rows: Iterable[Mapping[str, Any] | Any],
) -> JSONDict:
    """Return ``build_phase7_coverage_summary(rows).to_dict()``."""

    return build_phase7_coverage_summary(rows).to_dict()


def _coverage_counts(row: Mapping[str, Any]) -> dict[str, int]:
    counts = _empty_counts()
    counts["discovered"] = 1

    candidate_status = _text(row, "candidate_status")
    parse_status = _text(row, "parse_status")
    review_status = _text(row, "review_status")
    validation_status = _text(row, "validation_status")
    publication_status = _text(row, "publication_status", "published_status")

    if (
        candidate_status in _PARSED_CANDIDATE_STATUSES
        or parse_status in _PARSED_PARSE_STATUSES
        or bool(_text(row, "sympy_srepr", "canonical_expr_hash", "topology_hash"))
    ):
        counts["parsed"] = 1

    if (
        candidate_status in _DIMENSIONED_CANDIDATE_STATUSES
        or bool(_text(row, "dimensional_hash", "dim_signature"))
        or _payload_status(row, "dimension_status") in _PASSED_VALUES
        or _nested_status(row, ("evidence_json", "dimensional_analysis", "status"))
        in _PASSED_VALUES
        or _nested_status(row, ("evidence_json", "dimension_check", "status"))
        in _PASSED_VALUES
    ):
        counts["dimensioned"] = 1

    if (
        candidate_status in _REVIEWED_CANDIDATE_STATUSES
        or review_status in _REVIEWED_REVIEW_STATUSES
        or bool(_nested_value(row, ("evidence_json", "human_review")))
    ):
        counts["reviewed"] = 1

    if (
        candidate_status in _PUBLISHED_CANDIDATE_STATUSES
        or publication_status == "published"
        or _truthy(row.get("published"))
        or bool(_text(row, "published_at"))
    ):
        counts["published"] = 1

    if (
        candidate_status in _BLOCKED_CANDIDATE_STATUSES
        or parse_status in _BLOCKED_PARSE_STATUSES
        or review_status in _BLOCKED_REVIEW_STATUSES
        or validation_status in _FAILED_VALUES
        or _payload_status(row, "dimension_status") in _FAILED_VALUES
        or bool(row.get("blockers"))
    ):
        counts["blocked"] = 1

    return counts


def _source_key(row: Mapping[str, Any]) -> tuple[str, str]:
    payload = _mapping(row.get("source_payload"))
    evidence_source = _mapping(_nested_value(row, ("evidence_json", "source")))
    provenance = _mapping(row.get("provenance"))
    source_system = _first_text(
        row.get("source_system"),
        payload.get("source_system"),
        payload.get("source"),
        evidence_source.get("source_system"),
        evidence_source.get("source"),
        provenance.get("source_system"),
        provenance.get("source"),
        row.get("adapter_name"),
        "unknown",
    )
    source_family = _first_text(
        row.get("source_family"),
        payload.get("source_family"),
        payload.get("family"),
        evidence_source.get("source_family"),
        provenance.get("source_family"),
        source_system,
        "unknown",
    )
    return source_system, source_family


def _physics_families(row: Mapping[str, Any]) -> tuple[str, ...]:
    payload = _mapping(row.get("source_payload"))
    direct = _first_text(
        row.get("physics_family"),
        row.get("physics_domain"),
        payload.get("physics_family"),
        payload.get("physics_domain"),
        payload.get("domain"),
    )
    if direct:
        return (direct,)

    tags = [
        str(tag)
        for tag in _sequence(row.get("mechanism_tags") or payload.get("mechanism_tags"))
        if str(tag)
    ]
    if tags:
        return tuple(sorted(dict.fromkeys(tags)))
    return ("unknown",)


def _phase7_ring(
    row: Mapping[str, Any],
    source_system: str,
    source_family: str,
    physics_families: tuple[str, ...],
) -> str:
    payload = _mapping(row.get("source_payload"))
    explicit = _first_text(
        row.get("phase7_ring"),
        row.get("phase_7_ring"),
        row.get("backfill_ring"),
        row.get("ingestion_ring"),
        row.get("ring"),
        payload.get("phase7_ring"),
        payload.get("phase_7_ring"),
        payload.get("backfill_ring"),
        payload.get("ingestion_ring"),
        payload.get("ring"),
    )
    ring = _phase7_ring_alias(explicit)
    if ring:
        return ring

    source_system_norm = _norm(source_system)
    source_family_norm = _norm(source_family)
    family_norms = {_norm(family) for family in physics_families}
    if source_system_norm == "wikidata":
        return "ring_3_wikidata_equations"
    if source_system_norm in {"pdg", "physics derivation graph"}:
        return "ring_4_pdg_derivations"
    if source_family_norm == "derivation graph":
        return "ring_4_pdg_derivations"
    if (
        source_system_norm in {_norm(value) for value in _REFERENCE_DATA_SOURCE_SYSTEMS}
        or source_family_norm in {_norm(value) for value in _REFERENCE_DATA_SOURCE_FAMILIES}
    ):
        return "ring_5_reference_datasets"
    if family_norms & {_norm(value) for value in _FOUNDATIONAL_PHYSICS_FAMILIES}:
        return "ring_1_foundational_physics"
    if family_norms & {_norm(value) for value in _EXISTING_SCIONA_PHYSICS_FAMILIES}:
        return "ring_2_existing_sciona_domains"
    if source_family_norm in {"long tail", "lower metadata quality"}:
        return "ring_6_long_tail"
    return "unknown"


def _phase7_ring_alias(value: str) -> str:
    if not value:
        return ""
    normalized = _norm(value)
    if normalized in _PHASE7_RING_ALIASES:
        return _PHASE7_RING_ALIASES[normalized]
    return value if value in PHASE7_RING_LABELS else ""


def _row_dict(value: Mapping[str, Any] | Any) -> JSONDict:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json", exclude_none=True))
    if hasattr(value, "to_insert_dict"):
        return dict(value.to_insert_dict())
    if hasattr(value, "to_dict"):
        row = value.to_dict()
        if isinstance(row, Mapping):
            return dict(row)
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"cannot treat {type(value)!r} as a coverage row")


def _empty_counts() -> dict[str, int]:
    return {key: 0 for key in COVERAGE_COUNT_KEYS}


def _add_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key in COVERAGE_COUNT_KEYS:
        target[key] += int(source.get(key, 0))


def _summary_counts_and_metrics(
    total_rows: int,
    counts: Mapping[str, int],
) -> JSONDict:
    return {
        "total_rows": total_rows,
        **dict(counts),
        "metrics": _coverage_metrics(counts),
    }


def _coverage_metrics(counts: Mapping[str, int]) -> JSONDict:
    discovered = int(counts.get("discovered", 0))
    parsed = int(counts.get("parsed", 0))
    dimensioned = int(counts.get("dimensioned", 0))
    reviewed = int(counts.get("reviewed", 0))
    published = int(counts.get("published", 0))
    blocked = int(counts.get("blocked", 0))

    return {
        "parsed_rate": _rate(parsed, discovered),
        "dimensioned_rate": _rate(dimensioned, discovered),
        "reviewed_rate": _rate(reviewed, discovered),
        "published_rate": _rate(published, discovered),
        "blocked_rate": _rate(blocked, discovered),
        "discovered_to_parsed_loss": _loss(discovered, parsed),
        "parsed_to_dimensioned_loss": _loss(parsed, dimensioned),
        "dimensioned_to_reviewed_loss": _loss(dimensioned, reviewed),
        "reviewed_to_published_loss": _loss(reviewed, published),
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _loss(previous_stage_count: int, next_stage_count: int) -> int:
    return max(previous_stage_count - next_stage_count, 0)


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return str(value).strip().lower()
    return ""


def _norm(value: str) -> str:
    return " ".join(
        str(value).strip().casefold().replace("_", " ").replace("-", " ").split()
    )


def _first_text(*values: Any) -> str:
    for value in values:
        if value is not None and value != "":
            return str(value).strip()
    return ""


def _payload_status(row: Mapping[str, Any], key: str) -> str:
    return str(_mapping(row.get("source_payload")).get(key, "")).strip().lower()


def _nested_status(row: Mapping[str, Any], path: tuple[str, ...]) -> str:
    return str(_nested_value(row, path) or "").strip().lower()


def _nested_value(row: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        value = _mapping(value).get(key)
        if value is None:
            return None
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str) or value is None:
        return ()
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _PASSED_VALUES
