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


@dataclass(frozen=True)
class Phase7CoverageBucket:
    """Coverage counts for one dashboard grouping bucket."""

    key: Mapping[str, str]
    counts: Mapping[str, int]

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class Phase7CoverageSummary:
    """JSON-safe, side-effect-free Phase 7 coverage report."""

    report_version: str
    summary: Mapping[str, int]
    by_source: tuple[Phase7CoverageBucket, ...]
    by_physics_family: tuple[Phase7CoverageBucket, ...]
    by_source_and_physics_family: tuple[Phase7CoverageBucket, ...]

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
    family_counts: dict[str, dict[str, int]] = {}
    source_family_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    total_rows = 0

    for raw_row in rows:
        row = _row_dict(raw_row)
        counts = _coverage_counts(row)
        source_system, source_family = _source_key(row)
        physics_families = _physics_families(row)

        total_rows += 1
        _add_counts(total_counts, counts)
        _add_counts(
            source_counts.setdefault((source_system, source_family), _empty_counts()),
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

    return Phase7CoverageSummary(
        report_version=REPORT_VERSION,
        summary={"total_rows": total_rows, **total_counts},
        by_source=tuple(
            Phase7CoverageBucket(
                key={"source_system": source_system, "source_family": source_family},
                counts=dict(counts),
            )
            for (source_system, source_family), counts in sorted(source_counts.items())
        ),
        by_physics_family=tuple(
            Phase7CoverageBucket(
                key={"physics_family": physics_family},
                counts=dict(counts),
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
            )
            for (
                source_system,
                source_family,
                physics_family,
            ), counts in sorted(source_family_counts.items())
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


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return str(value).strip().lower()
    return ""


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
