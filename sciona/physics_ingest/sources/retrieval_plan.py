"""Side-effect-free retrieval planning for physics ingestion sources.

This module declares where production backfills should retrieve source
payloads from, without performing any network or filesystem IO.  Downstream
workers can consume the JSON-safe manifest to execute source-specific fetches,
write retrieval reports, or feed the existing offline adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sciona.physics_ingest.sources._manifest import jsonable


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class RetryPolicy:
    """Retry/backoff settings for a retrieval job."""

    max_attempts: int = 4
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: str = "full"
    retry_on_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class RateLimitHint:
    """Non-binding rate-limit guidance for downstream executors."""

    requests_per_second: float
    burst: int = 1
    concurrency: int = 1
    min_delay_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class PaginationSpec:
    """Cursor/limit metadata for a source retrieval endpoint."""

    strategy: str
    cursor_parameter: str = ""
    cursor_response_path: str = ""
    limit_parameter: str = ""
    default_limit: int | None = None
    max_limit: int | None = None
    terminal_condition: str = ""
    notes: str = ""

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class RetrievalEndpoint:
    """Declarative endpoint metadata for one source family."""

    endpoint_id: str
    source_system: str
    source_family: str
    snapshot_key: str
    adapter_name: str
    adapter_version: str
    method: str
    url: str
    endpoint_kind: str
    license_expression: str
    provenance_summary: str
    pagination: PaginationSpec
    rate_limit: RateLimitHint
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    headers: Mapping[str, str] = field(default_factory=dict)
    query_params: Mapping[str, Any] = field(default_factory=dict)
    body_template: Mapping[str, Any] = field(default_factory=dict)
    requires_auth: bool = False
    auth_hint: str = ""
    content_type: str = "application/json"
    notes: str = ""

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class RetrievalJob:
    """Executable planning unit derived from an endpoint."""

    job_id: str
    endpoint_id: str
    source_system: str
    source_family: str
    snapshot_key: str
    adapter_name: str
    adapter_version: str
    target_adapter_input: str
    enabled: bool = True
    priority: int = 100
    cursor: str | None = None
    limit: int | None = None
    request_overrides: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceRetrievalManifest:
    """JSON-safe retrieval manifest for the current physics source set."""

    manifest_version: str
    snapshot_key_prefix: str
    endpoints: tuple[RetrievalEndpoint, ...]
    jobs: tuple[RetrievalJob, ...]

    def to_dict(self) -> JSONDict:
        return {
            "manifest_version": self.manifest_version,
            "snapshot_key_prefix": self.snapshot_key_prefix,
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
            "jobs": [job.to_dict() for job in self.jobs],
        }

    def endpoint_by_id(self) -> dict[str, RetrievalEndpoint]:
        return {endpoint.endpoint_id: endpoint for endpoint in self.endpoints}

    def job_by_id(self) -> dict[str, RetrievalJob]:
        return {job.job_id: job for job in self.jobs}


def build_physics_source_retrieval_manifest(
    *,
    snapshot_key_prefix: str = "physics-ingest",
    manifest_version: str = "physics-source-retrieval-plan.v1",
) -> SourceRetrievalManifest:
    """Return the production retrieval plan without touching the network."""

    endpoints = _current_endpoints(snapshot_key_prefix=snapshot_key_prefix)
    jobs = tuple(_job_for_endpoint(endpoint) for endpoint in endpoints)
    _validate_manifest(endpoints=endpoints, jobs=jobs)
    return SourceRetrievalManifest(
        manifest_version=manifest_version,
        snapshot_key_prefix=snapshot_key_prefix,
        endpoints=endpoints,
        jobs=jobs,
    )


def build_physics_source_retrieval_manifest_dict(
    *,
    snapshot_key_prefix: str = "physics-ingest",
    manifest_version: str = "physics-source-retrieval-plan.v1",
) -> JSONDict:
    """Return ``build_physics_source_retrieval_manifest(...).to_dict()``."""

    return build_physics_source_retrieval_manifest(
        snapshot_key_prefix=snapshot_key_prefix,
        manifest_version=manifest_version,
    ).to_dict()


def _current_endpoints(*, snapshot_key_prefix: str) -> tuple[RetrievalEndpoint, ...]:
    cautious_retry = RetryPolicy(max_attempts=5, initial_delay_seconds=2.0)
    bulk_retry = RetryPolicy(
        max_attempts=6,
        initial_delay_seconds=5.0,
        max_delay_seconds=300.0,
        backoff_multiplier=2.0,
    )
    return (
        RetrievalEndpoint(
            endpoint_id="wikidata_equation_candidates",
            source_system="wikidata",
            source_family="knowledge_graph",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "wikidata"),
            adapter_name="sciona.physics_ingest.sources.wikidata",
            adapter_version="0.1.0",
            method="POST",
            url="https://query.wikidata.org/sparql",
            endpoint_kind="sparql",
            license_expression="CC0-1.0",
            provenance_summary=(
                "Wikidata SPARQL discovery for entities with equation-like "
                "formula statements and use relationships."
            ),
            pagination=PaginationSpec(
                strategy="offset_limit",
                cursor_parameter="OFFSET",
                limit_parameter="LIMIT",
                default_limit=500,
                max_limit=1000,
                terminal_condition="empty_result_bindings",
                notes="Cursor is the integer SPARQL OFFSET appended by executor.",
            ),
            rate_limit=RateLimitHint(
                requests_per_second=0.1,
                min_delay_seconds=10.0,
                notes="Respect Wikidata Query Service etiquette and set User-Agent.",
            ),
            retry_policy=cautious_retry,
            headers={"Accept": "application/sparql-results+json"},
            body_template={
                "query_builder": "build_physical_equation_candidates_query"
            },
            notes="Adapter already exposes the query builder; executor injects offset.",
        ),
        RetrievalEndpoint(
            endpoint_id="qudt_units_quantity_kinds",
            source_system="qudt",
            source_family="ontology",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "qudt"),
            adapter_name="sciona.physics_ingest.sources.qudt",
            adapter_version="wave1-qudt-dim-resolution-v2",
            method="GET",
            url="https://qudt.org/vocab/",
            endpoint_kind="rdf_dump",
            license_expression="CC-BY-4.0",
            provenance_summary=(
                "QUDT unit, quantity-kind, and dimension-vector vocabularies "
                "used to resolve dimensional signatures."
            ),
            pagination=PaginationSpec(
                strategy="static_artifact_set",
                terminal_condition="all_declared_artifacts_retrieved",
                notes="Executor retrieves configured RDF vocabulary artifacts.",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.2, min_delay_seconds=5.0),
            retry_policy=bulk_retry,
            content_type="text/turtle",
        ),
        RetrievalEndpoint(
            endpoint_id="pdg_derivation_graph",
            source_system="physics_derivation_graph",
            source_family="derivation_graph",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "pdg"),
            adapter_name="sciona.physics_ingest.sources.pdg",
            adapter_version="wave1.pdg_scaffold.v1",
            method="GET",
            url="https://github.com/woojin1063/Physics-Derivation-Graph",
            endpoint_kind="repository_snapshot",
            license_expression="upstream-license-required",
            provenance_summary=(
                "Physics Derivation Graph equation and inference-rule payloads "
                "captured from an immutable repository snapshot."
            ),
            pagination=PaginationSpec(
                strategy="repository_tree",
                cursor_parameter="path",
                limit_parameter="batch_size",
                default_limit=100,
                terminal_condition="all_matching_files_retrieved",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.5, concurrency=1),
            retry_policy=bulk_retry,
            content_type="application/json",
        ),
        RetrievalEndpoint(
            endpoint_id="nist_codata_constants",
            source_system="nist_codata",
            source_family="standards_reference",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "nist-codata"),
            adapter_name="sciona.physics_ingest.sources.nist",
            adapter_version="0.1.0",
            method="GET",
            url="https://physics.nist.gov/cuu/Constants/Table/allascii.txt",
            endpoint_kind="ascii_table",
            license_expression=(
                "NIST public-domain U.S. government work; verify downstream "
                "attribution and international reuse constraints."
            ),
            provenance_summary="NIST CODATA fundamental physical constants table.",
            pagination=PaginationSpec(
                strategy="single_artifact",
                terminal_condition="artifact_retrieved",
                notes="No cursor; file is parsed into CODATA constant records.",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.2, min_delay_seconds=5.0),
            retry_policy=bulk_retry,
            content_type="text/plain",
        ),
        RetrievalEndpoint(
            endpoint_id="nist_dlmf_equations",
            source_system="nist_dlmf",
            source_family="standards_reference",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "nist-dlmf"),
            adapter_name="sciona.physics_ingest.sources.nist",
            adapter_version="0.1.0",
            method="GET",
            url="https://dlmf.nist.gov/",
            endpoint_kind="html_or_xml_corpus",
            license_expression=(
                "NIST DLMF content; preserve DLMF attribution and verify "
                "redistribution terms before publishing derived payloads."
            ),
            provenance_summary=(
                "NIST Digital Library of Mathematical Functions equations and "
                "symbolic function metadata."
            ),
            pagination=PaginationSpec(
                strategy="section_cursor",
                cursor_parameter="section",
                limit_parameter="batch_size",
                default_limit=25,
                terminal_condition="all_declared_sections_retrieved",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.2, min_delay_seconds=5.0),
            retry_policy=bulk_retry,
            content_type="text/html",
        ),
        RetrievalEndpoint(
            endpoint_id="hitran_lines",
            source_system="hitran",
            source_family="spectroscopy",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "hitran"),
            adapter_name="sciona.physics_ingest.sources.hitran",
            adapter_version="wave1.hitran_scaffold.v1",
            method="GET",
            url="https://hitran.org/",
            endpoint_kind="api_or_bulk_export",
            license_expression=(
                "HITRAN license/citation terms required; preserve upstream "
                "citation and redistribution constraints."
            ),
            provenance_summary=(
                "HITRAN spectral line and molecular property payloads for "
                "physics reference data ingestion."
            ),
            pagination=PaginationSpec(
                strategy="cursor_limit",
                cursor_parameter="offset",
                limit_parameter="limit",
                default_limit=1000,
                max_limit=10000,
                terminal_condition="fewer_than_limit_records",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.5, concurrency=1),
            retry_policy=cautious_retry,
            requires_auth=True,
            auth_hint="HITRAN account/API credentials if executor uses API access.",
        ),
        RetrievalEndpoint(
            endpoint_id="materials_project_documents",
            source_system="materials_project",
            source_family="materials",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "materials-project"),
            adapter_name="sciona.physics_ingest.sources.materials_project",
            adapter_version="wave1.materials_project_scaffold.v1",
            method="POST",
            url="https://api.materialsproject.org/materials/summary/",
            endpoint_kind="json_api",
            license_expression=(
                "Materials Project API terms and citation requirements apply; "
                "preserve provenance for computed materials metadata."
            ),
            provenance_summary=(
                "Materials Project computed materials documents and property "
                "mappings for future data artifacts."
            ),
            pagination=PaginationSpec(
                strategy="offset_limit",
                cursor_parameter="_skip",
                limit_parameter="_limit",
                default_limit=1000,
                max_limit=1000,
                terminal_condition="fewer_than_limit_records",
            ),
            rate_limit=RateLimitHint(requests_per_second=2.0, concurrency=1),
            retry_policy=cautious_retry,
            requires_auth=True,
            auth_hint="Materials Project API key.",
        ),
        RetrievalEndpoint(
            endpoint_id="opb_problem_payloads",
            source_system="opb",
            source_family="benchmark",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "opb"),
            adapter_name="sciona.physics_ingest.sources.opb",
            adapter_version="wave1.opb_scaffold.v1",
            method="GET",
            url="https://github.com/eth-sri/OPB",
            endpoint_kind="repository_snapshot",
            license_expression=(
                "OPB upstream benchmark/problem license required before "
                "redistribution."
            ),
            provenance_summary=(
                "OPB problem/equation payloads captured from immutable "
                "repository snapshots."
            ),
            pagination=PaginationSpec(
                strategy="repository_tree",
                cursor_parameter="path",
                limit_parameter="batch_size",
                default_limit=100,
                terminal_condition="all_matching_files_retrieved",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.5),
            retry_policy=bulk_retry,
        ),
        RetrievalEndpoint(
            endpoint_id="theoria_payloads",
            source_system="theoria",
            source_family="benchmark",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "theoria"),
            adapter_name="sciona.physics_ingest.sources.theoria",
            adapter_version="wave1.theoria_scaffold.v1",
            method="GET",
            url="https://github.com/martius-lab/TheorIA",
            endpoint_kind="repository_snapshot",
            license_expression=(
                "TheorIA upstream dataset/license terms required before "
                "redistribution."
            ),
            provenance_summary=(
                "TheorIA theory, problem, and evaluation payloads for raw "
                "symbolic candidate ingestion."
            ),
            pagination=PaginationSpec(
                strategy="repository_tree",
                cursor_parameter="path",
                limit_parameter="batch_size",
                default_limit=100,
                terminal_condition="all_matching_files_retrieved",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.5),
            retry_policy=bulk_retry,
        ),
        RetrievalEndpoint(
            endpoint_id="phy_srbench_payloads",
            source_system="phy_srbench",
            source_family="benchmark",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "phy-srbench"),
            adapter_name="sciona.physics_ingest.sources.phy_srbench",
            adapter_version="wave1.phy_srbench_scaffold.v1",
            method="GET",
            url="https://github.com/cavalab/srbench",
            endpoint_kind="repository_snapshot",
            license_expression=(
                "Phy-SRBench/SRBench upstream dataset and benchmark license "
                "terms required before redistribution."
            ),
            provenance_summary=(
                "Phy-SRBench symbolic-regression task and benchmark metadata."
            ),
            pagination=PaginationSpec(
                strategy="repository_tree",
                cursor_parameter="path",
                limit_parameter="batch_size",
                default_limit=100,
                terminal_condition="all_matching_files_retrieved",
            ),
            rate_limit=RateLimitHint(requests_per_second=0.5),
            retry_policy=bulk_retry,
        ),
        RetrievalEndpoint(
            endpoint_id="foundational_manual_seed",
            source_system="manual",
            source_family="curated_foundational",
            snapshot_key=_snapshot_key(snapshot_key_prefix, "foundational-manual"),
            adapter_name="sciona.physics_ingest.sources.foundational_physics",
            adapter_version="wave1.foundational_physics_backfill.v1",
            method="MANUAL",
            url="manual://sciona.physics_ingest/foundational_physics/v1",
            endpoint_kind="curated_seed",
            license_expression=(
                "Curated factual formula metadata; references identify source "
                "works for human verification before publication."
            ),
            provenance_summary=(
                "Deterministic manual seed set for foundational physics laws."
            ),
            pagination=PaginationSpec(
                strategy="in_memory_seed_set",
                terminal_condition="all_declared_seed_records_emitted",
                notes="No network retrieval; adapter owns the curated records.",
            ),
            rate_limit=RateLimitHint(
                requests_per_second=0.0,
                notes="No external requests are made for manual seeds.",
            ),
            retry_policy=RetryPolicy(max_attempts=1, retry_on_statuses=()),
            content_type="application/json",
        ),
    )


def _job_for_endpoint(endpoint: RetrievalEndpoint) -> RetrievalJob:
    return RetrievalJob(
        job_id=f"{endpoint.endpoint_id}.backfill",
        endpoint_id=endpoint.endpoint_id,
        source_system=endpoint.source_system,
        source_family=endpoint.source_family,
        snapshot_key=endpoint.snapshot_key,
        adapter_name=endpoint.adapter_name,
        adapter_version=endpoint.adapter_version,
        target_adapter_input=_target_adapter_input(endpoint),
        limit=endpoint.pagination.default_limit,
        priority=_source_priority(endpoint.source_system),
        provenance={
            "license_expression": endpoint.license_expression,
            "provenance_summary": endpoint.provenance_summary,
        },
    )


def _snapshot_key(prefix: str, suffix: str) -> str:
    return f"{prefix}/{suffix}"


def _target_adapter_input(endpoint: RetrievalEndpoint) -> str:
    if endpoint.source_system == "manual":
        return "curated_seed_records"
    if endpoint.endpoint_kind in {"repository_snapshot", "rdf_dump"}:
        return "raw_records"
    if endpoint.endpoint_kind in {"ascii_table", "html_or_xml_corpus"}:
        return "raw_document"
    return "json_records"


def _source_priority(source_system: str) -> int:
    priority = {
        "manual": 10,
        "nist_codata": 20,
        "qudt": 30,
        "wikidata": 40,
        "nist_dlmf": 50,
        "physics_derivation_graph": 60,
        "hitran": 70,
        "materials_project": 80,
        "opb": 90,
        "theoria": 100,
        "phy_srbench": 110,
    }
    return priority[source_system]


def _validate_manifest(
    *,
    endpoints: tuple[RetrievalEndpoint, ...],
    jobs: tuple[RetrievalJob, ...],
) -> None:
    endpoint_ids = [endpoint.endpoint_id for endpoint in endpoints]
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError("retrieval endpoint ids must be unique")
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("retrieval job ids must be unique")
    endpoint_id_set = set(endpoint_ids)
    missing = sorted({job.endpoint_id for job in jobs} - endpoint_id_set)
    if missing:
        raise ValueError(f"retrieval jobs reference unknown endpoints: {missing}")
