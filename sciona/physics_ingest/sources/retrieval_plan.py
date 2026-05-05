"""Side-effect-free retrieval planning for physics ingestion sources.

This module declares where production backfills should retrieve source
payloads from, without performing any network or filesystem IO.  Downstream
workers can consume the JSON-safe manifest to execute source-specific fetches,
write retrieval reports, or feed the existing offline adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from sciona.physics_ingest.sources._manifest import jsonable, stable_payload_sha256


JSONDict = dict[str, Any]


PHASE7_RING_ORDER: Mapping[str, int] = {
    "ring_1_foundational": 1,
    "ring_2_existing_sciona_domains": 2,
    "ring_3_wikidata_physical_equations": 3,
    "ring_4_pdg_derivations": 4,
    "ring_5_reference_datasets": 5,
    "ring_6_long_tail": 6,
}

_PHASE7_PRIMARY_RING_BY_SOURCE_SYSTEM: Mapping[str, str] = {
    "manual": "ring_1_foundational",
    "nist_codata": "ring_1_foundational",
    "qudt": "ring_1_foundational",
    "nist_dlmf": "ring_1_foundational",
    "hitran": "ring_2_existing_sciona_domains",
    "materials_project": "ring_2_existing_sciona_domains",
    "opb": "ring_2_existing_sciona_domains",
    "wikidata": "ring_3_wikidata_physical_equations",
    "physics_derivation_graph": "ring_4_pdg_derivations",
    "theoria": "ring_6_long_tail",
    "phy_srbench": "ring_6_long_tail",
}

_PHASE7_ADDITIONAL_RINGS_BY_SOURCE_SYSTEM: Mapping[str, tuple[str, ...]] = {
    "nist_codata": ("ring_5_reference_datasets",),
    "qudt": ("ring_5_reference_datasets",),
    "hitran": ("ring_5_reference_datasets",),
    "materials_project": ("ring_5_reference_datasets",),
}


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
    phase7_ring: str
    phase7_ring_order: int
    phase7_rings: tuple[str, ...]
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
    phase7_ring: str
    phase7_ring_order: int
    phase7_rings: tuple[str, ...]
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


@dataclass(frozen=True)
class RetrievalRunDiagnostic:
    """Warning emitted while building a side-effect-free retrieval run plan."""

    severity: str
    job_id: str
    endpoint_id: str
    message: str

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class RetrievalRunStep:
    """Deterministic executor-facing retrieval step with no IO behavior."""

    step_index: int
    job_id: str
    endpoint_id: str
    source_system: str
    source_family: str
    phase7_ring: str
    phase7_ring_order: int
    phase7_rings: tuple[str, ...]
    snapshot_key: str
    method: str
    url: str
    endpoint_kind: str
    params: Mapping[str, Any]
    body_template: Mapping[str, Any]
    paging: Mapping[str, Any]
    headers: Mapping[str, str]
    content_type: str
    requires_auth: bool
    auth_hint: str
    rate_limit: Mapping[str, Any]
    retry_policy: Mapping[str, Any]
    provenance: Mapping[str, Any]
    adapter_module: str
    adapter_version: str
    target_adapter_input: str
    dry_run: bool
    replay_key: str
    request_envelope: Mapping[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class RetrievalRunPlan:
    """Side-effect-free execution plan derived from a retrieval manifest."""

    manifest_version: str
    snapshot_key_prefix: str
    dry_run: bool
    filters: Mapping[str, Any]
    steps: tuple[RetrievalRunStep, ...]
    diagnostics: tuple[RetrievalRunDiagnostic, ...] = ()

    def to_dict(self) -> JSONDict:
        return {
            "manifest_version": self.manifest_version,
            "snapshot_key_prefix": self.snapshot_key_prefix,
            "dry_run": self.dry_run,
            "filters": jsonable(self.filters),
            "summary": _run_plan_summary(
                steps=self.steps,
                diagnostics=self.diagnostics,
                dry_run=self.dry_run,
            ),
            "steps": [step.to_dict() for step in self.steps],
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


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


def build_physics_source_retrieval_run_plan(
    *,
    snapshot_key_prefix: str = "physics-ingest",
    manifest_version: str = "physics-source-retrieval-plan.v1",
    manifest: SourceRetrievalManifest | None = None,
    source_system: str | Iterable[str] | None = None,
    source_family: str | Iterable[str] | None = None,
    phase7_ring: str | Iterable[str] | None = None,
    job_id: str | Iterable[str] | None = None,
    max_jobs: int | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> RetrievalRunPlan:
    """Build deterministic retrieval execution steps without doing IO.

    ``limit`` overrides each selected job's per-request page size in the run
    plan only; the manifest and its jobs remain unchanged.
    """

    if max_jobs is not None and max_jobs < 0:
        raise ValueError("max_jobs must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    source_systems = _normalize_filter(source_system)
    source_families = _normalize_filter(source_family)
    phase7_rings = _normalize_filter(phase7_ring)
    job_ids = _normalize_filter(job_id)
    base_manifest = manifest or build_physics_source_retrieval_manifest(
        snapshot_key_prefix=snapshot_key_prefix,
        manifest_version=manifest_version,
    )
    endpoints = base_manifest.endpoint_by_id()
    selected_jobs = [
        job
        for job in sorted(base_manifest.jobs, key=lambda item: (item.priority, item.job_id))
        if _matches_filter(job.source_system, source_systems)
        and _matches_filter(job.source_family, source_families)
        and _matches_any_filter(job.phase7_rings, phase7_rings)
        and _matches_filter(job.job_id, job_ids)
    ]
    if max_jobs is not None:
        selected_jobs = selected_jobs[:max_jobs]

    diagnostics: list[RetrievalRunDiagnostic] = []
    steps: list[RetrievalRunStep] = []
    for step_index, job in enumerate(selected_jobs, start=1):
        endpoint = endpoints[job.endpoint_id]
        step, step_diagnostics = _run_step_for_job(
            step_index=step_index,
            job=job,
            endpoint=endpoint,
            limit=limit,
            dry_run=dry_run,
        )
        steps.append(step)
        diagnostics.extend(step_diagnostics)

    return RetrievalRunPlan(
        manifest_version=base_manifest.manifest_version,
        snapshot_key_prefix=base_manifest.snapshot_key_prefix,
        dry_run=dry_run,
        filters={
            "source_system": sorted(source_systems)
            if source_systems is not None
            else None,
            "source_family": sorted(source_families)
            if source_families is not None
            else None,
            "phase7_ring": sorted(phase7_rings)
            if phase7_rings is not None
            else None,
            "job_id": sorted(job_ids) if job_ids is not None else None,
            "max_jobs": max_jobs,
            "limit": limit,
        },
        steps=tuple(steps),
        diagnostics=tuple(diagnostics),
    )


def build_physics_source_retrieval_run_plan_dict(
    **kwargs: Any,
) -> JSONDict:
    """Return ``build_physics_source_retrieval_run_plan(...).to_dict()``."""

    return build_physics_source_retrieval_run_plan(**kwargs).to_dict()


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
            **_phase7_kwargs("wikidata"),
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
                "query_builder": "build_physics_ingestion_candidate_query"
            },
            notes=(
                "Adapter exposes a physics-scoped query builder for the first "
                "ingestion queue; use the broader formula query for long-tail "
                "P2534 discovery."
            ),
        ),
        RetrievalEndpoint(
            endpoint_id="qudt_units_quantity_kinds",
            source_system="qudt",
            source_family="ontology",
            **_phase7_kwargs("qudt"),
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
            **_phase7_kwargs("physics_derivation_graph"),
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
            **_phase7_kwargs("nist_codata"),
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
            **_phase7_kwargs("nist_dlmf"),
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
            **_phase7_kwargs("hitran"),
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
            **_phase7_kwargs("materials_project"),
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
            **_phase7_kwargs("opb"),
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
            **_phase7_kwargs("theoria"),
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
            **_phase7_kwargs("phy_srbench"),
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
            **_phase7_kwargs("manual"),
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
        phase7_ring=endpoint.phase7_ring,
        phase7_ring_order=endpoint.phase7_ring_order,
        phase7_rings=endpoint.phase7_rings,
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
    phase7_order = _phase7_ring_order(_phase7_primary_ring(source_system))
    source_order = {
        "manual": 0,
        "nist_codata": 1,
        "qudt": 2,
        "nist_dlmf": 3,
        "hitran": 0,
        "materials_project": 1,
        "opb": 2,
        "wikidata": 0,
        "physics_derivation_graph": 0,
        "theoria": 0,
        "phy_srbench": 1,
    }
    return phase7_order * 100 + source_order[source_system]


def _phase7_primary_ring(source_system: str) -> str:
    return _PHASE7_PRIMARY_RING_BY_SOURCE_SYSTEM[source_system]


def _phase7_rings(source_system: str) -> tuple[str, ...]:
    rings = {
        _phase7_primary_ring(source_system),
        *_PHASE7_ADDITIONAL_RINGS_BY_SOURCE_SYSTEM.get(source_system, ()),
    }
    return tuple(sorted(rings, key=_phase7_ring_order))


def _phase7_ring_order(phase7_ring: str) -> int:
    return PHASE7_RING_ORDER[phase7_ring]


def _phase7_kwargs(source_system: str) -> dict[str, Any]:
    primary_ring = _phase7_primary_ring(source_system)
    return {
        "phase7_ring": primary_ring,
        "phase7_ring_order": _phase7_ring_order(primary_ring),
        "phase7_rings": _phase7_rings(source_system),
    }


def _run_step_for_job(
    *,
    step_index: int,
    job: RetrievalJob,
    endpoint: RetrievalEndpoint,
    limit: int | None,
    dry_run: bool,
) -> tuple[RetrievalRunStep, tuple[RetrievalRunDiagnostic, ...]]:
    effective_limit = limit if limit is not None else job.limit
    params = _request_params(job=job, endpoint=endpoint, limit=effective_limit)
    paging = {
        **endpoint.pagination.to_dict(),
        "cursor": job.cursor,
        "limit": effective_limit,
    }
    body_template = dict(endpoint.body_template)
    headers = dict(endpoint.headers)
    rate_limit = endpoint.rate_limit.to_dict()
    retry_policy = endpoint.retry_policy.to_dict()
    provenance = {
        "license_expression": endpoint.license_expression,
        "provenance_summary": endpoint.provenance_summary,
    }
    auth = {
        "requires_auth": endpoint.requires_auth,
        "auth_hint": endpoint.auth_hint,
    }
    storage = _storage_hint(
        snapshot_key=job.snapshot_key,
        target_adapter_input=job.target_adapter_input,
        manual_source=_is_manual_endpoint(method=endpoint.method, url=endpoint.url),
    )
    adapter_target = {
        "module": endpoint.adapter_name,
        "version": endpoint.adapter_version,
        "target_input": job.target_adapter_input,
    }
    warnings = _run_step_warnings(job=job, endpoint=endpoint)
    replay_payload = {
        "job_id": job.job_id,
        "endpoint_id": endpoint.endpoint_id,
        "snapshot_key": job.snapshot_key,
        "phase7_ring": job.phase7_ring,
        "phase7_ring_order": job.phase7_ring_order,
        "phase7_rings": job.phase7_rings,
        "method": endpoint.method,
        "url": endpoint.url,
        "headers": headers,
        "params": params,
        "body_template": body_template,
        "paging": paging,
        "auth": auth,
        "storage": storage,
        "provenance": provenance,
        "retry_policy": retry_policy,
        "rate_limit": rate_limit,
        "adapter_target": adapter_target,
        "dry_run": dry_run,
    }
    replay_key = f"physics-source-retrieval:{stable_payload_sha256(replay_payload)}"
    request_envelope = _request_envelope(
        method=endpoint.method,
        url=endpoint.url,
        headers=headers,
        params=params,
        body_template=body_template,
        paging=paging,
        auth=auth,
        storage=storage,
        provenance=provenance,
        retry_policy=retry_policy,
        rate_limit=rate_limit,
        replay_key=replay_key,
        snapshot_key=job.snapshot_key,
        adapter_target=adapter_target,
        dry_run=dry_run,
    )
    step = RetrievalRunStep(
        step_index=step_index,
        job_id=job.job_id,
        endpoint_id=endpoint.endpoint_id,
        source_system=job.source_system,
        source_family=job.source_family,
        phase7_ring=job.phase7_ring,
        phase7_ring_order=job.phase7_ring_order,
        phase7_rings=job.phase7_rings,
        snapshot_key=job.snapshot_key,
        method=endpoint.method,
        url=endpoint.url,
        endpoint_kind=endpoint.endpoint_kind,
        params=params,
        body_template=body_template,
        paging=paging,
        headers=headers,
        content_type=endpoint.content_type,
        requires_auth=endpoint.requires_auth,
        auth_hint=endpoint.auth_hint,
        rate_limit=rate_limit,
        retry_policy=retry_policy,
        provenance=provenance,
        adapter_module=endpoint.adapter_name,
        adapter_version=endpoint.adapter_version,
        target_adapter_input=job.target_adapter_input,
        dry_run=dry_run,
        replay_key=replay_key,
        request_envelope=request_envelope,
        warnings=warnings,
    )
    diagnostics = tuple(
        RetrievalRunDiagnostic(
            severity="warning",
            job_id=job.job_id,
            endpoint_id=endpoint.endpoint_id,
            message=warning,
        )
        for warning in warnings
    )
    return step, diagnostics


def _request_envelope(
    *,
    method: str,
    url: str,
    headers: Mapping[str, Any],
    params: Mapping[str, Any],
    body_template: Mapping[str, Any],
    paging: Mapping[str, Any],
    auth: Mapping[str, Any],
    storage: Mapping[str, Any],
    provenance: Mapping[str, Any],
    retry_policy: Mapping[str, Any],
    rate_limit: Mapping[str, Any],
    replay_key: str,
    snapshot_key: str,
    adapter_target: Mapping[str, Any],
    dry_run: bool,
) -> JSONDict:
    manual_source = _is_manual_endpoint(method=method, url=url)
    return jsonable(
        {
            "envelope_version": "physics-source-request-envelope.v1",
            "method": method,
            "url": url,
            "headers": headers,
            "params": params,
            "body_template": body_template,
            "paging": paging,
            "auth": auth,
            "storage": storage,
            "provenance": provenance,
            "retry_policy": retry_policy,
            "rate_limit": rate_limit,
            "replay_key": replay_key,
            "snapshot_key": snapshot_key,
            "adapter_target": adapter_target,
            "execution": {
                "mode": "manual" if manual_source else "network",
                "dry_run": dry_run,
                "io_performed": False,
                "network_required": not manual_source,
                "network_io_allowed": False if manual_source else not dry_run,
                "manual_source": manual_source,
            },
        }
    )


def _storage_hint(
    *,
    snapshot_key: str,
    target_adapter_input: str,
    manual_source: bool,
) -> JSONDict:
    return {
        "snapshot_key": snapshot_key,
        "target_adapter_input": target_adapter_input,
        "storage_required": not manual_source,
        "write_required": False,
        "side_effect_free": True,
    }


def _is_manual_endpoint(*, method: str, url: str) -> bool:
    return method == "MANUAL" or url.startswith("manual://")


def _request_params(
    *,
    job: RetrievalJob,
    endpoint: RetrievalEndpoint,
    limit: int | None,
) -> dict[str, Any]:
    params = dict(endpoint.query_params)
    pagination = endpoint.pagination
    if pagination.cursor_parameter and job.cursor is not None:
        params[pagination.cursor_parameter] = job.cursor
    if pagination.limit_parameter and limit is not None:
        params[pagination.limit_parameter] = limit
    overrides = dict(job.request_overrides)
    override_params = overrides.pop("params", {})
    if isinstance(override_params, Mapping):
        params.update(override_params)
    params.update(overrides)
    return params


def _run_step_warnings(
    *,
    job: RetrievalJob,
    endpoint: RetrievalEndpoint,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not endpoint.url:
        warnings.append("endpoint url is missing")
    if not endpoint.method:
        warnings.append("endpoint method is missing")
    if not endpoint.endpoint_kind:
        warnings.append("endpoint kind is missing")
    if not endpoint.license_expression:
        warnings.append("license expression is missing")
    if not endpoint.provenance_summary:
        warnings.append("provenance summary is missing")
    if not job.adapter_name or not endpoint.adapter_name:
        warnings.append("adapter module is missing")
    if not job.adapter_version or not endpoint.adapter_version:
        warnings.append("adapter version is missing")
    return tuple(warnings)


def _run_plan_summary(
    *,
    steps: tuple[RetrievalRunStep, ...],
    diagnostics: tuple[RetrievalRunDiagnostic, ...],
    dry_run: bool,
) -> JSONDict:
    dry_run_step_count = sum(1 for step in steps if step.dry_run)
    return {
        "step_count": len(steps),
        "diagnostic_count": len(diagnostics),
        "dry_run": dry_run,
        "dry_run_step_count": dry_run_step_count,
        "non_dry_run_step_count": len(steps) - dry_run_step_count,
        "by_source_family": _step_count_rollup(
            step.source_family for step in steps
        ),
        "by_source_system": _step_count_rollup(
            step.source_system for step in steps
        ),
        "by_phase7_ring": _step_count_rollup(step.phase7_ring for step in steps),
        "by_endpoint_kind": _step_count_rollup(
            step.endpoint_kind for step in steps
        ),
        "by_method": _step_count_rollup(step.method for step in steps),
        "by_target_adapter_input": _step_count_rollup(
            step.target_adapter_input for step in steps
        ),
    }


def _step_count_rollup(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _normalize_filter(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {item for item in value}


def _matches_filter(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or value in allowed


def _matches_any_filter(values: Iterable[str], allowed: set[str] | None) -> bool:
    return allowed is None or bool(set(values) & allowed)


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
    endpoints_by_id = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
    for endpoint in endpoints:
        _validate_phase7_metadata(
            phase7_ring=endpoint.phase7_ring,
            phase7_ring_order=endpoint.phase7_ring_order,
            phase7_rings=endpoint.phase7_rings,
            owner_id=endpoint.endpoint_id,
        )
    for job in jobs:
        endpoint = endpoints_by_id[job.endpoint_id]
        if job.phase7_ring != endpoint.phase7_ring:
            raise ValueError(f"retrieval job phase7 ring mismatch: {job.job_id}")
        if job.phase7_ring_order != endpoint.phase7_ring_order:
            raise ValueError(f"retrieval job phase7 ring order mismatch: {job.job_id}")
        if job.phase7_rings != endpoint.phase7_rings:
            raise ValueError(f"retrieval job phase7 rings mismatch: {job.job_id}")


def _validate_phase7_metadata(
    *,
    phase7_ring: str,
    phase7_ring_order: int,
    phase7_rings: tuple[str, ...],
    owner_id: str,
) -> None:
    if phase7_ring not in PHASE7_RING_ORDER:
        raise ValueError(f"unknown phase7 ring for {owner_id}: {phase7_ring}")
    if phase7_ring_order != PHASE7_RING_ORDER[phase7_ring]:
        raise ValueError(f"phase7 ring order mismatch for {owner_id}")
    unknown_rings = sorted(set(phase7_rings) - set(PHASE7_RING_ORDER))
    if unknown_rings:
        raise ValueError(f"unknown phase7 rings for {owner_id}: {unknown_rings}")
    if phase7_ring not in phase7_rings:
        raise ValueError(f"primary phase7 ring missing from phase7 rings: {owner_id}")
    if phase7_rings != tuple(sorted(phase7_rings, key=_phase7_ring_order)):
        raise ValueError(f"phase7 rings must be sorted by ring order: {owner_id}")
