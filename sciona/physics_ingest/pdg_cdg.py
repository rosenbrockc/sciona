"""Phase 4 PDG relationship and derivation-CDG extraction scaffolds.

This module is intentionally side-effect free. It consumes parsed PDG payloads
from :mod:`sciona.physics_ingest.sources.pdg`, materializes validated
``artifact_relationships`` rows when symbolic expression ids are available, and
emits compact candidate manifests for algebraic derivation chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping
from uuid import NAMESPACE_URL, uuid5

from sciona.physics_ingest.sources.pdg import (
    PDGIngestBundle,
    PDGInferenceEdge,
    PDG_SOURCE_SYSTEM,
)
from sciona.physics_ingest.staging import (
    ArtifactRelationshipRow,
    validate_artifact_relationship_row,
)


JSONDict = dict[str, Any]

PHASE4_SCAFFOLD_VERSION = "phase4.pdg_cdg_scaffold.v1"
_CDG_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.pdg_cdg.version")
_CDG_ARTIFACT_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.pdg_cdg.artifact")

_CHAIN_OPERATIONS = frozenset(
    {
        "solve",
        "solve_for",
        "substitute",
        "substitution",
        "limit",
        "take_limit",
        "derive",
        "simplify",
        "differentiate",
        "integrate",
        "nondimensionalize",
        "approximate",
    }
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, data: Any) -> str:
    digest = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _manifest_operation_kind(edge: PDGInferenceEdge) -> str:
    operation = edge.operation_kind
    if operation == "solve_for":
        return "solve"
    if operation == "substitution":
        return "substitute"
    if operation == "take_limit":
        return "limit"
    if operation == "approximation":
        return "approximate"
    if operation in {
        "nondimensionalization",
        "non_dimensionalize",
        "non_dimensionalization",
    }:
        return "nondimensionalize"
    return operation


def _edge_is_chain_candidate(edge: PDGInferenceEdge) -> bool:
    return edge.operation_kind in _CHAIN_OPERATIONS or edge.relationship_kind in {
        "algebraic_rearrangement_of",
        "derives_from",
        "limit_case_of",
    }


@dataclass(frozen=True)
class PDGExpressionBinding:
    """Resolved symbolic expression endpoint for a PDG equation node."""

    pdg_node_id: str
    expression_id: str
    label: str = ""
    artifact_id: str = ""
    version_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, pdg_node_id: str, value: str | Mapping[str, Any]) -> "PDGExpressionBinding":
        if isinstance(value, str):
            return cls(pdg_node_id=pdg_node_id, expression_id=value)
        return cls(
            pdg_node_id=pdg_node_id,
            expression_id=str(value.get("expression_id") or ""),
            label=str(value.get("label") or ""),
            artifact_id=str(value.get("artifact_id") or ""),
            version_id=str(value.get("version_id") or ""),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_manifest_ref(self) -> JSONDict:
        ref: JSONDict = {
            "pdg_node_id": self.pdg_node_id,
            "expression_id": self.expression_id,
        }
        if self.label:
            ref["label"] = self.label
        if self.artifact_id:
            ref["artifact_id"] = self.artifact_id
        if self.version_id:
            ref["version_id"] = self.version_id
        if self.metadata:
            ref["metadata"] = dict(self.metadata)
        return ref


@dataclass(frozen=True)
class PDGRelationshipIngestResult:
    """Side-effect-free Phase 4 output for PDG relationship ingestion."""

    artifact_relationship_rows: tuple[ArtifactRelationshipRow, ...]
    cdg_candidate_manifests: tuple[JSONDict, ...]
    skipped_edges: tuple[JSONDict, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.summary:
            object.__setattr__(self, "summary", _json_safe_mapping(self.summary))
        else:
            object.__setattr__(
                self,
                "summary",
                _build_relationship_ingest_summary(self),
            )

    def relationship_insert_rows(self) -> list[JSONDict]:
        """Return JSON-ready rows for ``artifact_relationships`` insertion."""

        return [row.to_insert_dict() for row in self.artifact_relationship_rows]

    def to_dict(self) -> JSONDict:
        return {
            "artifact_relationship_rows": self.relationship_insert_rows(),
            "cdg_candidate_manifests": list(self.cdg_candidate_manifests),
            "skipped_edges": list(self.skipped_edges),
            "metadata": {
                "source_system": PDG_SOURCE_SYSTEM,
                "scaffold_version": PHASE4_SCAFFOLD_VERSION,
            },
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class PDGPublicationWriteRows:
    """PDG relationship/CDG rows ready for publication write planning."""

    insert_rows_by_table: Mapping[str, tuple[JSONDict, ...]]
    diagnostics: tuple[JSONDict, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.summary:
            object.__setattr__(self, "summary", _json_safe_mapping(self.summary))
        else:
            object.__setattr__(
                self,
                "summary",
                _build_publication_write_rows_summary(self),
            )

    def to_insert_rows(self) -> dict[str, list[JSONDict]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.insert_rows_by_table.items()
        }

    def to_dict(self) -> JSONDict:
        return {
            "insert_rows": self.to_insert_rows(),
            "diagnostics": list(self.diagnostics),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class PDGCDGCatalogProjectionRows:
    """Side-effect-free catalog/search projections for PDG-derived CDGs."""

    projection_rows_by_table: Mapping[str, tuple[JSONDict, ...]]
    diagnostics: tuple[JSONDict, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.summary:
            object.__setattr__(self, "summary", _json_safe_mapping(self.summary))
        else:
            object.__setattr__(
                self,
                "summary",
                _build_catalog_projection_summary(self),
            )

    def to_projection_rows(self) -> dict[str, list[JSONDict]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.projection_rows_by_table.items()
        }

    def to_dict(self) -> JSONDict:
        return {
            "projection_rows": self.to_projection_rows(),
            "diagnostics": list(self.diagnostics),
            "summary": dict(self.summary),
        }


def build_pdg_cdg_catalog_projection_rows(
    rows: PDGPublicationWriteRows | Mapping[str, Iterable[Mapping[str, Any]]],
) -> PDGCDGCatalogProjectionRows:
    """Build deterministic JSON-safe catalog projection rows for PDG CDGs.

    The projection intentionally does not write to production catalog storage.
    It accepts either ``PDGPublicationWriteRows`` or a table-to-rows mapping and
    emits flattened rows that are useful for catalog views/search while storage
    wiring is still pending.
    """

    rows_by_table = (
        rows.to_insert_rows() if isinstance(rows, PDGPublicationWriteRows) else rows
    )
    diagnostics: list[JSONDict] = []
    artifacts = _catalog_projection_table_rows(rows_by_table, "artifacts", diagnostics)
    versions = _catalog_projection_table_rows(
        rows_by_table, "artifact_versions", diagnostics
    )
    nodes = _catalog_projection_table_rows(
        rows_by_table, "artifact_cdg_nodes", diagnostics
    )
    edges = _catalog_projection_table_rows(
        rows_by_table, "artifact_cdg_edges", diagnostics
    )
    bindings = _catalog_projection_table_rows(
        rows_by_table, "artifact_cdg_bindings", diagnostics
    )
    relationships = _catalog_projection_table_rows(
        rows_by_table, "artifact_relationships", diagnostics
    )

    artifact_by_id = _catalog_artifacts_by_id(artifacts, diagnostics)
    version_by_id = _catalog_versions_by_id(versions, diagnostics)
    nodes_by_version = _rows_by_text_key(nodes, "version_id")
    edges_by_version = _rows_by_text_key(edges, "version_id")
    bindings_by_version = _rows_by_text_key(bindings, "version_id")
    cdg_version_ids = sorted(
        set(nodes_by_version) | set(edges_by_version) | set(bindings_by_version)
    )

    for version_id in cdg_version_ids:
        version = version_by_id.get(version_id)
        if version is None:
            diagnostics.append(
                _catalog_projection_diagnostic(
                    reason="missing_version_envelope",
                    table="artifact_versions",
                    detail={"version_id": version_id},
                )
            )
            diagnostics.append(
                _catalog_projection_diagnostic(
                    reason="orphan_cdg_row",
                    table="artifact_cdg_nodes",
                    detail={
                        "version_id": version_id,
                        "node_count": len(nodes_by_version.get(version_id, ())),
                        "edge_count": len(edges_by_version.get(version_id, ())),
                        "binding_count": len(bindings_by_version.get(version_id, ())),
                    },
                )
            )
            continue
        artifact_id = _row_text(version, "artifact_id")
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            diagnostics.append(
                _catalog_projection_diagnostic(
                    reason="missing_artifact_envelope",
                    table="artifacts",
                    detail={"version_id": version_id, "artifact_id": artifact_id},
                )
            )
            diagnostics.append(
                _catalog_projection_diagnostic(
                    reason="orphan_cdg_row",
                    table="artifact_cdg_nodes",
                    detail={"version_id": version_id, "artifact_id": artifact_id},
                )
            )

    projection_rows: dict[str, list[JSONDict]] = {
        "catalog_cdg_artifacts": [],
        "catalog_cdg_versions": [],
        "catalog_cdg_nodes": [],
        "catalog_cdg_relationships": [],
        "catalog_symbolic_artifacts": [],
    }

    catalogable_count = 0
    relationship_rows = [row for _, row in relationships]
    for version_id in cdg_version_ids:
        version = version_by_id.get(version_id)
        if version is None:
            continue
        artifact_id = _row_text(version, "artifact_id")
        artifact = artifact_by_id.get(artifact_id)
        version_nodes = tuple(row for _, row in nodes_by_version.get(version_id, ()))
        if artifact is None or not _row_text(artifact, "fqdn") or not version_nodes:
            continue
        version_edges = tuple(row for _, row in edges_by_version.get(version_id, ()))
        version_bindings = tuple(
            row for _, row in bindings_by_version.get(version_id, ())
        )
        topology_hash = _projection_topology_hash(
            artifact=artifact,
            version_nodes=version_nodes,
            version_edges=version_edges,
        )
        version_relationships = _catalog_relationship_projection_rows(
            artifact=artifact,
            version=version,
            cdg_edges=version_edges,
            relationship_rows=relationship_rows,
            topology_hash=topology_hash,
        )
        operation_kinds = _sorted_text_values(
            _row_text(node, "matched_primitive") for node in version_nodes
        )
        relationship_kinds = _sorted_text_values(
            _row_text(row, "relationship_kind") for row in version_relationships
        )
        expression_ids = _projection_expression_ids(
            version_nodes=version_nodes,
            relationship_rows=relationship_rows,
        )
        relationship_summaries = [
            _catalog_relationship_summary(row) for row in version_relationships
        ]
        replay_key_seed = {
            "artifact_id": artifact_id,
            "version_id": version_id,
            "topology_hash": topology_hash,
            "operation_kinds": operation_kinds,
            "relationship_kinds": relationship_kinds,
        }
        replay_hash = hashlib.sha256(
            _canonical_json(replay_key_seed).encode("utf-8")
        ).hexdigest()
        common = _catalog_projection_common_fields(
            artifact=artifact,
            version=version,
            topology_hash=topology_hash,
            replay_hash=replay_hash,
        )

        projection_rows["catalog_cdg_artifacts"].append(
            {
                **common,
                "projection_kind": "pdg_cdg_catalog_artifact.v1",
                "node_count": len(version_nodes),
                "edge_count": len(version_edges),
                "expression_node_count": len(expression_ids),
                "relationship_count": len(version_relationships),
                "operation_kinds": operation_kinds,
                "relationship_kinds": relationship_kinds,
                "operation_kind_counts": _count_mapping_values(
                    version_nodes, "matched_primitive"
                ),
                "relationship_kind_counts": _count_mapping_values(
                    version_relationships, "relationship_kind"
                ),
                "provenance": _catalog_projection_provenance(
                    artifact=artifact,
                    version=version,
                    replay_hash=replay_hash,
                ),
            }
        )
        projection_rows["catalog_cdg_versions"].append(
            {
                **common,
                "projection_kind": "pdg_cdg_catalog_version.v1",
                "semver": _row_text(version, "semver"),
                "is_latest": _row_bool(version, "is_latest"),
                "content_hash": _row_text(version, "content_hash"),
                "fingerprint": _row_text(version, "fingerprint"),
                "node_count": len(version_nodes),
                "edge_count": len(version_edges),
                "binding_count": len(version_bindings),
                "expression_node_count": len(expression_ids),
                "operation_summary": _count_mapping_values(
                    version_nodes, "matched_primitive"
                ),
                "relationship_summary": _count_mapping_values(
                    version_relationships, "relationship_kind"
                ),
            }
        )
        projection_rows["catalog_cdg_nodes"].extend(
            _catalog_node_projection_rows(
                artifact=artifact,
                version=version,
                nodes=version_nodes,
                topology_hash=topology_hash,
                replay_hash=replay_hash,
            )
        )
        projection_rows["catalog_cdg_relationships"].extend(version_relationships)
        projection_rows["catalog_symbolic_artifacts"].append(
            {
                **common,
                "projection_kind": "pdg_cdg_catalog_symbolic_artifact.v1",
                "expression_id": "",
                "expression_kind": "cdg_derivation_graph",
                "raw_formula": _catalog_raw_formula(
                    operation_kinds=operation_kinds,
                    relationship_kinds=relationship_kinds,
                ),
                "dimensional_hash": "",
                "dim_signatures": [],
                "mechanism_tags": operation_kinds,
                "behavioral_archetypes": [
                    "computational_derivation_graph",
                    "pdg_derivation_chain",
                ],
                "relationships": relationship_summaries,
                "relationship_kinds": relationship_kinds,
                "operation_kinds": operation_kinds,
                "node_count": len(version_nodes),
                "edge_count": len(version_edges),
                "expression_node_count": len(expression_ids),
                "relationship_count": len(version_relationships),
                "operation_summary": _count_mapping_values(
                    version_nodes, "matched_primitive"
                ),
                "relationship_summary": _count_mapping_values(
                    version_relationships, "relationship_kind"
                ),
                "symbolic_variables": [],
                "validity_bounds": [],
                "source_domains": _catalog_source_domains(artifact),
                "known_analogues": expression_ids,
                "data_artifact_dependencies": [
                    _row_text(row, "bound_artifact_fqdn")
                    for row in version_bindings
                    if _row_text(row, "bound_artifact_fqdn")
                ],
                "review_status": _row_text(artifact, "review_status"),
                "validation_status": _row_text(artifact, "validation_status"),
                "publish_status": _row_text(artifact, "status"),
                "candidate_status": "catalog_projection",
                "trust_readiness": _catalog_trust_readiness(artifact),
                "is_publishable": _row_bool(artifact, "is_publishable"),
                "provenance": _catalog_projection_provenance(
                    artifact=artifact,
                    version=version,
                    replay_hash=replay_hash,
                ),
            }
        )
        catalogable_count += 1

    if catalogable_count == 0:
        diagnostics.append(
            _catalog_projection_diagnostic(
                reason="no_catalogable_cdgs",
                table="catalog_symbolic_artifacts",
                severity="warning",
                detail={
                    "cdg_version_count": len(cdg_version_ids),
                    "artifact_envelope_count": len(artifact_by_id),
                    "version_envelope_count": len(version_by_id),
                },
            )
        )

    return PDGCDGCatalogProjectionRows(
        projection_rows_by_table={
            table: tuple(_json_safe_value(row) for row in table_rows)
            for table, table_rows in projection_rows.items()
            if table_rows
        },
        diagnostics=tuple(_json_safe_value(row) for row in diagnostics),
    )


def validate_pdg_cdg_publication_graph(
    rows: PDGPublicationWriteRows | Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[JSONDict, ...]:
    """Validate publication CDG graph rows without mutating or writing them.

    The validator accepts either ``PDGPublicationWriteRows`` or a table-to-rows
    mapping shaped like ``to_insert_rows()``. It returns deterministic,
    JSON-serializable diagnostics and treats ordinary malformed row mappings as
    diagnostics instead of exceptions.
    """

    rows_by_table = (
        rows.to_insert_rows() if isinstance(rows, PDGPublicationWriteRows) else rows
    )
    diagnostics: list[JSONDict] = []
    nodes = _cdg_table_rows(rows_by_table, "artifact_cdg_nodes", diagnostics)
    edges = _cdg_table_rows(rows_by_table, "artifact_cdg_edges", diagnostics)
    bindings = _cdg_table_rows(rows_by_table, "artifact_cdg_bindings", diagnostics)
    artifacts = _cdg_table_rows(rows_by_table, "artifacts", diagnostics)
    artifact_versions = _cdg_table_rows(rows_by_table, "artifact_versions", diagnostics)

    node_keys: set[tuple[str, str]] = set()
    seen_node_keys: dict[tuple[str, str], int] = {}
    for row_index, row in nodes:
        version_id = _row_text(row, "version_id")
        node_id = _row_text(row, "node_id")
        _require_cdg_fields(
            row,
            row_index=row_index,
            table="artifact_cdg_nodes",
            fields=("version_id", "node_id"),
            diagnostics=diagnostics,
        )
        if not version_id or not node_id:
            continue
        key = (version_id, node_id)
        _record_duplicate_key(
            key,
            seen=seen_node_keys,
            row_index=row_index,
            table="artifact_cdg_nodes",
            reason="duplicate_node_key",
            diagnostics=diagnostics,
        )
        node_keys.add(key)

    seen_edge_keys: dict[tuple[str, str, str, str, str], int] = {}
    for row_index, row in edges:
        _require_cdg_fields(
            row,
            row_index=row_index,
            table="artifact_cdg_edges",
            fields=("version_id", "source_id", "target_id"),
            diagnostics=diagnostics,
        )
        version_id = _row_text(row, "version_id")
        source_id = _row_text(row, "source_id")
        target_id = _row_text(row, "target_id")
        if version_id and source_id and target_id:
            key = (
                version_id,
                source_id,
                target_id,
                _row_text(row, "output_name"),
                _row_text(row, "input_name"),
            )
            _record_duplicate_key(
                key,
                seen=seen_edge_keys,
                row_index=row_index,
                table="artifact_cdg_edges",
                reason="duplicate_edge_key",
                diagnostics=diagnostics,
            )
            missing = [
                endpoint
                for endpoint in (source_id, target_id)
                if (version_id, endpoint) not in node_keys
            ]
            if missing:
                diagnostics.append(
                    _cdg_validation_diagnostic(
                        table="artifact_cdg_edges",
                        reason="edge_endpoint_node_missing",
                        detail={
                            "row_index": row_index,
                            "version_id": version_id,
                            "source_id": source_id,
                            "target_id": target_id,
                            "missing_node_ids": missing,
                        },
                    )
                )

    seen_binding_keys: dict[tuple[str, str, str], int] = {}
    for row_index, row in bindings:
        _require_cdg_fields(
            row,
            row_index=row_index,
            table="artifact_cdg_bindings",
            fields=("version_id", "node_id"),
            diagnostics=diagnostics,
        )
        version_id = _row_text(row, "version_id")
        node_id = _row_text(row, "node_id")
        if version_id and node_id:
            key = (version_id, node_id, _row_text(row, "bound_artifact_fqdn"))
            _record_duplicate_key(
                key,
                seen=seen_binding_keys,
                row_index=row_index,
                table="artifact_cdg_bindings",
                reason="duplicate_binding_key",
                diagnostics=diagnostics,
            )
            if (version_id, node_id) not in node_keys:
                diagnostics.append(
                    _cdg_validation_diagnostic(
                        table="artifact_cdg_bindings",
                        reason="binding_node_missing",
                        detail={
                            "row_index": row_index,
                            "version_id": version_id,
                            "node_id": node_id,
                        },
                    )
            )

    _validate_cdg_artifact_envelope_rows(
        artifacts=artifacts,
        artifact_versions=artifact_versions,
        diagnostics=diagnostics,
    )

    known_version_ids = {
        _row_text(row, "version_id")
        for _, row in artifact_versions
        if _row_text(row, "version_id")
    }
    if known_version_ids:
        for table, table_rows in (
            ("artifact_cdg_nodes", nodes),
            ("artifact_cdg_edges", edges),
            ("artifact_cdg_bindings", bindings),
        ):
            for row_index, row in table_rows:
                version_id = _row_text(row, "version_id")
                if version_id and version_id not in known_version_ids:
                    diagnostics.append(
                        _cdg_validation_diagnostic(
                            table=table,
                            reason="orphan_cdg_version",
                            detail={
                                "row_index": row_index,
                                "version_id": version_id,
                            },
                        )
                    )

    return tuple(diagnostics)


@dataclass(frozen=True)
class PDGCDGArtifactEnvelope:
    """Opt-in artifact registry metadata for CDG candidate publication."""

    fqdn_prefix: str = ""
    semver: str = "0.1.0"
    namespace_root: str = ""
    namespace_path: str = ""
    owner_id: str = ""
    source_repo_id: str = ""
    source_package: str = ""
    source_module_path: str = ""
    source_symbol_prefix: str = "pdg_cdg_candidate"
    status: str = "draft"
    visibility_tier: str = "internal"
    description: str = "PDG candidate derivation CDG manifest"
    source_kind: str = "generated"
    is_publishable: bool = False
    is_latest: bool = False
    s3_key_prefix: str = ""

    @classmethod
    def from_value(
        cls,
        value: "PDGCDGArtifactEnvelope | Mapping[str, Any]",
    ) -> "PDGCDGArtifactEnvelope":
        if isinstance(value, cls):
            return value
        allowed = cls.__dataclass_fields__
        kwargs = {key: raw for key, raw in value.items() if key in allowed}
        return cls(**kwargs)


def build_pdg_relationship_ingest(
    bundle: PDGIngestBundle,
    *,
    expression_bindings_by_pdg_node_id: Mapping[str, str | Mapping[str, Any]],
    chain_edge_ids: Iterable[str] | None = None,
) -> PDGRelationshipIngestResult:
    """Build validated relationship rows and CDG candidate manifests.

    ``expression_bindings_by_pdg_node_id`` is the only required publication
    input. Values may be plain ``expression_id`` strings or dictionaries with
    ``expression_id``, ``label``, ``artifact_id``, ``version_id``, and
    ``metadata`` keys. Edges with missing endpoint bindings are skipped and
    reported; no database calls are made.
    """

    bindings = _normalize_bindings(expression_bindings_by_pdg_node_id)
    relationship_rows: list[ArtifactRelationshipRow] = []
    manifest_edges: list[PDGInferenceEdge] = []
    skipped_edges: list[JSONDict] = []
    wanted_edge_ids = set(chain_edge_ids or ())

    for edge in bundle.inference_edges:
        if wanted_edge_ids and edge.edge_id not in wanted_edge_ids:
            continue
        source_binding = bindings.get(edge.target_node_id)
        target_binding = bindings.get(edge.source_node_id)
        if source_binding is None or target_binding is None:
            skipped_edges.append(
                {
                    "pdg_edge_id": edge.edge_id,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "reason": "missing_expression_binding",
                    "missing_node_ids": [
                        node_id
                        for node_id, binding in (
                            (edge.source_node_id, target_binding),
                            (edge.target_node_id, source_binding),
                        )
                        if binding is None
                    ],
                }
            )
            continue

        row = edge.to_relationship_hint().to_artifact_relationship_row(
            expression_id_by_pdg_node_id={
                edge.source_node_id: target_binding.expression_id,
                edge.target_node_id: source_binding.expression_id,
            }
        )
        relationship_rows.append(validate_artifact_relationship_row(row))
        if _edge_is_chain_candidate(edge):
            manifest_edges.append(edge)

    manifests = _build_cdg_candidate_manifests(
        manifest_edges,
        bindings=bindings,
        labels={equation.node_id: equation.label for equation in bundle.equations},
    )
    return PDGRelationshipIngestResult(
        artifact_relationship_rows=tuple(relationship_rows),
        cdg_candidate_manifests=tuple(manifests),
        skipped_edges=tuple(skipped_edges),
    )


def build_pdg_publication_write_rows(
    result: PDGRelationshipIngestResult,
    *,
    cdg_artifact_envelope: PDGCDGArtifactEnvelope | Mapping[str, Any] | None = None,
) -> PDGPublicationWriteRows:
    """Adapt PDG relationship/CDG results into write-plan-ready rows.

    The adapter is intentionally narrow and side-effect free. CDG node and edge
    rows are deterministic for each candidate manifest. Binding rows are emitted
    only when expression refs carry reusable artifact binding metadata
    (``bound_artifact_fqdn`` and ``bound_version_content_hash``), otherwise a
    skipped diagnostic records the unsupported publication gap.
    """

    envelope = (
        PDGCDGArtifactEnvelope.from_value(cdg_artifact_envelope)
        if cdg_artifact_envelope is not None
        else None
    )
    rows: dict[str, list[JSONDict]] = {
        "artifact_relationships": result.relationship_insert_rows(),
    }
    diagnostics: list[JSONDict] = []

    for manifest in result.cdg_candidate_manifests:
        if envelope is not None:
            artifact_row, version_row = _candidate_manifest_artifact_rows(
                manifest,
                envelope=envelope,
            )
            rows.setdefault("artifacts", []).append(artifact_row)
            rows.setdefault("artifact_versions", []).append(version_row)
        cdg_rows, cdg_diagnostics = _candidate_manifest_to_cdg_rows(manifest)
        diagnostics.extend(cdg_diagnostics)
        for table, table_rows in cdg_rows.items():
            rows.setdefault(table, []).extend(table_rows)

    return PDGPublicationWriteRows(
        insert_rows_by_table={
            table: tuple(table_rows)
            for table, table_rows in rows.items()
            if table_rows
        },
        diagnostics=tuple(diagnostics),
    )


def _candidate_manifest_artifact_rows(
    manifest: Mapping[str, Any],
    *,
    envelope: PDGCDGArtifactEnvelope,
) -> tuple[JSONDict, JSONDict]:
    manifest_id = str(manifest.get("manifest_id") or _stable_id("pdg_cdg_candidate", manifest))
    artifact_id = _manifest_artifact_id(manifest)
    version_id = _manifest_version_id(manifest)
    fqdn = _manifest_artifact_fqdn(manifest, envelope=envelope)
    content_hash = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    source_symbol = _manifest_source_symbol(manifest_id, envelope=envelope)

    artifact_row: JSONDict = {
        "artifact_id": artifact_id,
        "artifact_kind": "cdg",
        "fqdn": fqdn,
        "namespace_root": envelope.namespace_root,
        "namespace_path": envelope.namespace_path,
        "source_package": envelope.source_package,
        "source_module_path": envelope.source_module_path,
        "source_symbol": source_symbol,
        "status": envelope.status,
        "visibility_tier": envelope.visibility_tier,
        "description": envelope.description,
        "source_kind": envelope.source_kind,
        "is_stochastic": False,
        "is_ffi": False,
        "is_publishable": envelope.is_publishable,
        "topo_hash": str(manifest.get("topology_hash") or ""),
        "top_level_input_arity": _manifest_top_level_input_arity(manifest),
        "top_level_output_arity": 1,
        "leaf_count": len(_manifest_rows(manifest, "nodes")),
        "verified_leaf_coverage": 0.0,
    }
    if envelope.owner_id:
        artifact_row["owner_id"] = envelope.owner_id
    if envelope.source_repo_id:
        artifact_row["source_repo_id"] = envelope.source_repo_id

    version_row: JSONDict = {
        "version_id": version_id,
        "artifact_id": artifact_id,
        "content_hash": content_hash,
        "semver": _manifest_semver(manifest, envelope=envelope),
        "is_latest": envelope.is_latest,
        "s3_key": _manifest_s3_key(fqdn, envelope=envelope),
        "fingerprint": content_hash,
    }
    return artifact_row, version_row


def _manifest_artifact_id(manifest: Mapping[str, Any]) -> str:
    explicit = str(manifest.get("artifact_id") or "")
    if explicit:
        return explicit
    manifest_id = str(manifest.get("manifest_id") or _canonical_json(manifest))
    return str(uuid5(_CDG_ARTIFACT_NAMESPACE, manifest_id))


def _manifest_artifact_fqdn(
    manifest: Mapping[str, Any],
    *,
    envelope: PDGCDGArtifactEnvelope,
) -> str:
    explicit = str(manifest.get("artifact_fqdn") or "")
    if explicit:
        return explicit
    manifest_id = str(manifest.get("manifest_id") or "")
    if not envelope.fqdn_prefix:
        raise ValueError(
            "cdg_artifact_envelope.fqdn_prefix is required when manifests do not "
            "carry artifact_fqdn"
        )
    return f"{envelope.fqdn_prefix}.{_fqdn_token(manifest_id)}"


def _fqdn_token(value: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in value.lower())
    token = "_".join(part for part in token.split("_") if part)
    return token or "manifest"


def _manifest_source_symbol(
    manifest_id: str,
    *,
    envelope: PDGCDGArtifactEnvelope,
) -> str:
    if not envelope.source_symbol_prefix:
        return ""
    return f"{envelope.source_symbol_prefix}_{_fqdn_token(manifest_id)}"


def _manifest_s3_key(fqdn: str, *, envelope: PDGCDGArtifactEnvelope) -> str:
    if not envelope.s3_key_prefix:
        return ""
    return f"{envelope.s3_key_prefix.rstrip('/')}/{fqdn}.json"


def _manifest_semver(
    manifest: Mapping[str, Any],
    *,
    envelope: PDGCDGArtifactEnvelope,
) -> str:
    semver = str(manifest.get("semver") or envelope.semver)
    if not semver:
        raise ValueError("cdg_artifact_envelope.semver is required")
    return semver


def _manifest_top_level_input_arity(manifest: Mapping[str, Any]) -> int:
    refs = manifest.get("referenced_expressions")
    if isinstance(refs, (list, tuple)):
        return len(refs)
    return 0


def _normalize_bindings(
    values: Mapping[str, str | Mapping[str, Any]]
) -> dict[str, PDGExpressionBinding]:
    bindings: dict[str, PDGExpressionBinding] = {}
    for pdg_node_id, value in values.items():
        binding = PDGExpressionBinding.from_value(pdg_node_id, value)
        if binding.expression_id:
            bindings[pdg_node_id] = binding
    return bindings


def _build_cdg_candidate_manifests(
    edges: Iterable[PDGInferenceEdge],
    *,
    bindings: Mapping[str, PDGExpressionBinding],
    labels: Mapping[str, str],
) -> list[JSONDict]:
    selected_edges = list(edges)
    if not selected_edges:
        return []

    nodes: list[JSONDict] = []
    cdg_edges: list[JSONDict] = []
    output_to_step_id: dict[str, str] = {}
    referenced_node_ids: set[str] = set()

    for index, edge in enumerate(selected_edges, start=1):
        step_id = f"pdg_step_{index}"
        operation_kind = _manifest_operation_kind(edge)
        input_binding = bindings[edge.source_node_id]
        output_binding = bindings[edge.target_node_id]
        referenced_node_ids.update((edge.source_node_id, edge.target_node_id))
        node = {
            "node_id": step_id,
            "operation_kind": operation_kind,
            "label": edge.inference_rule_label or edge.inference_rule_id,
            "input_expressions": [input_binding.to_manifest_ref()],
            "output_expression": output_binding.to_manifest_ref(),
            "pdg_edge_id": edge.edge_id,
            "source_pdg_inference_id": edge.edge_id,
            "relationship_kind": edge.relationship_kind,
            "inference_rule_id": edge.inference_rule_id,
            "assumptions": list(edge.assumptions),
            "binding_metadata": dict(edge.binding_metadata),
            "variable_bindings": _edge_variable_bindings(edge),
            "dimensions": _edge_dimensions(edge),
        }
        if labels.get(edge.source_node_id) or labels.get(edge.target_node_id):
            node["equation_labels"] = {
                "input": labels.get(edge.source_node_id, ""),
                "output": labels.get(edge.target_node_id, ""),
            }
        nodes.append(node)

        upstream_step_id = output_to_step_id.get(edge.source_node_id)
        if upstream_step_id is not None:
            cdg_edges.append(
                {
                    "source_id": upstream_step_id,
                    "target_id": step_id,
                    "edge_kind": "symbolic_equation_flow",
                    "pdg_node_id": edge.source_node_id,
                    "expression_id": input_binding.expression_id,
                }
            )
        output_to_step_id[edge.target_node_id] = step_id

    manifest_seed = {
        "edge_ids": [edge.edge_id for edge in selected_edges],
        "expression_ids": [
            bindings[node_id].expression_id for node_id in sorted(referenced_node_ids)
        ],
    }
    return [
        {
            "manifest_id": _stable_id("pdg_cdg_candidate", manifest_seed),
            "manifest_kind": "pdg_derivation_chain_candidate",
            "source_system": PDG_SOURCE_SYSTEM,
            "scaffold_version": PHASE4_SCAFFOLD_VERSION,
            "nodes": nodes,
            "edges": cdg_edges,
            "referenced_expressions": [
                bindings[node_id].to_manifest_ref() for node_id in sorted(referenced_node_ids)
            ],
            "metadata": {
                "relationship_edge_ids": [edge.edge_id for edge in selected_edges],
                "source_pdg_inference_coverage": [
                    {
                        "pdg_edge_id": edge.edge_id,
                        "source_node_id": edge.source_node_id,
                        "target_node_id": edge.target_node_id,
                        "variable_bindings": _edge_variable_bindings(edge),
                        "dimensions": _edge_dimensions(edge),
                        "assumptions": list(edge.assumptions),
                    }
                    for edge in selected_edges
                ],
                "candidate_scope": "algebraic_rearrangement_derivation_chain",
            },
        }
    ]


def _candidate_manifest_to_cdg_rows(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, list[JSONDict]], list[JSONDict]]:
    manifest_id = str(manifest.get("manifest_id") or "")
    version_id = _manifest_version_id(manifest)
    rows: dict[str, list[JSONDict]] = {
        "artifact_cdg_nodes": [],
        "artifact_cdg_edges": [],
        "artifact_cdg_bindings": [],
    }
    diagnostics: list[JSONDict] = []

    for node in _manifest_rows(manifest, "nodes"):
        node_id = str(node.get("node_id") or "")
        if not node_id:
            diagnostics.append(
                _cdg_diagnostic(
                    table="artifact_cdg_nodes",
                    reason="missing_node_id",
                    manifest_id=manifest_id,
                    severity="error",
                )
            )
            continue

        rows["artifact_cdg_nodes"].append(
            {
                "version_id": version_id,
                "node_id": node_id,
                "parent_node_id": str(node.get("parent_node_id") or ""),
                "name": str(node.get("label") or node_id),
                "description": _node_description(node),
                "concept_type": "pdg_derivation_step",
                "status": "candidate",
                "type_signature": _node_type_signature(node),
                "matched_primitive": str(node.get("operation_kind") or ""),
            }
        )
        binding_rows, binding_diagnostics = _node_binding_rows(
            node,
            version_id=version_id,
            manifest_id=manifest_id,
        )
        rows["artifact_cdg_bindings"].extend(binding_rows)
        diagnostics.extend(binding_diagnostics)

    for edge in _manifest_rows(manifest, "edges"):
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if not source_id or not target_id:
            diagnostics.append(
                _cdg_diagnostic(
                    table="artifact_cdg_edges",
                    reason="missing_edge_endpoint",
                    manifest_id=manifest_id,
                    severity="error",
                    detail=_canonical_json(edge),
                )
            )
            continue
        rows["artifact_cdg_edges"].append(
            {
                "version_id": version_id,
                "source_id": source_id,
                "target_id": target_id,
                "output_name": str(edge.get("output_name") or edge.get("expression_id") or "output"),
                "input_name": str(edge.get("input_name") or "input"),
            }
        )

    return rows, diagnostics


def _manifest_version_id(manifest: Mapping[str, Any]) -> str:
    explicit = str(manifest.get("version_id") or "")
    if explicit:
        return explicit
    manifest_id = str(manifest.get("manifest_id") or _canonical_json(manifest))
    return str(uuid5(_CDG_VERSION_NAMESPACE, manifest_id))


def _manifest_rows(manifest: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in manifest.get(key, ()) if isinstance(row, Mapping))


def _node_description(node: Mapping[str, Any]) -> str:
    return (
        f"PDG {node.get('relationship_kind', 'derivation')} step"
        f" from edge {node.get('pdg_edge_id', '')}"
    ).strip()


def _node_type_signature(node: Mapping[str, Any]) -> str:
    inputs = [
        str(ref.get("expression_id") or "")
        for ref in _expression_refs(node.get("input_expressions"))
    ]
    output = _expression_ref(node.get("output_expression"))
    return _canonical_json(
        {
            "inputs": [value for value in inputs if value],
            "output": "" if output is None else str(output.get("expression_id") or ""),
            "operation_kind": str(node.get("operation_kind") or ""),
            "source_pdg_inference_id": str(
                node.get("source_pdg_inference_id") or node.get("pdg_edge_id") or ""
            ),
            "inference_rule_id": str(node.get("inference_rule_id") or ""),
            "relationship_kind": str(node.get("relationship_kind") or ""),
            "assumptions": _text_list(node.get("assumptions")),
            "variable_bindings": _json_mapping(node.get("variable_bindings")),
            "dimensions": _json_mapping(node.get("dimensions")),
        }
    )


def _edge_variable_bindings(edge: PDGInferenceEdge) -> JSONDict:
    bindings = dict(edge.binding_metadata)
    explicit = bindings.get("variable_bindings")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    variables = bindings.get("variables")
    if isinstance(variables, Mapping):
        return dict(variables)
    dimensions = bindings.get("dimensions")
    return {
        key: value
        for key, value in bindings.items()
        if key not in {"dimensions", "dimension_bindings"} and value is not dimensions
    }


def _edge_dimensions(edge: PDGInferenceEdge) -> JSONDict:
    for key in ("dimensions", "dimension_bindings"):
        value = edge.binding_metadata.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    value = edge.raw_payload.get("dimensions")
    return dict(value) if isinstance(value, Mapping) else {}


def _json_mapping(value: Any) -> JSONDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _build_relationship_ingest_summary(
    result: PDGRelationshipIngestResult,
) -> JSONDict:
    relationship_rows = result.relationship_insert_rows()
    skipped_edges = list(result.skipped_edges)
    return _json_safe_mapping(
        {
            "summary_kind": "pdg_relationship_ingest_summary.v1",
            "source_system": PDG_SOURCE_SYSTEM,
            "scaffold_version": PHASE4_SCAFFOLD_VERSION,
            "relationship_row_count": len(relationship_rows),
            "skipped_edge_count": len(skipped_edges),
            "skipped_edge_reasons": _count_mapping_values(skipped_edges, "reason"),
            "cdg_candidate_manifest_count": len(result.cdg_candidate_manifests),
            "operation_kind_counts": _count_values(
                _relationship_row_operation_kind(row) for row in relationship_rows
            ),
            "relationship_kind_counts": _count_mapping_values(
                relationship_rows, "relationship_kind"
            ),
        }
    )


def _build_publication_write_rows_summary(
    publication_rows: PDGPublicationWriteRows,
) -> JSONDict:
    insert_rows = publication_rows.to_insert_rows()
    table_row_counts = {
        table: len(rows) for table, rows in sorted(insert_rows.items())
    }
    artifact_envelope_row_counts = {
        "artifacts": table_row_counts.get("artifacts", 0),
        "artifact_versions": table_row_counts.get("artifact_versions", 0),
    }
    diagnostics = list(publication_rows.diagnostics)
    return _json_safe_mapping(
        {
            "summary_kind": "pdg_cdg_publication_rows_summary.v1",
            "source_system": PDG_SOURCE_SYSTEM,
            "scaffold_version": PHASE4_SCAFFOLD_VERSION,
            "table_row_counts": table_row_counts,
            "artifact_relationship_row_count": table_row_counts.get(
                "artifact_relationships", 0
            ),
            "cdg_candidate_manifest_count": _count_distinct_cdg_versions(insert_rows),
            "operation_kind_counts": _count_mapping_values(
                insert_rows.get("artifact_cdg_nodes", ()), "matched_primitive"
            ),
            "relationship_kind_counts": _count_mapping_values(
                insert_rows.get("artifact_relationships", ()), "relationship_kind"
            ),
            "diagnostic_count": len(diagnostics),
            "diagnostics_by_severity": _count_mapping_values(
                diagnostics, "severity"
            ),
            "diagnostics_by_reason": _count_mapping_values(diagnostics, "reason"),
            "diagnostics_by_table": _count_mapping_values(diagnostics, "table"),
            "artifact_envelope_row_counts": artifact_envelope_row_counts,
            "artifact_envelope_total_row_count": sum(
                artifact_envelope_row_counts.values()
            ),
        }
    )


def _build_catalog_projection_summary(
    projection_rows: PDGCDGCatalogProjectionRows,
) -> JSONDict:
    rows = projection_rows.to_projection_rows()
    diagnostics = list(projection_rows.diagnostics)
    symbolic_rows = rows.get("catalog_symbolic_artifacts", ())
    relationship_rows = rows.get("catalog_cdg_relationships", ())
    node_rows = rows.get("catalog_cdg_nodes", ())
    return _json_safe_mapping(
        {
            "summary_kind": "pdg_cdg_catalog_projection_summary.v1",
            "source_system": PDG_SOURCE_SYSTEM,
            "scaffold_version": PHASE4_SCAFFOLD_VERSION,
            "table_row_counts": {
                table: len(table_rows) for table, table_rows in sorted(rows.items())
            },
            "projected_row_count": sum(len(table_rows) for table_rows in rows.values()),
            "catalogable_cdg_count": len(symbolic_rows),
            "diagnostic_count": len(diagnostics),
            "diagnostics_by_severity": _count_mapping_values(
                diagnostics, "severity"
            ),
            "diagnostics_by_reason": _count_mapping_values(diagnostics, "reason"),
            "diagnostics_by_table": _count_mapping_values(diagnostics, "table"),
            "source_systems": _sorted_text_values(
                _row_text(row, "source_system") for row in symbolic_rows
            ),
            "operation_kinds": _sorted_text_values(
                kind
                for row in symbolic_rows
                for kind in _sequence_texts(row.get("operation_kinds"))
            ),
            "operation_kind_counts": _count_mapping_values(
                node_rows, "operation_kind"
            ),
            "relationship_kinds": _sorted_text_values(
                kind
                for row in symbolic_rows
                for kind in _sequence_texts(row.get("relationship_kinds"))
            ),
            "relationship_kind_counts": _count_mapping_values(
                relationship_rows, "relationship_kind"
            ),
        }
    )


def _relationship_row_operation_kind(row: Mapping[str, Any]) -> str:
    evidence = row.get("evidence_json")
    if isinstance(evidence, Mapping):
        return "" if evidence.get("operation_kind") is None else str(
            evidence.get("operation_kind")
        )
    return ""


def _count_mapping_values(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    return _count_values(_mapping_text(row, key) for row in rows)


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_value in values:
        value = "" if raw_value is None else str(raw_value)
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _mapping_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def _count_distinct_cdg_versions(
    insert_rows: Mapping[str, Iterable[Mapping[str, Any]]],
) -> int:
    version_ids = {
        _mapping_text(row, "version_id")
        for row in insert_rows.get("artifact_cdg_nodes", ())
        if _mapping_text(row, "version_id")
    }
    if version_ids:
        return len(version_ids)
    return len(insert_rows.get("artifact_versions", ()))


def _json_safe_mapping(value: Mapping[str, Any]) -> JSONDict:
    return json.loads(_canonical_json(value))


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(raw_value)
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return str(value)


def _sequence_texts(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if str(item))
    if value in (None, ""):
        return ()
    return (str(value),)


def _catalog_projection_table_rows(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    table: str,
    diagnostics: list[JSONDict],
) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    raw_rows = rows_by_table.get(table, ())
    if raw_rows is None:
        return ()
    normalized: list[tuple[int, Mapping[str, Any]]] = []
    try:
        iterator = enumerate(raw_rows)
    except TypeError:
        diagnostics.append(
            _catalog_projection_diagnostic(
                table=table,
                reason="malformed_table_rows",
                detail={"table": table, "row_type": type(raw_rows).__name__},
            )
        )
        return ()
    for row_index, row in iterator:
        if not isinstance(row, Mapping):
            diagnostics.append(
                _catalog_projection_diagnostic(
                    table=table,
                    reason="malformed_row",
                    detail={"row_index": row_index, "row_type": type(row).__name__},
                )
            )
            continue
        normalized.append((row_index, row))
    return tuple(normalized)


def _catalog_artifacts_by_id(
    artifacts: tuple[tuple[int, Mapping[str, Any]], ...],
    diagnostics: list[JSONDict],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row_index, row in artifacts:
        artifact_id = _row_text(row, "artifact_id")
        if not artifact_id:
            diagnostics.append(
                _catalog_projection_diagnostic(
                    table="artifacts",
                    reason="missing_artifact_id",
                    detail={"row_index": row_index},
                )
            )
            continue
        if _row_text(row, "artifact_kind") and _row_text(row, "artifact_kind") != "cdg":
            diagnostics.append(
                _catalog_projection_diagnostic(
                    table="artifacts",
                    reason="non_cdg_artifact_envelope",
                    detail={
                        "row_index": row_index,
                        "artifact_id": artifact_id,
                        "artifact_kind": _row_text(row, "artifact_kind"),
                    },
                )
            )
            continue
        result.setdefault(artifact_id, row)
    return result


def _catalog_versions_by_id(
    versions: tuple[tuple[int, Mapping[str, Any]], ...],
    diagnostics: list[JSONDict],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row_index, row in versions:
        version_id = _row_text(row, "version_id")
        if not version_id:
            diagnostics.append(
                _catalog_projection_diagnostic(
                    table="artifact_versions",
                    reason="missing_version_id",
                    detail={"row_index": row_index},
                )
            )
            continue
        result.setdefault(version_id, row)
    return result


def _rows_by_text_key(
    rows: tuple[tuple[int, Mapping[str, Any]], ...],
    key: str,
) -> dict[str, list[tuple[int, Mapping[str, Any]]]]:
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for row_index, row in rows:
        value = _row_text(row, key)
        if value:
            grouped.setdefault(value, []).append((row_index, row))
    return grouped


def _row_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _sorted_text_values(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _projection_topology_hash(
    *,
    artifact: Mapping[str, Any],
    version_nodes: tuple[Mapping[str, Any], ...],
    version_edges: tuple[Mapping[str, Any], ...],
) -> str:
    explicit = _row_text(artifact, "topology_hash") or _row_text(artifact, "topo_hash")
    if explicit:
        return explicit
    seed = {
        "nodes": [
            {
                "node_id": _row_text(row, "node_id"),
                "parent_node_id": _row_text(row, "parent_node_id"),
                "operation_kind": _row_text(row, "matched_primitive"),
            }
            for row in version_nodes
        ],
        "edges": [
            {
                "source_id": _row_text(row, "source_id"),
                "target_id": _row_text(row, "target_id"),
                "output_name": _row_text(row, "output_name"),
                "input_name": _row_text(row, "input_name"),
            }
            for row in version_edges
        ],
    }
    return hashlib.sha256(_canonical_json(seed).encode("utf-8")).hexdigest()[:16]


def _projection_expression_ids(
    *,
    version_nodes: tuple[Mapping[str, Any], ...],
    relationship_rows: list[Mapping[str, Any]],
) -> list[str]:
    values: list[str] = []
    for row in relationship_rows:
        values.extend(
            [
                _row_text(row, "source_expression_id"),
                _row_text(row, "target_expression_id"),
            ]
        )
    for row in version_nodes:
        signature = _parsed_json_mapping(row.get("type_signature"))
        values.extend(_sequence_texts(signature.get("inputs")))
        output = _row_text(signature, "output")
        if output:
            values.append(output)
    return _sorted_text_values(values)


def _parsed_json_mapping(value: Any) -> JSONDict:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _catalog_projection_common_fields(
    *,
    artifact: Mapping[str, Any],
    version: Mapping[str, Any],
    topology_hash: str,
    replay_hash: str,
) -> JSONDict:
    return {
        "artifact_id": _row_text(artifact, "artifact_id"),
        "version_id": _row_text(version, "version_id"),
        "fqdn": _row_text(artifact, "fqdn"),
        "artifact_kind": "cdg",
        "source_system": PDG_SOURCE_SYSTEM,
        "source_kind": _row_text(artifact, "source_kind") or "generated",
        "namespace_root": _row_text(artifact, "namespace_root"),
        "namespace_path": _row_text(artifact, "namespace_path"),
        "source_package": _row_text(artifact, "source_package"),
        "source_module_path": _row_text(artifact, "source_module_path"),
        "source_symbol": _row_text(artifact, "source_symbol"),
        "topology_hash": topology_hash,
        "topo_hash": topology_hash,
        "content_hash": _row_text(version, "content_hash"),
        "fingerprint": _row_text(version, "fingerprint"),
        "replay_key": f"pdg-cdg-catalog-projection:{replay_hash}",
    }


def _catalog_node_projection_rows(
    *,
    artifact: Mapping[str, Any],
    version: Mapping[str, Any],
    nodes: tuple[Mapping[str, Any], ...],
    topology_hash: str,
    replay_hash: str,
) -> list[JSONDict]:
    common = _catalog_projection_common_fields(
        artifact=artifact,
        version=version,
        topology_hash=topology_hash,
        replay_hash=replay_hash,
    )
    rows: list[JSONDict] = []
    for ordinal, node in enumerate(
        sorted(nodes, key=lambda row: _row_text(row, "node_id")), start=1
    ):
        signature = _parsed_json_mapping(node.get("type_signature"))
        rows.append(
            {
                **common,
                "projection_kind": "pdg_cdg_catalog_node.v1",
                "node_id": _row_text(node, "node_id"),
                "parent_node_id": _row_text(node, "parent_node_id"),
                "node_ordinal": ordinal,
                "name": _row_text(node, "name"),
                "description": _row_text(node, "description"),
                "concept_type": _row_text(node, "concept_type"),
                "status": _row_text(node, "status"),
                "operation_kind": _row_text(node, "matched_primitive"),
                "matched_primitive": _row_text(node, "matched_primitive"),
                "relationship_kind": _row_text(signature, "relationship_kind"),
                "inference_rule_id": _row_text(signature, "inference_rule_id"),
                "source_pdg_inference_id": _row_text(
                    signature, "source_pdg_inference_id"
                ),
                "input_expression_ids": list(_sequence_texts(signature.get("inputs"))),
                "output_expression_id": _row_text(signature, "output"),
                "type_signature": _row_text(node, "type_signature"),
                "provenance": _catalog_projection_provenance(
                    artifact=artifact,
                    version=version,
                    replay_hash=replay_hash,
                ),
            }
        )
    return rows


def _catalog_relationship_projection_rows(
    *,
    artifact: Mapping[str, Any],
    version: Mapping[str, Any],
    cdg_edges: tuple[Mapping[str, Any], ...],
    relationship_rows: list[Mapping[str, Any]],
    topology_hash: str,
) -> list[JSONDict]:
    replay_seed = {
        "artifact_id": _row_text(artifact, "artifact_id"),
        "version_id": _row_text(version, "version_id"),
        "relationship_count": len(relationship_rows),
        "edge_count": len(cdg_edges),
    }
    replay_hash = hashlib.sha256(
        _canonical_json(replay_seed).encode("utf-8")
    ).hexdigest()
    common = _catalog_projection_common_fields(
        artifact=artifact,
        version=version,
        topology_hash=topology_hash,
        replay_hash=replay_hash,
    )
    rows: list[JSONDict] = []
    for edge in sorted(
        cdg_edges,
        key=lambda row: (
            _row_text(row, "source_id"),
            _row_text(row, "target_id"),
            _row_text(row, "output_name"),
            _row_text(row, "input_name"),
        ),
    ):
        row_seed = {
            "version_id": _row_text(version, "version_id"),
            "source_id": _row_text(edge, "source_id"),
            "target_id": _row_text(edge, "target_id"),
            "output_name": _row_text(edge, "output_name"),
            "input_name": _row_text(edge, "input_name"),
        }
        rows.append(
            {
                **common,
                "projection_kind": "pdg_cdg_catalog_relationship.v1",
                "relationship_id": _stable_id("catalog_cdg_edge", row_seed),
                "relationship_kind": "cdg_data_flow",
                "relationship_label": "CDG data flow",
                "source_node_id": _row_text(edge, "source_id"),
                "target_node_id": _row_text(edge, "target_id"),
                "output_name": _row_text(edge, "output_name"),
                "input_name": _row_text(edge, "input_name"),
                "verified": False,
                "confidence": 0.0,
            }
        )
    for relationship in sorted(
        relationship_rows,
        key=lambda row: (
            _row_text(row, "relationship_kind"),
            _row_text(row, "source_expression_id"),
            _row_text(row, "target_expression_id"),
        ),
    ):
        evidence = relationship.get("evidence_json")
        evidence_mapping = dict(evidence) if isinstance(evidence, Mapping) else {}
        rows.append(
            {
                **common,
                "projection_kind": "pdg_cdg_catalog_relationship.v1",
                "relationship_id": _row_text(relationship, "relationship_id")
                or _stable_id("catalog_relationship", _json_safe_value(relationship)),
                "relationship_kind": _row_text(relationship, "relationship_kind"),
                "relationship_label": _row_text(relationship, "relationship_kind"),
                "operation_kind": _row_text(evidence_mapping, "operation_kind"),
                "source_expression_id": _row_text(relationship, "source_expression_id"),
                "target_expression_id": _row_text(relationship, "target_expression_id"),
                "source_kind": _row_text(relationship, "source_kind")
                or PDG_SOURCE_SYSTEM,
                "verified": _row_bool(relationship, "verified"),
                "confidence": _row_float(relationship, "confidence"),
            }
        )
    return rows


def _catalog_relationship_summary(row: Mapping[str, Any]) -> JSONDict:
    return {
        "relationship_kind": _row_text(row, "relationship_kind"),
        "relationship_label": _row_text(row, "relationship_label"),
        "confidence": _row_float(row, "confidence"),
        "verified": _row_bool(row, "verified"),
        "source_kind": _row_text(row, "source_kind") or PDG_SOURCE_SYSTEM,
    }


def _catalog_raw_formula(
    *,
    operation_kinds: list[str],
    relationship_kinds: list[str],
) -> str:
    pieces = []
    if operation_kinds:
        pieces.append("operations=" + ",".join(operation_kinds))
    if relationship_kinds:
        pieces.append("relationships=" + ",".join(relationship_kinds))
    return "; ".join(pieces)


def _catalog_source_domains(artifact: Mapping[str, Any]) -> list[str]:
    return _sorted_text_values(
        (
            "physics",
            _row_text(artifact, "namespace_root"),
            _row_text(artifact, "namespace_path"),
            _row_text(artifact, "source_package"),
        )
    )


def _catalog_trust_readiness(artifact: Mapping[str, Any]) -> str:
    explicit = _row_text(artifact, "trust_readiness")
    if explicit:
        return explicit
    return "not_ready" if not _row_bool(artifact, "is_publishable") else "ready"


def _catalog_projection_provenance(
    *,
    artifact: Mapping[str, Any],
    version: Mapping[str, Any],
    replay_hash: str,
) -> JSONDict:
    return {
        "source_system": PDG_SOURCE_SYSTEM,
        "scaffold_version": PHASE4_SCAFFOLD_VERSION,
        "artifact_id": _row_text(artifact, "artifact_id"),
        "version_id": _row_text(version, "version_id"),
        "content_hash": _row_text(version, "content_hash"),
        "fingerprint": _row_text(version, "fingerprint"),
        "replay_key": f"pdg-cdg-catalog-projection:{replay_hash}",
    }


def _catalog_projection_diagnostic(
    *,
    table: str,
    reason: str,
    detail: Mapping[str, Any],
    severity: str = "warning",
) -> JSONDict:
    return {
        "stage": "pdg_cdg_catalog_projection",
        "table": table,
        "reason": reason,
        "severity": severity,
        "artifact_key": "",
        "atom_name": PDG_SOURCE_SYSTEM,
        "detail": _canonical_json(_json_safe_value(detail)),
    }


def _node_binding_rows(
    node: Mapping[str, Any],
    *,
    version_id: str,
    manifest_id: str,
) -> tuple[list[JSONDict], list[JSONDict]]:
    rows: list[JSONDict] = []
    diagnostics: list[JSONDict] = []
    seen: set[tuple[str, str]] = set()
    node_id = str(node.get("node_id") or "")
    refs = [
        ("input", ref)
        for ref in _expression_refs(node.get("input_expressions"))
    ]
    output_ref = _expression_ref(node.get("output_expression"))
    if output_ref is not None:
        refs.append(("output", output_ref))

    for role, ref in refs:
        bound_fqdn = _ref_text(ref, "bound_artifact_fqdn")
        content_hash = _ref_text(ref, "bound_version_content_hash")
        if not bound_fqdn or not content_hash:
            diagnostics.append(
                _cdg_diagnostic(
                    table="artifact_cdg_bindings",
                    reason="missing_cdg_binding_artifact_metadata",
                    manifest_id=manifest_id,
                    detail=_canonical_json(
                        {
                            "node_id": node_id,
                            "role": role,
                            "expression_id": str(ref.get("expression_id") or ""),
                            "required": [
                                "bound_artifact_fqdn",
                                "bound_version_content_hash",
                            ],
                        }
                    ),
                )
            )
            continue
        key = (node_id, bound_fqdn)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "version_id": version_id,
                "node_id": node_id,
                "bound_artifact_fqdn": bound_fqdn,
                "bound_version_content_hash": content_hash,
                "binding_confidence": _ref_number(ref, "binding_confidence", 0.0),
                "binding_source": _ref_text(ref, "binding_source")
                or f"pdg_candidate_manifest:{role}",
            }
        )
    return rows, diagnostics


def _expression_refs(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(ref for ref in value if isinstance(ref, Mapping))


def _expression_ref(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _ref_text(ref: Mapping[str, Any], key: str) -> str:
    value = ref.get(key)
    if value in (None, "") and isinstance(ref.get("metadata"), Mapping):
        value = ref["metadata"].get(key)
    return "" if value is None else str(value)


def _ref_number(ref: Mapping[str, Any], key: str, default: float) -> float:
    value = ref.get(key)
    if value in (None, "") and isinstance(ref.get("metadata"), Mapping):
        value = ref["metadata"].get(key)
    if value in (None, ""):
        return default
    return float(value)


def _cdg_table_rows(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    table: str,
    diagnostics: list[JSONDict],
) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    raw_rows = rows_by_table.get(table, ())
    if raw_rows is None:
        return ()
    normalized: list[tuple[int, Mapping[str, Any]]] = []
    try:
        iterator = enumerate(raw_rows)
    except TypeError:
        diagnostics.append(
            _cdg_validation_diagnostic(
                table=table,
                reason="malformed_table_rows",
                detail={"table": table, "row_type": type(raw_rows).__name__},
            )
        )
        return ()
    for row_index, row in iterator:
        if not isinstance(row, Mapping):
            diagnostics.append(
                _cdg_validation_diagnostic(
                    table=table,
                    reason="malformed_row",
                    detail={"row_index": row_index, "row_type": type(row).__name__},
                )
            )
            continue
        normalized.append((row_index, row))
    return tuple(normalized)


def _row_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def _validate_cdg_artifact_envelope_rows(
    *,
    artifacts: tuple[tuple[int, Mapping[str, Any]], ...],
    artifact_versions: tuple[tuple[int, Mapping[str, Any]], ...],
    diagnostics: list[JSONDict],
) -> None:
    seen_artifact_ids: dict[tuple[str, ...], int] = {}
    artifact_ids: set[str] = set()
    for row_index, row in artifacts:
        _require_cdg_fields(
            row,
            row_index=row_index,
            table="artifacts",
            fields=("artifact_id", "artifact_kind", "fqdn"),
            diagnostics=diagnostics,
        )
        artifact_id = _row_text(row, "artifact_id")
        if artifact_id:
            _record_duplicate_key(
                (artifact_id,),
                seen=seen_artifact_ids,
                row_index=row_index,
                table="artifacts",
                reason="duplicate_artifact_id",
                diagnostics=diagnostics,
            )
            artifact_ids.add(artifact_id)
        artifact_kind = _row_text(row, "artifact_kind")
        if artifact_kind and artifact_kind != "cdg":
            diagnostics.append(
                _cdg_validation_diagnostic(
                    table="artifacts",
                    reason="invalid_artifact_kind",
                    detail={
                        "row_index": row_index,
                        "artifact_id": artifact_id,
                        "artifact_kind": artifact_kind,
                        "expected_artifact_kind": "cdg",
                    },
                )
            )

    seen_version_ids: dict[tuple[str, ...], int] = {}
    version_artifact_ids: set[str] = set()
    for row_index, row in artifact_versions:
        _require_cdg_fields(
            row,
            row_index=row_index,
            table="artifact_versions",
            fields=("version_id", "artifact_id"),
            diagnostics=diagnostics,
        )
        version_id = _row_text(row, "version_id")
        artifact_id = _row_text(row, "artifact_id")
        if version_id:
            _record_duplicate_key(
                (version_id,),
                seen=seen_version_ids,
                row_index=row_index,
                table="artifact_versions",
                reason="duplicate_version_id",
                diagnostics=diagnostics,
            )
        if not artifact_id:
            continue
        version_artifact_ids.add(artifact_id)
        if artifacts and artifact_id not in artifact_ids:
            diagnostics.append(
                _cdg_validation_diagnostic(
                    table="artifact_versions",
                    reason="artifact_version_artifact_missing",
                    detail={
                        "row_index": row_index,
                        "version_id": version_id,
                        "artifact_id": artifact_id,
                    },
                )
            )

    if artifact_versions:
        for row_index, row in artifacts:
            artifact_id = _row_text(row, "artifact_id")
            if artifact_id and artifact_id not in version_artifact_ids:
                diagnostics.append(
                    _cdg_validation_diagnostic(
                        table="artifacts",
                        reason="artifact_version_missing",
                        detail={
                            "row_index": row_index,
                            "artifact_id": artifact_id,
                        },
                    )
                )


def _require_cdg_fields(
    row: Mapping[str, Any],
    *,
    row_index: int,
    table: str,
    fields: Iterable[str],
    diagnostics: list[JSONDict],
) -> None:
    missing = [field for field in fields if not _row_text(row, field)]
    if not missing:
        return
    diagnostics.append(
        _cdg_validation_diagnostic(
            table=table,
            reason="missing_cdg_row_identity",
            detail={"row_index": row_index, "missing_fields": missing},
        )
    )


def _record_duplicate_key(
    key: tuple[str, ...],
    *,
    seen: dict[tuple[str, ...], int],
    row_index: int,
    table: str,
    reason: str,
    diagnostics: list[JSONDict],
) -> None:
    first_index = seen.get(key)
    if first_index is None:
        seen[key] = row_index
        return
    diagnostics.append(
        _cdg_validation_diagnostic(
            table=table,
            reason=reason,
            detail={
                "row_index": row_index,
                "first_row_index": first_index,
                "key": list(key),
            },
        )
    )


def _cdg_validation_diagnostic(
    *,
    table: str,
    reason: str,
    detail: Mapping[str, Any],
    severity: str = "error",
) -> JSONDict:
    return _cdg_diagnostic(
        table=table,
        reason=reason,
        manifest_id="",
        severity=severity,
        detail=_canonical_json(detail),
    )


def _cdg_diagnostic(
    *,
    table: str,
    reason: str,
    manifest_id: str,
    severity: str = "skipped",
    detail: str = "",
) -> JSONDict:
    return {
        "stage": "pdg_cdg_publication",
        "table": table,
        "reason": reason,
        "severity": severity,
        "artifact_key": manifest_id,
        "atom_name": PDG_SOURCE_SYSTEM,
        "detail": detail,
    }


__all__ = [
    "PDGCDGCatalogProjectionRows",
    "PDGExpressionBinding",
    "PDGCDGArtifactEnvelope",
    "PDGPublicationWriteRows",
    "PDGRelationshipIngestResult",
    "PHASE4_SCAFFOLD_VERSION",
    "build_pdg_cdg_catalog_projection_rows",
    "build_pdg_publication_write_rows",
    "build_pdg_relationship_ingest",
    "validate_pdg_cdg_publication_graph",
]
