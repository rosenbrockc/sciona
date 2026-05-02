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
        }


@dataclass(frozen=True)
class PDGPublicationWriteRows:
    """PDG relationship/CDG rows ready for publication write planning."""

    insert_rows_by_table: Mapping[str, tuple[JSONDict, ...]]
    diagnostics: tuple[JSONDict, ...] = ()

    def to_insert_rows(self) -> dict[str, list[JSONDict]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.insert_rows_by_table.items()
        }


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
    "PDGExpressionBinding",
    "PDGCDGArtifactEnvelope",
    "PDGPublicationWriteRows",
    "PDGRelationshipIngestResult",
    "PHASE4_SCAFFOLD_VERSION",
    "build_pdg_publication_write_rows",
    "build_pdg_relationship_ingest",
    "validate_pdg_cdg_publication_graph",
]
