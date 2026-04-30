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
            "relationship_kind": edge.relationship_kind,
            "inference_rule_id": edge.inference_rule_id,
            "assumptions": list(edge.assumptions),
            "binding_metadata": dict(edge.binding_metadata),
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
                "candidate_scope": "algebraic_rearrangement_derivation_chain",
            },
        }
    ]


__all__ = [
    "PDGExpressionBinding",
    "PDGRelationshipIngestResult",
    "PHASE4_SCAFFOLD_VERSION",
    "build_pdg_relationship_ingest",
]
