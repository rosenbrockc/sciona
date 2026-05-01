"""Physics Derivation Graph ingestion scaffold.

The Physics Derivation Graph (PDG) is a source of equation nodes and
inference-rule edges. This module intentionally does not fetch PDG content.
It normalizes already retrieved payloads into Wave 0-compatible row shapes:

* ``physics_ingest_snapshots`` for immutable source snapshots.
* ``physics_equation_candidates`` for raw equation candidates.
* ``artifact_relationships`` once equation candidates have been published as
  symbolic artifact expressions.

PDG edges are premise-to-conclusion in source payloads. Wave 0
``artifact_relationships`` are modeled as "source derives from target", so a
PDG edge ``premise -> conclusion`` becomes a relationship row whose source is
the conclusion expression and whose target is the premise expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


JSONDict = dict[str, Any]

PDG_SOURCE_SYSTEM = "physics_derivation_graph"
PDG_ADAPTER_NAME = "sciona.physics_ingest.sources.pdg"
PDG_ADAPTER_VERSION = "wave1.pdg_scaffold.v1"

_FORMULA_FORMATS = {
    "",
    "latex",
    "mathml",
    "content_mathml",
    "wikidata_math",
    "asciimath",
    "sympy",
    "plain_text",
}

_DERIVATION_OPERATIONS = {
    "solve",
    "solve_for",
    "substitute",
    "substitution",
    "limit",
    "take_limit",
    "nondimensionalize",
    "nondimensionalization",
    "non_dimensionalize",
    "non_dimensionalization",
    "approximate",
    "approximation",
    "simplify",
    "differentiate",
    "integrate",
}

_CDG_CHAIN_OPERATIONS = {
    "solve",
    "solve_for",
    "substitute",
    "substitution",
    "limit",
    "take_limit",
    "nondimensionalize",
    "approximate",
}

_OPERATION_ALIASES = {
    "approximation": "approximate",
    "nondimensionalization": "nondimensionalize",
    "non_dimensionalize": "nondimensionalize",
    "non_dimensionalization": "nondimensionalize",
}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif isinstance(value, (int, float)):
            return str(value)
    return ""


def _first_list(row: Mapping[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [str(item) for item in value if str(item)]
    return []


def _compact_rule_id(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return normalized or "inference"


def _formula_format(row: Mapping[str, Any]) -> str:
    explicit = _first_text(row, "formula_format", "format", "raw_formula_format")
    if explicit and explicit in _FORMULA_FORMATS:
        return explicit
    if _first_text(row, "latex", "tex"):
        return "latex"
    if _first_text(row, "mathml", "content_mathml"):
        return "mathml"
    if _first_text(row, "sympy"):
        return "sympy"
    return "plain_text" if _first_text(row, "formula", "equation", "expression") else ""


def _formula_text(row: Mapping[str, Any]) -> str:
    return _first_text(
        row,
        "formula",
        "equation",
        "expression",
        "latex",
        "tex",
        "mathml",
        "content_mathml",
        "sympy",
    )


def _payload_collection(payload: Mapping[str, Any], *keys: str) -> Sequence[Any]:
    for key in keys:
        values = _as_sequence(payload.get(key))
        if values:
            return values
    return ()


def _relationship_kind_for_rule(rule_id: str, rule_label: str) -> str:
    text = f"{rule_id} {rule_label}".lower()
    if "limit" in text:
        return "limit_case_of"
    if "approxim" in text:
        return "approximation_of"
    if "rearrang" in text or "solve" in text or "isolate" in text:
        return "algebraic_rearrangement_of"
    if "assumption" in text or "assume" in text:
        return "requires_assumption"
    return "derives_from"


def _operation_kind_for_rule(rule_id: str, rule_label: str) -> str:
    text = _compact_rule_id(f"{rule_id}_{rule_label}")
    for operation in sorted(_DERIVATION_OPERATIONS, key=len, reverse=True):
        if operation in text:
            return _OPERATION_ALIASES.get(operation, operation)
    return "derive"


@dataclass(frozen=True)
class PDGEquationNode:
    """Normalized PDG equation node."""

    node_id: str
    label: str = ""
    description: str = ""
    formula: str = ""
    formula_format: str = ""
    source_uri: str = ""
    mechanism_tags: tuple[str, ...] = ()
    behavioral_archetypes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PDGEquationNode":
        node_id = _first_text(payload, "id", "node_id", "equation_id", "uuid")
        formula = _formula_text(payload)
        if not node_id:
            stable = _first_text(payload, "uri", "url", "source_uri") or formula
            node_id = f"pdg:eq:{_sha256_text(stable)[:16]}"
        return cls(
            node_id=node_id,
            label=_first_text(payload, "label", "name", "title"),
            description=_first_text(payload, "description", "notes", "note"),
            formula=formula,
            formula_format=_formula_format(payload),
            source_uri=_first_text(payload, "uri", "url", "source_uri"),
            mechanism_tags=tuple(
                _first_list(payload, "mechanism_tags", "mechanisms", "physics_mechanisms")
            ),
            behavioral_archetypes=tuple(
                _first_list(payload, "behavioral_archetypes", "archetypes")
            ),
            assumptions=tuple(_first_list(payload, "assumptions", "conditions")),
            raw_payload=dict(payload),
        )

    def to_candidate_row(self, *, snapshot_id: str | None = None) -> JSONDict:
        row: JSONDict = {
            "source_candidate_id": self.node_id,
            "source_entity_uri": self.source_uri,
            "source_label": self.label,
            "source_description": self.description,
            "raw_formula": self.formula,
            "raw_formula_format": self.formula_format,
            "candidate_status": "raw_imported",
            "parse_confidence": 0.0,
            "priority_score": 0.0,
            "mechanism_tags": list(self.mechanism_tags),
            "behavioral_archetypes": list(self.behavioral_archetypes),
            "source_payload": dict(self.raw_payload),
            "notes": "Imported from Physics Derivation Graph; symbolic normalization pending.",
        }
        if snapshot_id is not None:
            row["snapshot_id"] = snapshot_id
        return row


@dataclass(frozen=True)
class PDGInferenceEdge:
    """Normalized PDG inference edge.

    ``source_node_id`` is the PDG premise/input equation and ``target_node_id``
    is the PDG conclusion/output equation.
    """

    edge_id: str
    source_node_id: str
    target_node_id: str
    inference_rule_id: str
    inference_rule_label: str = ""
    assumptions: tuple[str, ...] = ()
    binding_metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PDGInferenceEdge":
        source = _first_text(payload, "source", "source_id", "from", "premise", "input")
        target = _first_text(payload, "target", "target_id", "to", "conclusion", "output")
        rule_label = _first_text(payload, "rule", "rule_label", "inference_rule", "name")
        rule_id = _compact_rule_id(
            _first_text(payload, "rule_id", "inference_rule_id") or rule_label
        )
        edge_id = _first_text(payload, "id", "edge_id")
        if not edge_id:
            edge_id = f"pdg:edge:{_sha256_text(_canonical_json([source, target, rule_id, payload]))[:16]}"
        raw_confidence = payload.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        binding_metadata = _as_mapping(payload.get("bindings") or payload.get("binding_metadata"))
        evidence = _as_mapping(payload.get("evidence") or payload.get("references"))
        return cls(
            edge_id=edge_id,
            source_node_id=source,
            target_node_id=target,
            inference_rule_id=rule_id,
            inference_rule_label=rule_label,
            assumptions=tuple(_first_list(payload, "assumptions", "conditions")),
            binding_metadata=dict(binding_metadata),
            evidence=dict(evidence),
            confidence=confidence,
            raw_payload=dict(payload),
        )

    @property
    def relationship_kind(self) -> str:
        return _relationship_kind_for_rule(self.inference_rule_id, self.inference_rule_label)

    @property
    def operation_kind(self) -> str:
        return _operation_kind_for_rule(self.inference_rule_id, self.inference_rule_label)

    def to_relationship_hint(self) -> "PDGRelationshipHint":
        return PDGRelationshipHint(
            source_node_id=self.target_node_id,
            target_node_id=self.source_node_id,
            relationship_kind=self.relationship_kind,
            relationship_label=self.inference_rule_label,
            inference_rule_id=self.inference_rule_id,
            binding_metadata=dict(self.binding_metadata),
            assumptions_json={"assumptions": list(self.assumptions)},
            evidence_json={
                "source_system": PDG_SOURCE_SYSTEM,
                "pdg_edge_id": self.edge_id,
                "pdg_source_node_id": self.source_node_id,
                "pdg_target_node_id": self.target_node_id,
                "operation_kind": self.operation_kind,
                "evidence": dict(self.evidence),
            },
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class PDGRelationshipHint:
    """Relationship row staging model before artifact expression ids exist."""

    source_node_id: str
    target_node_id: str
    relationship_kind: str
    relationship_label: str = ""
    inference_rule_id: str = ""
    binding_metadata: Mapping[str, Any] = field(default_factory=dict)
    assumptions_json: Mapping[str, Any] = field(default_factory=dict)
    evidence_json: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_artifact_relationship_row(
        self, *, expression_id_by_pdg_node_id: Mapping[str, str]
    ) -> JSONDict:
        """Materialize a Wave 0 ``artifact_relationships`` insert row.

        Raises ``KeyError`` if either endpoint has not been published as an
        ``artifact_symbolic_expressions.expression_id`` yet. That fail-closed
        behavior prevents insertion of rows that violate Wave 0 relationship
        endpoint checks.
        """

        source_expression_id = expression_id_by_pdg_node_id[self.source_node_id]
        target_expression_id = expression_id_by_pdg_node_id[self.target_node_id]
        return {
            "source_expression_id": source_expression_id,
            "target_expression_id": target_expression_id,
            "relationship_kind": self.relationship_kind,
            "relationship_label": self.relationship_label,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "inference_rule_id": self.inference_rule_id,
            "binding_metadata": dict(self.binding_metadata),
            "assumptions_json": dict(self.assumptions_json),
            "evidence_json": dict(self.evidence_json),
            "confidence": self.confidence,
            "source_kind": PDG_SOURCE_SYSTEM,
            "verified": False,
        }


@dataclass(frozen=True)
class DerivationCDGNode:
    """One symbolic operation in a derivation-CDG sketch."""

    node_id: str
    operation_kind: str
    name: str
    input_equation_ids: tuple[str, ...]
    output_equation_id: str
    inference_rule_id: str
    assumptions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "node_id": self.node_id,
            "operation_kind": self.operation_kind,
            "name": self.name,
            "input_equation_ids": list(self.input_equation_ids),
            "output_equation_id": self.output_equation_id,
            "inference_rule_id": self.inference_rule_id,
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DerivationCDGEdge:
    """Data dependency between derivation operation nodes."""

    source_id: str
    target_id: str
    equation_id: str
    edge_kind: str = "symbolic_equation_flow"

    def to_dict(self) -> JSONDict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "equation_id": self.equation_id,
            "edge_kind": self.edge_kind,
        }


@dataclass(frozen=True)
class DerivationCDGSketch:
    """Lightweight derivation-CDG extraction sketch for symbolic chains."""

    nodes: tuple[DerivationCDGNode, ...]
    edges: tuple[DerivationCDGEdge, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PDGIngestBundle:
    """Normalized PDG ingest output."""

    snapshot_row: Mapping[str, Any]
    equations: tuple[PDGEquationNode, ...]
    inference_edges: tuple[PDGInferenceEdge, ...]

    @property
    def relationship_hints(self) -> tuple[PDGRelationshipHint, ...]:
        return tuple(edge.to_relationship_hint() for edge in self.inference_edges)

    def candidate_rows(self, *, snapshot_id: str | None = None) -> list[JSONDict]:
        return [equation.to_candidate_row(snapshot_id=snapshot_id) for equation in self.equations]

    def relationship_rows(
        self, *, expression_id_by_pdg_node_id: Mapping[str, str]
    ) -> list[JSONDict]:
        return [
            hint.to_artifact_relationship_row(
                expression_id_by_pdg_node_id=expression_id_by_pdg_node_id
            )
            for hint in self.relationship_hints
        ]


class PDGAdapter:
    """Adapter for already retrieved Physics Derivation Graph payloads."""

    def __init__(self, *, adapter_version: str = PDG_ADAPTER_VERSION) -> None:
        self.adapter_version = adapter_version

    def build_snapshot_row(
        self,
        payload: Mapping[str, Any],
        *,
        source_uri: str = "",
        source_version: str = "",
        retrieved_at: str | None = None,
        license_expression: str = "",
        provenance_summary: str = "",
    ) -> JSONDict:
        return {
            "source_system": PDG_SOURCE_SYSTEM,
            "source_version": source_version,
            "source_uri": source_uri,
            "retrieved_at": retrieved_at or _now_iso(),
            "adapter_name": PDG_ADAPTER_NAME,
            "adapter_version": self.adapter_version,
            "license_expression": license_expression,
            "provenance_summary": provenance_summary,
            "payload_sha256": _sha256_text(_canonical_json(payload)),
            "payload": dict(payload),
        }

    def parse_equation_nodes(self, payload: Mapping[str, Any]) -> tuple[PDGEquationNode, ...]:
        rows = _payload_collection(payload, "equations", "equation_nodes", "nodes")
        equations: list[PDGEquationNode] = []
        for row in rows:
            mapping = _as_mapping(row)
            if not mapping:
                continue
            node_type = _first_text(mapping, "type", "kind", "node_type").lower()
            if node_type and node_type not in {"equation", "symbolic_equation", "expression"}:
                continue
            equation = PDGEquationNode.from_payload(mapping)
            if equation.formula or equation.label:
                equations.append(equation)
        return tuple(equations)

    def parse_inference_edges(self, payload: Mapping[str, Any]) -> tuple[PDGInferenceEdge, ...]:
        rows = _payload_collection(payload, "inference_edges", "edges", "derivations")
        edges: list[PDGInferenceEdge] = []
        for row in rows:
            mapping = _as_mapping(row)
            if not mapping:
                continue
            edge = PDGInferenceEdge.from_payload(mapping)
            if edge.source_node_id and edge.target_node_id:
                edges.append(edge)
        return tuple(edges)

    def parse_document(
        self,
        payload: Mapping[str, Any],
        *,
        source_uri: str = "",
        source_version: str = "",
        retrieved_at: str | None = None,
        license_expression: str = "",
        provenance_summary: str = "",
    ) -> PDGIngestBundle:
        return PDGIngestBundle(
            snapshot_row=self.build_snapshot_row(
                payload,
                source_uri=source_uri,
                source_version=source_version,
                retrieved_at=retrieved_at,
                license_expression=license_expression,
                provenance_summary=provenance_summary,
            ),
            equations=self.parse_equation_nodes(payload),
            inference_edges=self.parse_inference_edges(payload),
        )


def parse_pdg_document(
    payload: Mapping[str, Any],
    *,
    source_uri: str = "",
    source_version: str = "",
    retrieved_at: str | None = None,
    license_expression: str = "",
    provenance_summary: str = "",
) -> PDGIngestBundle:
    """Parse an already retrieved PDG payload using the default adapter."""

    return PDGAdapter().parse_document(
        payload,
        source_uri=source_uri,
        source_version=source_version,
        retrieved_at=retrieved_at,
        license_expression=license_expression,
        provenance_summary=provenance_summary,
    )


def extract_derivation_cdg_sketch(
    edges: Iterable[PDGInferenceEdge],
    *,
    equation_labels: Mapping[str, str] | None = None,
    chain_edge_ids: Iterable[str] | None = None,
) -> DerivationCDGSketch:
    """Extract a lightweight derivation-CDG sketch from solve/substitute/limit edges.

    This is intentionally a sketch, not a production ``CDGExport``. It gives
    Phase 4 a typed, serializable shape for derivation chains before artifact
    ids, symbolic expressions, and executable nodes are all available.
    """

    wanted_edge_ids = set(chain_edge_ids or ())
    labels = equation_labels or {}
    selected_edges: list[PDGInferenceEdge] = []
    for edge in edges:
        if wanted_edge_ids and edge.edge_id not in wanted_edge_ids:
            continue
        if edge.operation_kind in _CDG_CHAIN_OPERATIONS:
            selected_edges.append(edge)

    nodes: list[DerivationCDGNode] = []
    output_to_operation: dict[str, str] = {}
    for idx, edge in enumerate(selected_edges):
        operation_kind = "solve" if edge.operation_kind == "solve_for" else edge.operation_kind
        operation_kind = "substitute" if operation_kind == "substitution" else operation_kind
        operation_kind = "limit" if operation_kind == "take_limit" else operation_kind
        node_id = f"pdg_derivation_step_{idx + 1}"
        source_label = labels.get(edge.source_node_id, edge.source_node_id)
        target_label = labels.get(edge.target_node_id, edge.target_node_id)
        nodes.append(
            DerivationCDGNode(
                node_id=node_id,
                operation_kind=operation_kind,
                name=f"{operation_kind}: {source_label} -> {target_label}",
                input_equation_ids=(edge.source_node_id,),
                output_equation_id=edge.target_node_id,
                inference_rule_id=edge.inference_rule_id,
                assumptions=edge.assumptions,
                metadata={
                    "pdg_edge_id": edge.edge_id,
                    "relationship_kind": edge.relationship_kind,
                },
            )
        )
        output_to_operation[edge.target_node_id] = node_id

    cdg_edges: list[DerivationCDGEdge] = []
    for node in nodes:
        for equation_id in node.input_equation_ids:
            source_operation = output_to_operation.get(equation_id)
            if source_operation is not None and source_operation != node.node_id:
                cdg_edges.append(
                    DerivationCDGEdge(
                        source_id=source_operation,
                        target_id=node.node_id,
                        equation_id=equation_id,
                    )
                )

    return DerivationCDGSketch(
        nodes=tuple(nodes),
        edges=tuple(cdg_edges),
        metadata={
            "source_system": PDG_SOURCE_SYSTEM,
            "sketch_kind": "solve_substitute_limit_chain",
            "extracted_at": _now_iso(),
        },
    )
