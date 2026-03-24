"""Passive enrichment proposal models for node-level architecture work.

Phase 1 only introduces a common representation for primitive, template, and
skeleton proposals. These models are metadata carriers and must not change live
selection behavior yet.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from sciona.architect.graph_retrieval import ExampleChild
from sciona.architect.models import AlgorithmicPrimitive
from sciona.architect.template_retriever import TemplateMatch


class ProposalType(str, Enum):
    """Kinds of node-enrichment proposals."""

    PRIMITIVE = "primitive"
    TEMPLATE = "template"
    SKELETON = "skeleton"


def _infer_source_family(label: str, *, fallback: str = "") -> str:
    """Infer a stable family label from a primitive/template identifier."""
    text = str(label or "").strip()
    if not text:
        return str(fallback or "").strip()
    if text.startswith("ageoa."):
        parts = text.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return text
    if "." in text:
        return text.split(".", 1)[0]
    return str(fallback or text).strip()


def _families_from_example_children(children: list[ExampleChild]) -> set[str]:
    """Infer distinct source families represented inside a template example."""
    families: set[str] = set()
    for child in children:
        family = _infer_source_family(child.matched_primitive)
        if family:
            families.add(family)
    return families


class EnrichmentProposal(BaseModel):
    """Passive representation of an enrichment candidate for a CDG node."""

    proposal_type: ProposalType
    source_family: str = ""
    source_label: str = ""
    confidence: float = 0.0
    compatibility_score: float = 0.0
    delta_nodes: int = 0
    delta_edges: int = 0
    delta_family_count: int = 0
    delta_concept_type_count: int = 0
    matched_primitive: str | None = None
    template_fqn: str | None = None
    skeleton_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "EnrichmentProposal":
        numeric_fields = (
            self.delta_nodes,
            self.delta_edges,
            self.delta_family_count,
            self.delta_concept_type_count,
        )
        if any(value < 0 for value in numeric_fields):
            raise ValueError("complexity deltas must be non-negative")

        if self.proposal_type == ProposalType.PRIMITIVE:
            if self.template_fqn is not None:
                raise ValueError("primitive proposals must not set template_fqn")
            if self.skeleton_name is not None:
                raise ValueError("primitive proposals must not set skeleton_name")
            if not str(self.matched_primitive or "").strip():
                raise ValueError("primitive proposals must set matched_primitive")

        if self.proposal_type == ProposalType.TEMPLATE:
            if self.skeleton_name is not None:
                raise ValueError("template proposals must not set skeleton_name")
            if not str(self.template_fqn or "").strip():
                raise ValueError("template proposals must set template_fqn")

        if self.proposal_type == ProposalType.SKELETON:
            if self.template_fqn is not None:
                raise ValueError("skeleton proposals must not set template_fqn")
            if not str(self.skeleton_name or "").strip():
                raise ValueError("skeleton proposals must set skeleton_name")

        return self


def proposal_from_primitive(
    primitive: AlgorithmicPrimitive,
    *,
    confidence: float = 0.0,
    compatibility_score: float = 0.0,
) -> EnrichmentProposal:
    """Build a passive proposal from a primitive candidate."""
    family = _infer_source_family(primitive.name, fallback=primitive.source)
    return EnrichmentProposal(
        proposal_type=ProposalType.PRIMITIVE,
        source_family=family,
        source_label=primitive.name,
        confidence=float(confidence),
        compatibility_score=float(compatibility_score),
        delta_nodes=0,
        delta_edges=0,
        delta_family_count=0,
        delta_concept_type_count=0,
        matched_primitive=primitive.name,
        payload={
            "source": primitive.source,
            "category": primitive.category.value,
        },
    )


def proposal_from_template_match(match: TemplateMatch) -> EnrichmentProposal:
    """Build a passive proposal from a retrieved template match."""
    example = match.example
    children = list(example.children or [])
    families = _families_from_example_children(children)
    concept_types = {
        str(child.concept_type or "").strip()
        for child in children
        if str(child.concept_type or "").strip()
    }
    source_family = _infer_source_family(example.fqn, fallback=example.repo)
    return EnrichmentProposal(
        proposal_type=ProposalType.TEMPLATE,
        source_family=source_family,
        source_label=example.fqn or example.repo or match.source,
        confidence=float(match.confidence),
        compatibility_score=float(match.alignment.total),
        delta_nodes=len(children),
        delta_edges=len(example.edges or []),
        delta_family_count=len(families),
        delta_concept_type_count=len(concept_types),
        template_fqn=example.fqn,
        payload={
            "repo": example.repo,
            "concept_type": example.concept_type,
            "retrieval_layer": example.retrieval_layer,
            "match_source": match.source,
        },
    )


def proposal_placeholder_skeleton(
    *,
    skeleton_name: str,
    source_family: str,
    source_label: str = "",
    confidence: float = 0.0,
    compatibility_score: float = 0.0,
    delta_nodes: int = 0,
    delta_edges: int = 0,
    delta_family_count: int = 0,
    delta_concept_type_count: int = 0,
    payload: dict[str, Any] | None = None,
) -> EnrichmentProposal:
    """Create a representable skeleton proposal without enabling live use yet."""
    return EnrichmentProposal(
        proposal_type=ProposalType.SKELETON,
        source_family=source_family,
        source_label=source_label or skeleton_name,
        confidence=float(confidence),
        compatibility_score=float(compatibility_score),
        delta_nodes=int(delta_nodes),
        delta_edges=int(delta_edges),
        delta_family_count=int(delta_family_count),
        delta_concept_type_count=int(delta_concept_type_count),
        skeleton_name=skeleton_name,
        payload=dict(payload or {}),
    )
