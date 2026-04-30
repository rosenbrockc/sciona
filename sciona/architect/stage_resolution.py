"""Policy resolution for non-atom CDG stages.

Generic data plumbing should not be forced through primitive retrieval.  This
module classifies those stages into explicit action classes so the Architect can
mark them resolved without asking Hunter for an atom match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from sciona.architect.models import AlgorithmicNode, ConceptType


class StageActionClass(str, Enum):
    """Binding-level action classes understood by solution CDG grounding."""

    REPLACE_STAGE = "replace_stage"
    ORCHESTRATION = "orchestration"
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    TRIVIAL_INLINE = "trivial_inline"


NON_ATOM_ACTION_CLASSES: frozenset[str] = frozenset(
    {
        StageActionClass.ORCHESTRATION.value,
        StageActionClass.EXTERNAL_KNOWLEDGE.value,
        StageActionClass.TRIVIAL_INLINE.value,
    }
)


@dataclass(frozen=True)
class StageResolution:
    """Resolved handling for a stage that should not become an atom."""

    action_class: StageActionClass
    reason: str
    confidence: float = 0.9


_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ATOM_OVERRIDES = {
    "difference_features",
    "target_scaling",
    "streaming_imputation",
    "label_smoothing",
    "dicom_windowing",
    "temporal_unrolling",
    "feature_aggregation",
}

_CONCAT_TERMS = {
    "concat",
    "concatenate",
    "concatenation",
    "stack",
    "stacking",
    "fusion",
    "fuse",
    "merge",
    "merging",
    "union",
    "join",
    "joining",
    "align",
    "alignment",
    "group",
    "grouping",
}

_CONCAT_CONTEXT = {
    "feature",
    "features",
    "embedding",
    "embeddings",
    "candidate",
    "candidates",
    "modality",
    "modalities",
    "view",
    "views",
    "asset",
    "assets",
    "channel",
    "channels",
    "upstream",
    "timestamp",
    "patient",
}

_EXTERNAL_TERMS = {
    "bson",
    "csv",
    "parquet",
    "hdf5",
    "h5ad",
    "gnss",
    "a3d",
    "cryo",
    "supplement",
    "supplements",
    "load",
    "loading",
    "ingest",
    "ingestion",
    "parse",
    "parsing",
    "reader",
    "read",
    "file",
    "files",
    "format",
    "conversion",
    "logs",
    "streaming",
}

_TRIVIAL_TERMS = {
    "reshape",
    "reshaping",
    "transpose",
    "astype",
    "cast",
    "downcast",
    "dtype",
    "float16",
    "float32",
    "float64",
    "uint8",
    "tensor",
    "formatting",
    "array",
    "orientation",
    "coordinate",
    "coordinates",
    "normalize",
    "normalization",
    "season",
    "average",
    "groupby",
    "window",
}


def is_non_atom_action_class(action_class: str | None) -> bool:
    """Return whether an action class is resolved without atom retrieval."""

    return str(action_class or "").strip() in NON_ATOM_ACTION_CLASSES


def is_non_atom_resolved_node(node: AlgorithmicNode) -> bool:
    """Return whether a node has already been resolved by policy."""

    if is_non_atom_action_class(getattr(node, "action_class", "")):
        return True
    return str(getattr(node, "primitive_binding_source", "") or "").startswith(
        "stage_resolution:"
    )


def classify_generic_data_op(node: AlgorithmicNode) -> StageResolution | None:
    """Classify generic CDG plumbing that should bypass atom retrieval."""

    text = " ".join(
        [
            node.node_id,
            node.name,
            node.description,
            node.concept_type.value,
            " ".join(port.name for port in node.inputs + node.outputs),
            " ".join(port.type_desc for port in node.inputs + node.outputs),
        ]
    ).lower()
    normalized_id = node.node_id.strip().lower()
    if normalized_id in _ATOM_OVERRIDES:
        return None

    tokens = set(_TOKEN_RE.findall(text))

    if tokens & _EXTERNAL_TERMS and (
        node.concept_type == ConceptType.EXTERNAL_KNOWLEDGE
        or tokens & {"bson", "csv", "parquet", "hdf5", "h5ad", "gnss", "a3d", "cryo"}
        or {"data", "ingestion"} <= tokens
        or {"format", "conversion"} <= tokens
    ):
        return StageResolution(
            action_class=StageActionClass.EXTERNAL_KNOWLEDGE,
            reason="File-format parsing, data loading, or external dataset integration.",
        )

    if tokens & _CONCAT_TERMS and (
        tokens & _CONCAT_CONTEXT
        or node.concept_type in {ConceptType.DATA_ASSEMBLY, ConceptType.MAP_OVER}
    ):
        return StageResolution(
            action_class=StageActionClass.ORCHESTRATION,
            reason="Combines, aligns, groups, or routes upstream artifacts as pipeline wiring.",
        )

    if "entity_splitting" in normalized_id or (
        {"split", "entity"} <= tokens and node.concept_type == ConceptType.MAP_OVER
    ):
        return StageResolution(
            action_class=StageActionClass.ORCHESTRATION,
            reason="Entity splitting is a MAP_OVER orchestration pattern.",
        )

    if tokens & _TRIVIAL_TERMS and (
        node.concept_type in {ConceptType.DATA_ASSEMBLY, ConceptType.DATA_EXTRACTION}
        or tokens & {"reshape", "astype", "cast", "downcast", "uint8", "float16"}
        or "formatting" in normalized_id
        or "normalization" in normalized_id
    ):
        return StageResolution(
            action_class=StageActionClass.TRIVIAL_INLINE,
            reason="Simple array/dataframe reshaping, dtype conversion, or coordinate formatting.",
            confidence=0.85,
        )

    return None
