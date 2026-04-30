from sciona.architect.handoff import CDGExport, to_pdg_nodes, validate_handoff
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.architect.stage_resolution import (
    StageActionClass,
    classify_generic_data_op,
    is_non_atom_resolved_node,
)


def _node(
    node_id: str,
    description: str,
    concept_type: ConceptType = ConceptType.DATA_ASSEMBLY,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=node_id.replace("_", " "),
        description=description,
        concept_type=concept_type,
        inputs=[IOSpec(name="x", type_desc="np.ndarray")],
        outputs=[IOSpec(name="y", type_desc="np.ndarray")],
    )


def test_classifies_feature_concat_as_orchestration() -> None:
    resolution = classify_generic_data_op(
        _node("feature_concatenation", "Concatenate sequence and structure embeddings")
    )

    assert resolution is not None
    assert resolution.action_class == StageActionClass.ORCHESTRATION


def test_classifies_file_loading_as_external_knowledge() -> None:
    resolution = classify_generic_data_op(
        _node(
            "sparse_loading",
            "Load sparse h5ad matrices from disk",
            ConceptType.EXTERNAL_KNOWLEDGE,
        )
    )

    assert resolution is not None
    assert resolution.action_class == StageActionClass.EXTERNAL_KNOWLEDGE


def test_classifies_dtype_cast_as_trivial_inline() -> None:
    resolution = classify_generic_data_op(
        _node("memory_optimization", "Downcast float64 feature arrays to float16")
    )

    assert resolution is not None
    assert resolution.action_class == StageActionClass.TRIVIAL_INLINE


def test_known_atom_overrides_are_not_swallowed_by_generic_policy() -> None:
    assert classify_generic_data_op(
        _node("label_smoothing", "Apply label smoothing to classification labels")
    ) is None
    assert classify_generic_data_op(
        _node("dicom_windowing", "Apply DICOM window and uint8 conversion")
    ) is None


def test_non_atom_resolved_nodes_are_not_handed_to_hunter() -> None:
    node = _node("feature_concatenation", "Concatenate upstream embeddings")
    node = node.model_copy(
        update={
            "status": NodeStatus.ATOMIC,
            "action_class": "orchestration",
            "primitive_binding_source": "stage_resolution:v1",
            "resolution_reason": "Pipeline wiring.",
        }
    )
    cdg = CDGExport(nodes=[node], edges=[])

    assert is_non_atom_resolved_node(node)
    assert validate_handoff(cdg) == []
    assert to_pdg_nodes(cdg, strict=True) == []
