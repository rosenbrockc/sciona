"""Expansion rules for deterministic ML model selection pipelines.

ML pipeline skeleton topology (linear chain):

    Data Assembly -> Preprocessing -> Estimator -> Validation

Expansion insertion points:
  - Before Estimator: insert preprocessing (PowerTransformer, StandardScaler)
  - Before Estimator: insert dimensionality reduction (PCA, TruncatedSVD)
  - At Estimator: replace with recommended sklearn class (Ridge, RandomForest, etc.)
  - At Validation: force cross-validation strategy (TimeSeriesSplit, GroupKFold)

Diagnostics are driven by the 8 recommendation atoms in
``sciona.atoms.ml.model_selection.recommendations``, whose outputs
land in ``context.intermediates`` under the ``model_selection.*`` prefix.
"""

from __future__ import annotations

import logging

from sciona.architect.graph_rewriter import GraphState, Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)

logger = logging.getLogger(__name__)

_DOMAIN = "ml_model_selection"
_PREFIX = "model_selection"

_SEVERITY_MAP = {"absolute": 1.0, "high": 0.85, "medium": 0.6}


def _node(
    node_id: str, name: str, concept_type: ConceptType, *,
    matched_primitive: str | None = None, inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None, description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id, name=name, description=description or name,
        concept_type=concept_type, status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive, inputs=inputs or [],
        outputs=outputs or [], type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str, target_id: str, output_name: str = "out",
    input_name: str = "in", type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id, target_id=target_id, output_name=output_name,
        input_name=input_name, source_type=type_desc, target_type=type_desc,
    )


def _semantic_only_rule(
    name: str,
    *,
    priority: int,
    semantic_apply,
) -> RewriteRule:
    """Build a rule whose runtime behavior is handled by ``semantic_apply``."""
    sentinel = _node(
        f"semantic_only_{name}",
        f"semantic_only_{name}",
        ConceptType.CUSTOM,
        matched_primitive=f"__semantic_only__.{name}",
    )
    graph = CDGExport(nodes=[sentinel], edges=[])
    return RewriteRule(
        name=name,
        lhs=graph,
        rhs=graph,
        interface=CDGExport(nodes=[], edges=[]),
        l_morphism=Morphism(node_map={}, edge_map={}),
        r_morphism=Morphism(node_map={}, edge_map={}),
        priority=priority,
        semantic_apply=semantic_apply,
    )


def _normalized_label(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _node_matches_any_label(node: AlgorithmicNode, labels: set[str]) -> bool:
    return (
        _normalized_label(node.node_id) in labels
        or _normalized_label(node.name) in labels
    )


def _fresh_node_id(base: str, used: set[str]) -> str:
    candidate = base
    ordinal = 2
    while candidate in used:
        candidate = f"{base}_{ordinal}"
        ordinal += 1
    used.add(candidate)
    return candidate


def _retarget_parent_children(
    nodes: list[AlgorithmicNode],
    *,
    parent_id: str | None,
    removed_ids: set[str],
    added_ids: list[str],
) -> list[AlgorithmicNode]:
    if not parent_id:
        return nodes
    updated: list[AlgorithmicNode] = []
    for node in nodes:
        if node.node_id != parent_id:
            updated.append(node)
            continue
        children = [child for child in node.children if child not in removed_ids]
        for child in added_ids:
            if child not in children:
                children.append(child)
        updated.append(node.model_copy(update={"children": children}))
    return updated


def _replace_nodes_with_chain(
    graph: CDGExport,
    *,
    target_labels: list[str],
    chain_nodes: list[AlgorithmicNode],
    chain_edges: list[DependencyEdge],
) -> GraphState[CDGExport]:
    labels = {_normalized_label(label) for label in target_labels}
    target_nodes = [
        node for node in graph.nodes if _node_matches_any_label(node, labels)
    ]
    if not target_nodes:
        return GraphState.failure(f"No target nodes found for labels: {sorted(labels)}")

    target_ids = {node.node_id for node in target_nodes}
    first_target = target_nodes[0]
    parent_id = first_target.parent_id
    depth = first_target.depth
    used_ids = {node.node_id for node in graph.nodes if node.node_id not in target_ids}

    id_map: dict[str, str] = {}
    added_nodes: list[AlgorithmicNode] = []
    for node in chain_nodes:
        new_id = _fresh_node_id(node.node_id, used_ids)
        id_map[node.node_id] = new_id
        added_nodes.append(
            node.model_copy(
                update={"node_id": new_id, "parent_id": parent_id, "depth": depth}
            )
        )

    first_new = added_nodes[0].node_id
    last_new = added_nodes[-1].node_id
    incoming = [
        edge
        for edge in graph.edges
        if edge.target_id in target_ids and edge.source_id not in target_ids
    ]
    outgoing = [
        edge
        for edge in graph.edges
        if edge.source_id in target_ids and edge.target_id not in target_ids
    ]
    retained_edges = [
        edge
        for edge in graph.edges
        if edge.source_id not in target_ids and edge.target_id not in target_ids
    ]

    new_edges = list(retained_edges)
    for edge in incoming:
        new_edges.append(edge.model_copy(update={"target_id": first_new}))
    for edge in outgoing:
        new_edges.append(edge.model_copy(update={"source_id": last_new}))
    for edge in chain_edges:
        new_edges.append(
            edge.model_copy(
                update={
                    "source_id": id_map[edge.source_id],
                    "target_id": id_map[edge.target_id],
                }
            )
        )

    new_nodes = [node for node in graph.nodes if node.node_id not in target_ids]
    new_nodes.extend(added_nodes)
    new_nodes = _retarget_parent_children(
        new_nodes,
        parent_id=parent_id,
        removed_ids=target_ids,
        added_ids=[node.node_id for node in added_nodes],
    )
    return GraphState.success(
        graph.model_copy(update={"nodes": new_nodes, "edges": new_edges})
    )


def _insert_chain_between(
    graph: CDGExport,
    *,
    source_labels: list[str],
    target_labels: list[str],
    chain_nodes: list[AlgorithmicNode],
    chain_edges: list[DependencyEdge],
) -> GraphState[CDGExport]:
    source_set = {_normalized_label(label) for label in source_labels}
    target_set = {_normalized_label(label) for label in target_labels}
    source_ids = {
        node.node_id for node in graph.nodes if _node_matches_any_label(node, source_set)
    }
    target_ids = {
        node.node_id for node in graph.nodes if _node_matches_any_label(node, target_set)
    }
    selected_edge = next(
        (
            edge
            for edge in graph.edges
            if edge.source_id in source_ids and edge.target_id in target_ids
        ),
        None,
    )
    if selected_edge is None:
        return GraphState.failure("No edge found between requested insertion endpoints")

    node_by_id = {node.node_id: node for node in graph.nodes}
    source_node = node_by_id.get(selected_edge.source_id)
    target_node = node_by_id.get(selected_edge.target_id)
    parent_id = (source_node.parent_id if source_node else None) or (
        target_node.parent_id if target_node else None
    )
    depth = min(
        value
        for value in [
            source_node.depth if source_node else None,
            target_node.depth if target_node else None,
        ]
        if value is not None
    )
    used_ids = {node.node_id for node in graph.nodes}
    id_map: dict[str, str] = {}
    added_nodes: list[AlgorithmicNode] = []
    for node in chain_nodes:
        new_id = _fresh_node_id(node.node_id, used_ids)
        id_map[node.node_id] = new_id
        added_nodes.append(
            node.model_copy(
                update={"node_id": new_id, "parent_id": parent_id, "depth": depth}
            )
        )

    first_new = added_nodes[0].node_id
    last_new = added_nodes[-1].node_id
    retained_edges = [
        edge
        for edge in graph.edges
        if not (
            edge.source_id == selected_edge.source_id
            and edge.target_id == selected_edge.target_id
        )
    ]
    new_edges = [
        *retained_edges,
        selected_edge.model_copy(update={"target_id": first_new}),
        selected_edge.model_copy(update={"source_id": last_new}),
    ]
    for edge in chain_edges:
        new_edges.append(
            edge.model_copy(
                update={
                    "source_id": id_map[edge.source_id],
                    "target_id": id_map[edge.target_id],
                }
            )
        )

    new_nodes = [*graph.nodes, *added_nodes]
    new_nodes = _retarget_parent_children(
        new_nodes,
        parent_id=parent_id,
        removed_ids=set(),
        added_ids=[node.node_id for node in added_nodes],
    )
    return GraphState.success(
        graph.model_copy(update={"nodes": new_nodes, "edges": new_edges})
    )


def _get_rec(context: ExpansionContext, key: str) -> dict | None:
    """Extract a recommendation dict from intermediates."""
    intermediates = context.intermediates or {}
    rec = intermediates.get(f"{_PREFIX}.{key}")
    if isinstance(rec, dict) and "recommendation" in rec:
        return rec
    return None


def _severity(rec: dict) -> float:
    """Map recommendation confidence to expansion severity."""
    return _SEVERITY_MAP.get(str(rec.get("confidence", "medium")), 0.5)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on", "recommended", "required"}


def _intermediate_truthy(context: ExpansionContext, *keys: str) -> bool:
    intermediates = context.intermediates or {}
    return any(_truthy(intermediates.get(key)) for key in keys)


def _planning_text(context: ExpansionContext) -> str:
    artifact = context.planning_artifact or {}
    return str(artifact).lower() if isinstance(artifact, dict) else ""


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_preprocessing_before_estimator() -> RewriteRule:
    """Insert a preprocessing step (PowerTransformer/StandardScaler) before estimator."""
    src = _node("src", "source", ConceptType.DATA_ASSEMBLY)
    estimator = _node("estimator", "Estimator", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[src, estimator], edges=[_edge("src", "estimator")])
    interface = CDGExport(nodes=[src, estimator], edges=[])

    preprocess = _node(
        "preprocess", "Preprocessing", ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.preprocessing.StandardScaler",
        inputs=[IOSpec(name="X", type_desc="ndarray")],
        outputs=[IOSpec(name="X_transformed", type_desc="ndarray")],
        description="Apply feature-level preprocessing (scaling, power transform).",
        type_signature="ndarray -> ndarray",
    )
    rhs = CDGExport(
        nodes=[src, preprocess, estimator],
        edges=[_edge("src", "preprocess"), _edge("preprocess", "estimator")],
    )

    return RewriteRule(
        name="insert_preprocessing_before_estimator",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        priority=3,
    )


def _build_replace_estimator_from_recommendation() -> RewriteRule:
    """Replace the estimator node with the recommended sklearn class.

    Uses ``semantic_apply`` to dynamically set the ``matched_primitive``
    from the recommendation dict rather than a static RHS.
    """
    estimator = _node("estimator", "Estimator", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[estimator], edges=[])
    interface = CDGExport(nodes=[estimator], edges=[])
    rhs = CDGExport(nodes=[estimator], edges=[])

    return RewriteRule(
        name="replace_estimator_from_recommendation",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"estimator": "estimator"}, edge_map={}),
        r_morphism=Morphism(node_map={"estimator": "estimator"}, edge_map={}),
        priority=5,
    )


def _build_insert_dimensionality_reduction() -> RewriteRule:
    """Insert PCA or TruncatedSVD before the estimator."""
    src = _node("src", "source", ConceptType.DATA_ASSEMBLY)
    estimator = _node("estimator", "Estimator", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[src, estimator], edges=[_edge("src", "estimator")])
    interface = CDGExport(nodes=[src, estimator], edges=[])

    dimred = _node(
        "dimred", "Dimensionality Reduction", ConceptType.DIMENSIONALITY_REDUCTION,
        matched_primitive="sklearn.decomposition.PCA",
        inputs=[IOSpec(name="X", type_desc="ndarray")],
        outputs=[IOSpec(name="X_reduced", type_desc="ndarray")],
        description="Reduce feature dimensionality to improve conditioning.",
        type_signature="ndarray -> ndarray",
    )
    rhs = CDGExport(
        nodes=[src, dimred, estimator],
        edges=[_edge("src", "dimred"), _edge("dimred", "estimator")],
    )

    return RewriteRule(
        name="insert_dimensionality_reduction_before_estimator",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        priority=2,
    )


def _build_force_cv_strategy() -> RewriteRule:
    """Force a specific cross-validation strategy (TimeSeriesSplit, GroupKFold)."""
    validation = _node("validation", "Validation", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[validation], edges=[])
    interface = CDGExport(nodes=[validation], edges=[])
    rhs = CDGExport(nodes=[validation], edges=[])

    return RewriteRule(
        name="force_cv_strategy",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"validation": "validation"}, edge_map={}),
        r_morphism=Morphism(node_map={"validation": "validation"}, edge_map={}),
        priority=4,
    )


def _build_apply_kfold_ensemble() -> RewriteRule:
    """Replace single model training with fold splitting plus per-fold training."""
    split = _node(
        "split_folds",
        "Split Folds",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.model_selection.StratifiedKFold",
        inputs=[
            IOSpec(name="X_train", type_desc="ndarray"),
            IOSpec(name="y_train", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="folds", type_desc="list[fold]")],
        description="Split training data into leakage-safe cross-validation folds.",
    )
    train = _node(
        "train_fold_models",
        "Train Fold Models",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="cross_validated_model_training",
        inputs=[IOSpec(name="folds", type_desc="list[fold]")],
        outputs=[
            IOSpec(name="models", type_desc="list[trained_model]"),
            IOSpec(name="oof_predictions", type_desc="ndarray"),
        ],
        description="Train one model per fold and produce out-of-fold predictions.",
    )
    return _semantic_only_rule(
        "apply_kfold_ensemble",
        priority=6,
        semantic_apply=lambda graph: _replace_nodes_with_chain(
            graph,
            target_labels=["model_training", "estimator"],
            chain_nodes=[split, train],
            chain_edges=[_edge("split_folds", "train_fold_models", "folds", "folds")],
        ),
    )


def _build_apply_stacking_ensemble() -> RewriteRule:
    """Replace flat prediction ensemble with two-level stacking."""
    collect = _node(
        "collect_oof_predictions",
        "Collect OOF Predictions",
        ConceptType.DATA_ASSEMBLY,
        matched_primitive=(
            "sciona.atoms.ml.sklearn.ensemble.stacking_meta_features."
            "stacking_meta_feature_matrix"
        ),
        inputs=[IOSpec(name="oof_predictions", type_desc="list[ndarray]")],
        outputs=[IOSpec(name="meta_features", type_desc="ndarray")],
        description="Collect out-of-fold level-one predictions into meta-features.",
    )
    train_meta = _node(
        "train_meta_learner",
        "Train Meta Learner",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.linear_model.Ridge",
        inputs=[
            IOSpec(name="meta_features", type_desc="ndarray"),
            IOSpec(name="y_train", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="meta_learner", type_desc="trained_model")],
        description="Train a simple second-level model on OOF meta-features.",
    )
    predict = _node(
        "meta_predict",
        "Meta Predict",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive=(
            "sciona.atoms.ml.sklearn.ensemble.stacking_classifier_outputs."
            "stacking_classifier_probability_matrix_from_blocks"
        ),
        inputs=[
            IOSpec(name="level1_predictions", type_desc="list[ndarray]"),
            IOSpec(name="meta_learner", type_desc="trained_model"),
        ],
        outputs=[IOSpec(name="predictions", type_desc="ndarray")],
        description="Predict with the second-level stacking model.",
    )
    return _semantic_only_rule(
        "apply_stacking_ensemble",
        priority=5,
        semantic_apply=lambda graph: _replace_nodes_with_chain(
            graph,
            target_labels=["prediction_ensemble", "ensemble", "validation"],
            chain_nodes=[collect, train_meta, predict],
            chain_edges=[
                _edge(
                    "collect_oof_predictions",
                    "train_meta_learner",
                    "meta_features",
                    "meta_features",
                ),
                _edge(
                    "train_meta_learner",
                    "meta_predict",
                    "meta_learner",
                    "meta_learner",
                ),
            ],
        ),
    )


def _build_insert_constraint_injection() -> RewriteRule:
    """Insert explicit constraint verification and correction before ensembling."""
    verify = _node(
        "verify_constraint",
        "Verify Constraint",
        ConceptType.OBSERVABILITY,
        matched_primitive=(
            "sciona.atoms.ml.constrained_ml.decorrelation."
            "compute_cvm_mass_decorrelation"
        ),
        inputs=[
            IOSpec(name="predictions", type_desc="ndarray"),
            IOSpec(name="protected_variable", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="constraint_result", type_desc="tuple[bool, float]")],
        description="Test whether predictions satisfy the declared statistical constraint.",
    )
    apply = _node(
        "apply_decorrelation",
        "Apply Decorrelation",
        ConceptType.OPTIMIZATION,
        matched_primitive=(
            "sciona.atoms.ml.constrained_ml.decorrelation."
            "noise_injection_decorrelation"
        ),
        inputs=[
            IOSpec(name="predictions", type_desc="ndarray"),
            IOSpec(name="constraint_result", type_desc="tuple[bool, float]"),
        ],
        outputs=[IOSpec(name="predictions", type_desc="ndarray")],
        description="Apply post-hoc or training-time correction when the constraint fails.",
    )
    return _semantic_only_rule(
        "insert_constraint_injection",
        priority=4,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["model_training", "train_fold_models", "estimator"],
            target_labels=["prediction_ensemble", "ensemble", "validation"],
            chain_nodes=[verify, apply],
            chain_edges=[
                _edge(
                    "verify_constraint",
                    "apply_decorrelation",
                    "constraint_result",
                    "constraint_result",
                )
            ],
        ),
    )


def _build_apply_dl_backbone_substitution() -> RewriteRule:
    """Rewrite feature engineering plus model training into pretrained finetuning."""
    load = _node(
        "load_pretrained",
        "Load Pretrained Backbone",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="pretrained_backbone_loader",
        inputs=[IOSpec(name="model_name", type_desc="str")],
        outputs=[IOSpec(name="backbone", type_desc="nn.Module")],
        description="Load a pretrained backbone and optionally freeze early layers.",
    )
    finetune = _node(
        "finetune_backbone",
        "Finetune Backbone",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="finetune_pretrained_backbone",
        inputs=[
            IOSpec(name="backbone", type_desc="nn.Module"),
            IOSpec(name="train_data", type_desc="DataLoader"),
        ],
        outputs=[IOSpec(name="model", type_desc="nn.Module")],
        description="Finetune the backbone on task data with augmentation and regularization.",
    )
    tta = _node(
        "test_time_augmentation",
        "Test-Time Augmentation",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="test_time_augmentation",
        inputs=[
            IOSpec(name="model", type_desc="nn.Module"),
            IOSpec(name="test_data", type_desc="DataLoader"),
        ],
        outputs=[IOSpec(name="predictions", type_desc="ndarray")],
        description="Average predictions over safe test-time augmentations.",
    )
    return _semantic_only_rule(
        "apply_dl_backbone_substitution",
        priority=4,
        semantic_apply=lambda graph: _replace_nodes_with_chain(
            graph,
            target_labels=["feature_engineering", "model_training"],
            chain_nodes=[load, finetune, tta],
            chain_edges=[
                _edge("load_pretrained", "finetune_backbone", "backbone", "backbone"),
                _edge("finetune_backbone", "test_time_augmentation", "model", "model"),
            ],
        ),
    )


def _build_apply_tree_ensemble_blend() -> RewriteRule:
    """Replace a single estimator with a heterogeneous tree-ensemble blend."""
    train_lgbm = _node(
        "train_lightgbm",
        "Train LightGBM",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="lightgbm.LGBMModel",
        inputs=[IOSpec(name="X_train", type_desc="ndarray"), IOSpec(name="y_train", type_desc="ndarray")],
        outputs=[IOSpec(name="lightgbm_model", type_desc="trained_model")],
        description="Train a LightGBM model as one member of a heterogeneous tabular ensemble.",
    )
    train_xgb = _node(
        "train_xgboost",
        "Train XGBoost CatBoost Or SVR",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="xgboost.XGBModel",
        inputs=[IOSpec(name="X_train", type_desc="ndarray"), IOSpec(name="y_train", type_desc="ndarray")],
        outputs=[IOSpec(name="boosted_tree_model", type_desc="trained_model")],
        description="Train an XGBoost, CatBoost, or SVR model to diversify the ensemble.",
    )
    blend = _node(
        "blend_tree_predictions",
        "Blend Tree Ensemble Predictions",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="tree_ensemble_weighted_blend",
        inputs=[IOSpec(name="models", type_desc="list[trained_model]")],
        outputs=[IOSpec(name="predictions", type_desc="ndarray")],
        description="Blend LightGBM, XGBoost, CatBoost, ExtraTrees, RandomForest, or SVR predictions.",
    )
    return _semantic_only_rule(
        "apply_tree_ensemble_blend",
        priority=5,
        semantic_apply=lambda graph: _replace_nodes_with_chain(
            graph,
            target_labels=["model_training", "estimator", "prediction_ensemble"],
            chain_nodes=[train_lgbm, train_xgb, blend],
            chain_edges=[
                _edge("train_lightgbm", "blend_tree_predictions", "lightgbm_model", "models"),
                _edge("train_xgboost", "blend_tree_predictions", "boosted_tree_model", "models"),
            ],
        ),
    )


def _build_apply_pretrained_backbone_ensemble() -> RewriteRule:
    """Replace single training with a blend of pretrained image/text backbones."""
    train_cnn = _node(
        "train_pretrained_cnn_backbone",
        "Train Pretrained CNN Backbone",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="efficientnet_resnet_densenet_finetuning",
        inputs=[IOSpec(name="train_data", type_desc="DataLoader")],
        outputs=[IOSpec(name="cnn_model", type_desc="nn.Module")],
        description="Finetune an EfficientNet, ResNet, DenseNet, Inception, or VGG-style backbone.",
    )
    train_transformer = _node(
        "train_pretrained_transformer_backbone",
        "Train Pretrained Transformer Backbone",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="swin_deit_convnext_finetuning",
        inputs=[IOSpec(name="train_data", type_desc="DataLoader")],
        outputs=[IOSpec(name="transformer_model", type_desc="nn.Module")],
        description="Finetune a Swin, DeiT, ConvNeXt, NFNet, or UNet-style transfer backbone.",
    )
    blend = _node(
        "blend_pretrained_backbones",
        "Blend Pretrained Backbone Predictions",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="pretrained_backbone_ensemble_blend",
        inputs=[IOSpec(name="models", type_desc="list[nn.Module]")],
        outputs=[IOSpec(name="predictions", type_desc="ndarray")],
        description="Blend predictions from diverse pretrained backbones.",
    )
    return _semantic_only_rule(
        "apply_pretrained_backbone_ensemble",
        priority=5,
        semantic_apply=lambda graph: _replace_nodes_with_chain(
            graph,
            target_labels=["model_training", "prediction_ensemble"],
            chain_nodes=[train_cnn, train_transformer, blend],
            chain_edges=[
                _edge("train_pretrained_cnn_backbone", "blend_pretrained_backbones", "cnn_model", "models"),
                _edge("train_pretrained_transformer_backbone", "blend_pretrained_backbones", "transformer_model", "models"),
            ],
        ),
    )


def _build_insert_recursive_feature_elimination() -> RewriteRule:
    """Insert recursive feature elimination before the estimator."""
    rfe = _node(
        "recursive_feature_elimination",
        "Recursive Feature Elimination",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.feature_selection.RFE",
        inputs=[IOSpec(name="X", type_desc="ndarray"), IOSpec(name="y", type_desc="ndarray")],
        outputs=[IOSpec(name="X_selected", type_desc="ndarray")],
        description="Select a compact feature subset with recursive feature elimination.",
    )
    return _semantic_only_rule(
        "insert_recursive_feature_elimination_before_estimator",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[rfe],
            chain_edges=[],
        ),
    )


def _build_insert_permutation_importance_feature_selection() -> RewriteRule:
    """Insert permutation-importance feature selection before model training."""
    select = _node(
        "permutation_importance_feature_selection",
        "Permutation Importance Feature Selection",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.inspection.permutation_importance",
        inputs=[IOSpec(name="X", type_desc="ndarray"), IOSpec(name="y", type_desc="ndarray")],
        outputs=[IOSpec(name="X_selected", type_desc="ndarray")],
        description="Select features by validating permutation importance and removing low-value columns.",
    )
    return _semantic_only_rule(
        "insert_permutation_importance_feature_selection_before_estimator",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[select],
            chain_edges=[],
        ),
    )


def _build_insert_balanced_sampling_before_training() -> RewriteRule:
    """Insert rare-class balanced sampling before model training."""
    sample = _node(
        "balanced_rare_class_sampling",
        "Balanced Rare-Class Sampling",
        ConceptType.DATA_ASSEMBLY,
        matched_primitive="balanced_oversampling_sampler",
        inputs=[IOSpec(name="X", type_desc="ndarray"), IOSpec(name="y", type_desc="ndarray")],
        outputs=[IOSpec(name="sampled_training_data", type_desc="tuple[ndarray, ndarray]")],
        description="Oversample or balance rare classes before estimator training.",
    )
    return _semantic_only_rule(
        "insert_balanced_sampling_before_training",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[sample],
            chain_edges=[],
        ),
    )


def _build_insert_pseudo_labeling_loop_before_training() -> RewriteRule:
    """Insert pseudo-label generation and filtered retraining before model training."""
    generate = _node(
        "generate_pseudo_labels",
        "Generate Pseudo Labels",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="pseudo_label_generation",
        inputs=[IOSpec(name="unlabeled_data", type_desc="DataFrame")],
        outputs=[IOSpec(name="pseudo_labeled_data", type_desc="DataFrame")],
        description="Generate pseudo-labels for unlabeled, test, external, scraped, or domain-adaptation data.",
    )
    filter_labels = _node(
        "filter_pseudo_labels",
        "Filter Pseudo Labels",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="confidence_filtered_pseudo_labels",
        inputs=[IOSpec(name="pseudo_labeled_data", type_desc="DataFrame")],
        outputs=[IOSpec(name="filtered_training_data", type_desc="DataFrame")],
        description="Keep high-confidence pseudo-labels before retraining.",
    )
    return _semantic_only_rule(
        "insert_pseudo_labeling_loop_before_training",
        priority=4,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[generate, filter_labels],
            chain_edges=[
                _edge("generate_pseudo_labels", "filter_pseudo_labels", "pseudo_labeled_data", "pseudo_labeled_data"),
            ],
        ),
    )


def _build_insert_iterative_imputation_before_estimator() -> RewriteRule:
    """Insert iterative imputation before estimator training."""
    impute = _node(
        "iterative_imputation",
        "Iterative Imputation",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.impute.IterativeImputer",
        inputs=[IOSpec(name="X", type_desc="ndarray")],
        outputs=[IOSpec(name="X_imputed", type_desc="ndarray")],
        description="Impute missing values with iterative chained models before estimator training.",
    )
    return _semantic_only_rule(
        "insert_iterative_imputation_before_estimator",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[impute],
            chain_edges=[],
        ),
    )


def _build_insert_feature_hashing_before_estimator() -> RewriteRule:
    """Insert feature hashing for high-cardinality sparse inputs."""
    hashing = _node(
        "feature_hashing",
        "Feature Hashing",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.feature_extraction.FeatureHasher",
        inputs=[IOSpec(name="categorical_or_text_features", type_desc="DataFrame")],
        outputs=[IOSpec(name="hashed_features", type_desc="sparse_matrix")],
        description="Hash high-cardinality categorical or text features into a fixed sparse feature space.",
    )
    return _semantic_only_rule(
        "insert_feature_hashing_before_estimator",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[hashing],
            chain_edges=[],
        ),
    )


def _build_insert_tree_early_stopping_validation() -> RewriteRule:
    """Insert an early-stopping validation callback for boosted trees."""
    early = _node(
        "tree_early_stopping",
        "Boosted Tree Early Stopping",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="lightgbm_xgboost_early_stopping",
        inputs=[IOSpec(name="validation_set", type_desc="tuple[ndarray, ndarray]")],
        outputs=[IOSpec(name="best_iteration", type_desc="int")],
        description="Stop LightGBM or XGBoost training at the best validation iteration.",
    )
    return _semantic_only_rule(
        "insert_tree_early_stopping_validation",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["model_training", "estimator"],
            target_labels=["prediction_ensemble", "ensemble", "validation", "output"],
            chain_nodes=[early],
            chain_edges=[],
        ),
    )


def _build_insert_log_target_transform() -> RewriteRule:
    """Insert a log target transform for skewed regression targets."""
    transform = _node(
        "log_target_transform",
        "Log Target Transform",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="numpy.log1p",
        inputs=[IOSpec(name="y", type_desc="ndarray")],
        outputs=[IOSpec(name="y_log", type_desc="ndarray")],
        description="Train on log-transformed targets and invert predictions with expm1.",
    )
    return _semantic_only_rule(
        "insert_log_target_transform_before_estimator",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[transform],
            chain_edges=[],
        ),
    )


def _build_replace_loss_with_metric_aligned_objective() -> RewriteRule:
    """Replace generic model training with a leaderboard-metric-aligned objective."""
    configure = _node(
        "configure_metric_aligned_objective",
        "Configure Metric-Aligned Objective",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="metric_aligned_custom_objective",
        inputs=[IOSpec(name="metric", type_desc="str")],
        outputs=[IOSpec(name="objective", type_desc="callable")],
        description="Configure custom loss for quantile, RMSLE/RMSE, partial AUC, Brier, Pearson, or F1-weighted objectives.",
    )
    train = _node(
        "train_metric_aligned_estimator",
        "Train Metric-Aligned Estimator",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="lightgbm_xgboost_custom_objective",
        inputs=[IOSpec(name="objective", type_desc="callable")],
        outputs=[IOSpec(name="model", type_desc="trained_model")],
        description="Train LightGBM, XGBoost, or a compatible estimator with the configured objective.",
    )
    return _semantic_only_rule(
        "replace_loss_with_metric_aligned_objective",
        priority=4,
        semantic_apply=lambda graph: _replace_nodes_with_chain(
            graph,
            target_labels=["model_training", "estimator"],
            chain_nodes=[configure, train],
            chain_edges=[
                _edge("configure_metric_aligned_objective", "train_metric_aligned_estimator", "objective", "objective"),
            ],
        ),
    )


def _build_insert_metric_optimized_thresholding() -> RewriteRule:
    """Insert threshold calibration before final output formatting."""
    threshold = _node(
        "metric_optimized_thresholding",
        "Metric-Optimized Thresholding",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="macro_f1_mcc_threshold_optimization",
        inputs=[IOSpec(name="probabilities", type_desc="ndarray")],
        outputs=[IOSpec(name="labels", type_desc="ndarray")],
        description="Tune class or label thresholds against Macro F1, MCC, tag F1, or related discrete metrics.",
    )
    return _semantic_only_rule(
        "insert_metric_optimized_thresholding_after_prediction",
        priority=4,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["prediction_ensemble", "ensemble", "meta_predict"],
            target_labels=["output", "submission", "validation"],
            chain_nodes=[threshold],
            chain_edges=[],
        ),
    )


def _build_insert_retrieval_reranking() -> RewriteRule:
    """Insert local-feature or pairwise model reranking after first-pass predictions."""
    rerank = _node(
        "retrieval_reranking",
        "Retrieval Re-Ranking",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="delf_xgboost_pairwise_reranking",
        inputs=[IOSpec(name="candidates", type_desc="DataFrame")],
        outputs=[IOSpec(name="reranked_candidates", type_desc="DataFrame")],
        description="Rerank first-pass retrieval candidates with DELF local features or an XGBoost pair-ranker.",
    )
    return _semantic_only_rule(
        "insert_retrieval_reranking_after_prediction",
        priority=4,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["prediction_ensemble", "ensemble", "meta_predict"],
            target_labels=["output", "submission", "validation"],
            chain_nodes=[rerank],
            chain_edges=[],
        ),
    )


def _build_insert_database_augmentation_for_retrieval() -> RewriteRule:
    """Insert database augmentation for retrieval embeddings."""
    dba = _node(
        "database_augmentation",
        "Database Augmentation",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="retrieval_database_augmentation",
        inputs=[IOSpec(name="embeddings", type_desc="ndarray")],
        outputs=[IOSpec(name="augmented_embeddings", type_desc="ndarray")],
        description="Average or diffuse retrieval database embeddings with nearest-neighbor evidence before search or reranking.",
    )
    return _semantic_only_rule(
        "insert_database_augmentation_for_retrieval",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["prediction_ensemble", "ensemble", "meta_predict"],
            target_labels=["output", "submission", "validation"],
            chain_nodes=[dba],
            chain_edges=[],
        ),
    )


def _build_insert_prompt_reasoning_augmentation() -> RewriteRule:
    """Insert prompt/chain-of-thought augmentation for LLM-style tasks."""
    prompt_aug = _node(
        "prompt_reasoning_augmentation",
        "Prompt Reasoning Augmentation",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="chain_of_thought_prompt_augmentation",
        inputs=[IOSpec(name="prompt_examples", type_desc="list[str]")],
        outputs=[IOSpec(name="augmented_prompts", type_desc="list[str]")],
        description="Augment prompts or training traces with chain-of-thought-style reasoning examples when the task requires explicit reasoning.",
    )
    return _semantic_only_rule(
        "insert_prompt_reasoning_augmentation_before_training",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[prompt_aug],
            chain_edges=[],
        ),
    )


def _build_insert_smoothed_target_encoding() -> RewriteRule:
    """Insert leakage-safe smoothed target encoding for categorical features."""
    encode = _node(
        "smoothed_target_encoding",
        "Smoothed Target Encoding",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="category_encoders.TargetEncoder",
        inputs=[IOSpec(name="categorical_features", type_desc="ndarray"), IOSpec(name="y", type_desc="ndarray")],
        outputs=[IOSpec(name="encoded_features", type_desc="ndarray")],
        description="Apply fold-aware mean target encoding with Bayesian smoothing.",
    )
    return _semantic_only_rule(
        "insert_smoothed_target_encoding_before_estimator",
        priority=3,
        semantic_apply=lambda graph: _insert_chain_between(
            graph,
            source_labels=["preprocessing", "feature_engineering", "load_features", "source"],
            target_labels=["estimator", "model_training"],
            chain_nodes=[encode],
            chain_edges=[],
        ),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_regularization(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_regularization")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    cond = intermediates.get(f"{_PREFIX}.condition_number")
    if cond is None:
        return None
    try:
        cond_val = float(cond)
    except (ValueError, TypeError):
        return None
    if cond_val <= 30.0:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Condition number {cond_val:.1f} > 30"),
        metric_name="condition_number",
        metric_value=cond_val,
        threshold=30.0,
        source_domain=_DOMAIN,
    )


def _diagnose_loss_function(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_loss_function")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    kurtosis = intermediates.get(f"{_PREFIX}.residual_kurtosis")
    if kurtosis is None:
        kurtosis = intermediates.get(f"{_PREFIX}.excess_kurtosis")
    if kurtosis is None:
        return None
    try:
        k_val = float(kurtosis)
    except (ValueError, TypeError):
        return None
    if k_val <= 1.0:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Kurtosis {k_val:.2f} > 1.0 — robust loss needed"),
        metric_name="residual_kurtosis",
        metric_value=k_val,
        threshold=1.0,
        source_domain=_DOMAIN,
    )


def _diagnose_linear_model(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_linear_model")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    di = intermediates.get(f"{_PREFIX}.dispersion_index")
    if di is None:
        return None
    try:
        di_val = float(di)
    except (ValueError, TypeError):
        return None
    if abs(di_val - 1.0) < 0.1:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Dispersion index {di_val:.2f} deviates from 1.0"),
        metric_name="dispersion_index",
        metric_value=di_val,
        threshold=1.1,
        source_domain=_DOMAIN,
    )


def _diagnose_tree_ensemble(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_tree_ensemble")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    noise = intermediates.get(f"{_PREFIX}.noise_level")
    if noise is None:
        return None
    try:
        noise_val = float(noise)
    except (ValueError, TypeError):
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Noise level {noise_val:.3f}"),
        metric_name="noise_level",
        metric_value=noise_val,
        threshold=0.5,
        source_domain=_DOMAIN,
    )


def _diagnose_preprocessing(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_preprocessing")
    explicit_scaling = _intermediate_truthy(
        context,
        f"{_PREFIX}.use_standard_scaler",
        f"{_PREFIX}.requires_feature_scaling",
    )
    planning = _planning_text(context)
    if rec is None and (
        explicit_scaling
        or "standardscaler" in planning
        or "standard scaler" in planning
    ):
        return ExpansionDiagnostic(
            rule_name="insert_preprocessing_before_estimator",
            severity=0.70,
            evidence="StandardScaler or equivalent numeric feature scaling is required.",
            metric_name="requires_feature_scaling",
            metric_value=1.0,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    if rec is None:
        return None
    # Only fire if preprocessing is actually recommended
    if rec.get("recommendation", "").lower().startswith("no preprocessing"):
        return None
    intermediates = context.intermediates or {}
    # Use max VIF or max abs skewness as the trigger metric
    vif_max = intermediates.get(f"{_PREFIX}.vif_max")
    skew_max = intermediates.get(f"{_PREFIX}.skewness_max")
    metric_name = "vif_max"
    metric_value = 0.0
    threshold = 5.0
    if vif_max is not None:
        try:
            metric_value = float(vif_max)
        except (ValueError, TypeError):
            metric_value = 0.0
    elif skew_max is not None:
        try:
            metric_value = abs(float(skew_max))
            metric_name = "skewness_max"
            threshold = 1.0
        except (ValueError, TypeError):
            metric_value = 0.0
    if metric_value <= threshold:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_preprocessing_before_estimator",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"{metric_name} {metric_value:.2f} > {threshold}"),
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        source_domain=_DOMAIN,
    )


def _diagnose_dimensionality_reduction(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_dimensionality_reduction")
    if rec is None:
        return None
    if rec.get("recommendation", "").lower().startswith("no reduction"):
        return None
    intermediates = context.intermediates or {}
    cond = intermediates.get(f"{_PREFIX}.condition_number")
    if cond is None:
        return None
    try:
        cond_val = float(cond)
    except (ValueError, TypeError):
        return None
    if cond_val <= 30.0:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_dimensionality_reduction_before_estimator",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Condition number {cond_val:.1f} > 30 — PCA recommended"),
        metric_name="condition_number",
        metric_value=cond_val,
        threshold=30.0,
        source_domain=_DOMAIN,
    )


def _diagnose_hyperparameter_ranges(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    """Hyperparameter range recommendations don't trigger graph rewrites.

    They annotate existing nodes rather than changing topology, so we log
    but don't produce an expansion diagnostic.
    """
    return None


def _diagnose_cv_strategy(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_cv_strategy")
    explicit_group = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_group_kfold",
        f"{_PREFIX}.use_group_kfold",
        f"{_PREFIX}.requires_group_aware_cv",
        f"{_PREFIX}.requires_stratified_group_kfold",
        f"{_PREFIX}.use_stratified_group_kfold",
    )
    explicit_spatial = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_spatial_cv",
        f"{_PREFIX}.use_spatial_cv",
        f"{_PREFIX}.requires_spatial_cross_validation",
    )
    planning = _planning_text(context)
    planning_group = any(
        token in planning
        for token in (
            "groupkfold",
            "group kfold",
            "stratified groupkfold",
            "stratified group",
            "group-aware",
            "patient id",
            "session id",
            "cultivar",
        )
    )
    planning_spatial = any(
        token in planning
        for token in (
            "spatial cross-validation",
            "spatial cross validation",
            "spatial cv",
            "scv",
            "location-based validation",
        )
    )
    requires_leakage_safe_cv = (
        explicit_group or planning_group or explicit_spatial or planning_spatial
    )
    if rec is None:
        if not requires_leakage_safe_cv:
            return None
        metric_name = (
            "requires_spatial_cv"
            if explicit_spatial or planning_spatial
            else "requires_group_aware_cv"
        )
        evidence = (
            "Spatial cross-validation is required to avoid leakage across nearby or location-related examples."
            if metric_name == "requires_spatial_cv"
            else "Group-aware cross-validation is required to avoid leakage across related examples."
        )
        return ExpansionDiagnostic(
            rule_name="force_cv_strategy",
            severity=1.0,
            evidence=evidence,
            metric_name=metric_name,
            metric_value=1.0,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    intermediates = context.intermediates or {}
    is_ts = intermediates.get(f"{_PREFIX}.is_time_series")
    if is_ts is None:
        if not requires_leakage_safe_cv:
            return None
        return ExpansionDiagnostic(
            rule_name="force_cv_strategy",
            severity=1.0,
            evidence=rec.get("reasoning", "Group-aware cross-validation is required."),
            metric_name=(
                "requires_spatial_cv"
                if explicit_spatial or planning_spatial
                else "requires_group_aware_cv"
            ),
            metric_value=1.0,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    try:
        ts_flag = bool(is_ts)
    except (ValueError, TypeError):
        return None
    if not ts_flag:
        if not requires_leakage_safe_cv:
            return None
        return ExpansionDiagnostic(
            rule_name="force_cv_strategy",
            severity=1.0,
            evidence=rec.get("reasoning", "Group-aware cross-validation is required."),
            metric_name=(
                "requires_spatial_cv"
                if explicit_spatial or planning_spatial
                else "requires_group_aware_cv"
            ),
            metric_value=1.0,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return ExpansionDiagnostic(
        rule_name="force_cv_strategy",
        severity=1.0,  # absolute confidence — time series leakage is critical
        evidence=rec.get("reasoning", "Time series detected — TimeSeriesSplit mandatory"),
        metric_name="is_time_series",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_kfold_ensemble(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_cv_strategy")
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.use_kfold_ensemble",
        f"{_PREFIX}.requires_oof_predictions",
        f"{_PREFIX}.requires_cross_validation",
    )
    recommendation = str((rec or {}).get("recommendation", "")).lower()
    if not explicit and "kfold" not in recommendation and "k-fold" not in recommendation:
        return None
    return ExpansionDiagnostic(
        rule_name="apply_kfold_ensemble",
        severity=_severity(rec or {"confidence": "high"}),
        evidence=(rec or {}).get(
            "reasoning",
            "Cross-validated training or out-of-fold predictions are required.",
        ),
        metric_name="requires_cross_validation",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_stacking_ensemble(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.use_stacking",
        f"{_PREFIX}.requires_meta_learner",
    )
    model_count = intermediates.get(f"{_PREFIX}.level1_model_count")
    try:
        count_value = float(model_count)
    except (TypeError, ValueError):
        count_value = 0.0
    if not explicit and count_value < 2.0:
        return None
    return ExpansionDiagnostic(
        rule_name="apply_stacking_ensemble",
        severity=0.75,
        evidence="Multiple level-one models or an explicit meta-learner requirement justify stacking.",
        metric_name="level1_model_count",
        metric_value=max(count_value, 2.0 if explicit else 0.0),
        threshold=1.0,
        source_domain=_DOMAIN,
    )


def _diagnose_constraint_injection(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_constraint_injection",
        f"{_PREFIX}.requires_decorrelation",
        f"{_PREFIX}.requires_fairness_constraint",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in (
            "decorrelation",
            "fairness",
            "protected_variable",
            "physics-invariance",
        )
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_constraint_injection",
        severity=0.85,
        evidence=(
            "A declared invariance, fairness, or decorrelation constraint needs "
            "an explicit verification/correction stage."
        ),
        metric_name="requires_constraint_injection",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_dl_backbone_substitution(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.use_pretrained_backbone",
        f"{_PREFIX}.requires_dl_backbone",
        f"{_PREFIX}.requires_transfer_learning",
    )
    rec = _get_rec(context, "recommend_dl_backbone")
    if not explicit and rec is None:
        return None
    return ExpansionDiagnostic(
        rule_name="apply_dl_backbone_substitution",
        severity=_severity(rec or {"confidence": "high"}),
        evidence=(rec or {}).get(
            "reasoning",
            "A pretrained backbone plus finetuning is a better fit than manual "
            "feature engineering and a shallow estimator.",
        ),
        metric_name="requires_dl_backbone",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_pretrained_backbone_ensemble(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_pretrained_backbone_ensemble",
        f"{_PREFIX}.use_pretrained_backbone_ensemble",
        f"{_PREFIX}.use_efficientnet_ensemble",
        f"{_PREFIX}.use_swin_unet_ensemble",
        f"{_PREFIX}.use_efficientnet_b4_b7_ensemble",
    )
    planning = _planning_text(context)
    backbone_terms = (
        "efficientnet",
        "densenet",
        "resnet",
        "inception",
        "vgg",
        "deit",
        "swin",
        "convnext",
        "nfnet",
        "unet",
    )
    planning_requires = "ensemble" in planning and any(term in planning for term in backbone_terms)
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="apply_pretrained_backbone_ensemble",
        severity=0.85,
        evidence="A diverse pretrained-backbone ensemble is required.",
        metric_name="requires_pretrained_backbone_ensemble",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_tree_ensemble_blend(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_tree_ensemble_blend",
        f"{_PREFIX}.use_lightgbm_catboost_ensemble",
        f"{_PREFIX}.use_xgboost_lightgbm_ensemble",
        f"{_PREFIX}.use_random_forest_xgboost_ensemble",
        f"{_PREFIX}.use_random_forest_extratrees_ensemble",
        f"{_PREFIX}.use_random_forest_svr_ensemble",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("catboost", "lightgbm", "xgboost", "extratrees", "random forest ensemble", "random forest and svr", "random forest and extratrees")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="apply_tree_ensemble_blend",
        severity=0.80,
        evidence="A heterogeneous LightGBM/XGBoost/CatBoost/RandomForest/ExtraTrees/SVR-style ensemble is required.",
        metric_name="requires_tree_ensemble_blend",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_permutation_importance_feature_selection(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_permutation_importance_feature_selection",
        f"{_PREFIX}.use_permutation_importance",
    )
    planning = _planning_text(context)
    planning_requires = "permutation importance" in planning or "feature importance-based selection" in planning
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_permutation_importance_feature_selection_before_estimator",
        severity=0.70,
        evidence="Permutation-importance feature selection is needed before estimator training.",
        metric_name="requires_permutation_importance_feature_selection",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_balanced_sampling(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_balanced_sampling",
        f"{_PREFIX}.use_balanced_oversampling",
        f"{_PREFIX}.use_rare_class_sampling",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("balanced over-sampling", "balanced oversampling", "rare classes", "rare species")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_balanced_sampling_before_training",
        severity=0.70,
        evidence="Rare classes require balanced sampling before training.",
        metric_name="requires_balanced_sampling",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_pseudo_labeling(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_pseudo_labeling",
        f"{_PREFIX}.use_pseudo_labeling",
        f"{_PREFIX}.use_test_pseudo_labels",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("pseudo-label", "pseudo label", "unlabeled regions", "unlabeled images", "external scraped")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_pseudo_labeling_loop_before_training",
        severity=0.75,
        evidence="Pseudo-labeling is required for unlabeled, test, external, or domain-adaptation data.",
        metric_name="requires_pseudo_labeling",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_iterative_imputation(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_iterative_imputation",
        f"{_PREFIX}.use_iterative_imputation",
    )
    planning = _planning_text(context)
    if not explicit and "iterative imputation" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_iterative_imputation_before_estimator",
        severity=0.70,
        evidence="Iterative imputation is required before estimator training.",
        metric_name="requires_iterative_imputation",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_feature_hashing(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_feature_hashing",
        f"{_PREFIX}.use_feature_hashing",
    )
    planning = _planning_text(context)
    if not explicit and "feature hashing" not in planning and "hashing trick" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_feature_hashing_before_estimator",
        severity=0.70,
        evidence="Feature hashing is required for high-cardinality sparse inputs.",
        metric_name="requires_feature_hashing",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_tree_early_stopping(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_tree_early_stopping",
        f"{_PREFIX}.use_tree_early_stopping",
    )
    planning = _planning_text(context)
    planning_requires = "early stopping" in planning and (
        "xgboost" in planning or "lightgbm" in planning
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_tree_early_stopping_validation",
        severity=0.65,
        evidence="Boosted-tree training requires early stopping against a validation set.",
        metric_name="requires_tree_early_stopping",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_recursive_feature_elimination(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_recursive_feature_elimination",
        f"{_PREFIX}.use_rfe",
    )
    if not explicit and "recursive feature elimination" not in _planning_text(context):
        return None
    return ExpansionDiagnostic(
        rule_name="insert_recursive_feature_elimination_before_estimator",
        severity=0.70,
        evidence="Recursive Feature Elimination is needed before estimator training.",
        metric_name="requires_recursive_feature_elimination",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_database_augmentation(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_database_augmentation",
        f"{_PREFIX}.use_database_augmentation",
    )
    planning = _planning_text(context)
    if not explicit and "database augmentation" not in planning and "dba" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_database_augmentation_for_retrieval",
        severity=0.70,
        evidence="Retrieval embeddings require database augmentation before final ranking.",
        metric_name="requires_database_augmentation",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_prompt_reasoning_augmentation(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_prompt_reasoning_augmentation",
        f"{_PREFIX}.use_chain_of_thought",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("chain-of-thought", "chain of thought", "cot prompting", "cot data augmentation")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_prompt_reasoning_augmentation_before_training",
        severity=0.65,
        evidence="Prompt or chain-of-thought reasoning augmentation is required.",
        metric_name="requires_prompt_reasoning_augmentation",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_log_target_transform(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_log_target_transform",
        f"{_PREFIX}.use_log_target_transform",
    )
    if not explicit and "log-target" not in _planning_text(context) and "log target" not in _planning_text(context):
        return None
    return ExpansionDiagnostic(
        rule_name="insert_log_target_transform_before_estimator",
        severity=0.70,
        evidence="A skewed regression target requires log1p target scaling and inverse-transform prediction.",
        metric_name="requires_log_target_transform",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_metric_aligned_objective(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_metric_aligned_objective",
        f"{_PREFIX}.use_custom_objective",
        f"{_PREFIX}.use_quantile_loss",
        f"{_PREFIX}.use_rmsle_objective",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in (
            "custom loss",
            "custom objective",
            "quantile loss",
            "rmsle",
            "rmse loss",
            "partial auc",
            "brier score",
            "pearson correlation",
            "f1-weighted loss",
        )
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_loss_with_metric_aligned_objective",
        severity=0.75,
        evidence="Training should use a loss or objective aligned with the competition metric.",
        metric_name="requires_metric_aligned_objective",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_metric_optimized_thresholding(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_metric_optimized_thresholding",
        f"{_PREFIX}.use_macro_f1_thresholding",
        f"{_PREFIX}.use_mcc_thresholding",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("macro f1", "mcc-optimized", "threshold optimization", "thresholding per", "tag f1")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_metric_optimized_thresholding_after_prediction",
        severity=0.75,
        evidence="Prediction thresholds should be optimized against the leaderboard metric.",
        metric_name="requires_metric_optimized_thresholding",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_retrieval_reranking(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_retrieval_reranking",
        f"{_PREFIX}.use_delf_reranking",
        f"{_PREFIX}.use_xgboost_reranking",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("delf", "deep local features", "pair re-ranking", "pair reranking", "xgboost for re-ranking", "xgboost for reranking")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_retrieval_reranking_after_prediction",
        severity=0.75,
        evidence="First-pass retrieval candidates require DELF or pairwise-model reranking.",
        metric_name="requires_retrieval_reranking",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_smoothed_target_encoding(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_target_encoding",
        f"{_PREFIX}.use_smoothed_target_encoding",
    )
    planning = _planning_text(context)
    if not explicit and "target encoding" not in planning and "mean target" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_smoothed_target_encoding_before_estimator",
        severity=0.75,
        evidence="Categorical features require leakage-safe smoothed target encoding.",
        metric_name="requires_target_encoding",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class MLModelSelectionRuleSet:
    """Expansion rules for deterministic ML model selection pipelines.

    Consumes outputs from the 8 recommendation atoms in
    ``sciona.atoms.ml.model_selection.recommendations`` and the 16
    diagnostic atoms in ``sciona.atoms.ml.model_selection.diagnostics``.
    """

    name = "ml_model_selection"
    domain = "ml_model_selection"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_preprocessing_before_estimator(),
            _build_replace_estimator_from_recommendation(),
            _build_insert_dimensionality_reduction(),
            _build_force_cv_strategy(),
            _build_apply_kfold_ensemble(),
            _build_apply_stacking_ensemble(),
            _build_insert_constraint_injection(),
            _build_apply_dl_backbone_substitution(),
            _build_apply_tree_ensemble_blend(),
            _build_apply_pretrained_backbone_ensemble(),
            _build_insert_recursive_feature_elimination(),
            _build_insert_permutation_importance_feature_selection(),
            _build_insert_balanced_sampling_before_training(),
            _build_insert_pseudo_labeling_loop_before_training(),
            _build_insert_iterative_imputation_before_estimator(),
            _build_insert_feature_hashing_before_estimator(),
            _build_insert_tree_early_stopping_validation(),
            _build_insert_log_target_transform(),
            _build_replace_loss_with_metric_aligned_objective(),
            _build_insert_metric_optimized_thresholding(),
            _build_insert_retrieval_reranking(),
            _build_insert_database_augmentation_for_retrieval(),
            _build_insert_prompt_reasoning_augmentation(),
            _build_insert_smoothed_target_encoding(),
        ]

    def diagnose(
        self, cdg: CDGExport, context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [
            _diagnose_regularization,
            _diagnose_loss_function,
            _diagnose_linear_model,
            _diagnose_tree_ensemble,
            _diagnose_preprocessing,
            _diagnose_dimensionality_reduction,
            _diagnose_hyperparameter_ranges,
            _diagnose_cv_strategy,
            _diagnose_kfold_ensemble,
            _diagnose_stacking_ensemble,
            _diagnose_constraint_injection,
            _diagnose_dl_backbone_substitution,
            _diagnose_pretrained_backbone_ensemble,
            _diagnose_tree_ensemble_blend,
            _diagnose_permutation_importance_feature_selection,
            _diagnose_balanced_sampling,
            _diagnose_pseudo_labeling,
            _diagnose_iterative_imputation,
            _diagnose_feature_hashing,
            _diagnose_tree_early_stopping,
            _diagnose_recursive_feature_elimination,
            _diagnose_log_target_transform,
            _diagnose_metric_aligned_objective,
            _diagnose_metric_optimized_thresholding,
            _diagnose_retrieval_reranking,
            _diagnose_database_augmentation,
            _diagnose_prompt_reasoning_augmentation,
            _diagnose_smoothed_target_encoding,
        ]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
