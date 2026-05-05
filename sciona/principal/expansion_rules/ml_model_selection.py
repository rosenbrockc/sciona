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
        "Train XGBoost Or CatBoost",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="xgboost.XGBModel",
        inputs=[IOSpec(name="X_train", type_desc="ndarray"), IOSpec(name="y_train", type_desc="ndarray")],
        outputs=[IOSpec(name="boosted_tree_model", type_desc="trained_model")],
        description="Train an XGBoost or CatBoost model to diversify the tree ensemble.",
    )
    blend = _node(
        "blend_tree_predictions",
        "Blend Tree Ensemble Predictions",
        ConceptType.ML_MODEL_SELECTION,
        matched_primitive="tree_ensemble_weighted_blend",
        inputs=[IOSpec(name="models", type_desc="list[trained_model]")],
        outputs=[IOSpec(name="predictions", type_desc="ndarray")],
        description="Blend LightGBM, XGBoost, CatBoost, ExtraTrees, or RandomForest predictions.",
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
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    is_ts = intermediates.get(f"{_PREFIX}.is_time_series")
    if is_ts is None:
        return None
    try:
        ts_flag = bool(is_ts)
    except (ValueError, TypeError):
        return None
    if not ts_flag:
        return None
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


def _diagnose_tree_ensemble_blend(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    explicit = _intermediate_truthy(
        context,
        f"{_PREFIX}.requires_tree_ensemble_blend",
        f"{_PREFIX}.use_lightgbm_catboost_ensemble",
        f"{_PREFIX}.use_xgboost_lightgbm_ensemble",
        f"{_PREFIX}.use_random_forest_xgboost_ensemble",
    )
    planning = _planning_text(context)
    planning_requires = any(
        token in planning
        for token in ("catboost", "lightgbm", "xgboost", "extratrees", "random forest ensemble")
    )
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="apply_tree_ensemble_blend",
        severity=0.80,
        evidence="A heterogeneous LightGBM/XGBoost/CatBoost-style tree ensemble is required.",
        metric_name="requires_tree_ensemble_blend",
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
            _build_insert_recursive_feature_elimination(),
            _build_insert_log_target_transform(),
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
            _diagnose_tree_ensemble_blend,
            _diagnose_recursive_feature_elimination,
            _diagnose_log_target_transform,
            _diagnose_smoothed_target_encoding,
        ]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
