from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_assets import (
    asset_backed_rule_sets,
    clear_local_expansion_asset_caches,
    load_local_expansion_assets_by_family,
)
from sciona.principal.expansion_rules.ml_model_selection import MLModelSelectionRuleSet
from sciona.principal.expansion_rules import default_rule_sets


def _tabular_pipeline_cdg() -> CDGExport:
    def edge(source_id: str, target_id: str) -> DependencyEdge:
        return DependencyEdge(
            source_id=source_id,
            target_id=target_id,
            output_name="result",
            input_name="input",
            source_type="object",
            target_type="object",
        )

    root = AlgorithmicNode(
        node_id="root",
        name="tabular_pipeline",
        description="tabular pipeline",
        concept_type=ConceptType.ML_MODEL_SELECTION,
        status=NodeStatus.DECOMPOSED,
        children=[
            "load_data",
            "feature_engineering",
            "model_training",
            "prediction_ensemble",
            "output",
        ],
        inputs=[IOSpec(name="raw_data", type_desc="DataFrame")],
        outputs=[IOSpec(name="result", type_desc="DataFrame")],
    )
    load = AlgorithmicNode(
        node_id="load_data",
        parent_id="root",
        name="load_data",
        description="load data",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
        depth=1,
    )
    features = AlgorithmicNode(
        node_id="feature_engineering",
        parent_id="root",
        name="feature_engineering",
        description="features",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
        depth=1,
    )
    train = AlgorithmicNode(
        node_id="model_training",
        parent_id="root",
        name="model_training",
        description="train model",
        concept_type=ConceptType.ML_MODEL_SELECTION,
        status=NodeStatus.ATOMIC,
        depth=1,
    )
    ensemble = AlgorithmicNode(
        node_id="prediction_ensemble",
        parent_id="root",
        name="prediction_ensemble",
        description="predict",
        concept_type=ConceptType.ML_MODEL_SELECTION,
        status=NodeStatus.ATOMIC,
        depth=1,
    )
    output = AlgorithmicNode(
        node_id="output",
        parent_id="root",
        name="output",
        description="format",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
        depth=1,
    )
    return CDGExport(
        nodes=[root, load, features, train, ensemble, output],
        edges=[
            edge("load_data", "feature_engineering"),
            edge("feature_engineering", "model_training"),
            edge("model_training", "prediction_ensemble"),
            edge("feature_engineering", "prediction_ensemble"),
            edge("prediction_ensemble", "output"),
        ],
    )


def _asset_backed_ml_rule_set():
    clear_local_expansion_asset_caches()
    return asset_backed_rule_sets([MLModelSelectionRuleSet()])[0]


def test_ml_model_selection_provider_expansion_asset_loads() -> None:
    clear_local_expansion_asset_caches()

    asset = load_local_expansion_assets_by_family()["ml_model_selection"]

    assert asset.audit.source_kind == "shared_asset"
    assert {operation.rule_name for operation in asset.operations} >= {
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
        "insert_constraint_injection",
        "apply_dl_backbone_substitution",
        "apply_tree_ensemble_blend",
        "insert_recursive_feature_elimination_before_estimator",
        "insert_log_target_transform_before_estimator",
        "insert_smoothed_target_encoding_before_estimator",
    }
    assert asset.operation("apply_kfold_ensemble").operation_type == "replace"
    assert (
        asset.operation("apply_kfold_ensemble").operation_id
        == "sciona.expansions.ml.kfold_ensemble"
    )


def test_known_provider_assets_wrap_default_rule_sets() -> None:
    clear_local_expansion_asset_caches()

    assets = load_local_expansion_assets_by_family()
    wrapped = {
        getattr(rule_set, "name", ""): type(rule_set).__name__
        for rule_set in default_rule_sets()
    }

    for family in assets:
        if family == "signal_event_rate":
            # signal_event_rate is an alias-backed specialized rule set in the
            # signal family and is already covered by provider asset tests.
            continue
        assert wrapped.get(family) == "AssetBackedExpansionRuleSet"


def test_default_rule_sets_have_provider_asset_operations() -> None:
    clear_local_expansion_asset_caches()

    assets = load_local_expansion_assets_by_family()
    rule_sets = default_rule_sets()
    missing_assets = sorted(
        getattr(rule_set, "name", "")
        for rule_set in rule_sets
        if getattr(rule_set, "name", "") not in assets
    )

    assert missing_assets == []

    for rule_set in rule_sets:
        family = getattr(rule_set, "name", "")
        asset_rule_names = {operation.rule_name for operation in assets[family].operations}
        runtime_rule_names = {rule.name for rule in rule_set.rules()}
        assert sorted(runtime_rule_names - asset_rule_names) == []


def test_provider_expansion_operations_have_applicability_contracts() -> None:
    clear_local_expansion_asset_caches()

    missing_contracts: list[tuple[str, str, list[str]]] = []
    for asset in load_local_expansion_assets_by_family().values():
        for operation in asset.operations:
            gaps: list[str] = []
            if not operation.operation_type:
                gaps.append("operation_type")
            if not operation.applies_to:
                gaps.append("applies_to")
            if not operation.trigger.metric_name:
                gaps.append("trigger.metric_name")
            if not operation.rewrite.before_summary:
                gaps.append("rewrite.before_summary")
            if not operation.rewrite.after_summary:
                gaps.append("rewrite.after_summary")
            if not operation.rewrite.information_flow_effect:
                gaps.append("rewrite.information_flow_effect")
            if gaps:
                missing_contracts.append((asset.family, operation.rule_name, gaps))

    assert missing_contracts == []


def test_kfold_ensemble_rule_uses_common_expansion_asset_metadata() -> None:
    result = ExpansionEngine([_asset_backed_ml_rule_set()]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(
            intermediates={
                "model_selection.use_kfold_ensemble": True,
            }
        ),
    )

    node_ids = {node.node_id for node in result.cdg.nodes}
    assert result.expanded is True
    assert "model_training" not in node_ids
    assert {"split_folds", "train_fold_models"}.issubset(node_ids)
    assert result.applied_rules == ("apply_kfold_ensemble",)
    assert result.applied_assets[0]["asset_operation_id"] == (
        "sciona.expansions.ml.kfold_ensemble"
    )
    assert result.applied_assets[0]["asset_operation_type"] == "replace"


def test_obvious_ml_expansion_rules_apply_to_tabular_pipeline() -> None:
    rule_set = _asset_backed_ml_rule_set()

    stacking = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(intermediates={"model_selection.use_stacking": True}),
    )
    assert {"collect_oof_predictions", "train_meta_learner", "meta_predict"}.issubset(
        {node.node_id for node in stacking.cdg.nodes}
    )

    constraint = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(
            intermediates={"model_selection.requires_constraint_injection": True}
        ),
    )
    assert {"verify_constraint", "apply_decorrelation"}.issubset(
        {node.node_id for node in constraint.cdg.nodes}
    )

    backbone = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(intermediates={"model_selection.use_pretrained_backbone": True}),
    )
    node_ids = {node.node_id for node in backbone.cdg.nodes}
    assert "feature_engineering" not in node_ids
    assert "model_training" not in node_ids
    assert {"load_pretrained", "finetune_backbone", "test_time_augmentation"}.issubset(
        node_ids
    )


def test_mined_ml_expansion_rules_apply_to_tabular_pipeline() -> None:
    rule_set = _asset_backed_ml_rule_set()

    tree_blend = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(
            intermediates={"model_selection.requires_tree_ensemble_blend": True}
        ),
    )
    assert {"train_lightgbm", "train_xgboost", "blend_tree_predictions"}.issubset(
        {node.node_id for node in tree_blend.cdg.nodes}
    )

    rfe = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(
            intermediates={
                "model_selection.requires_recursive_feature_elimination": True
            }
        ),
    )
    assert "recursive_feature_elimination" in {node.node_id for node in rfe.cdg.nodes}

    log_target = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(
            intermediates={"model_selection.requires_log_target_transform": True}
        ),
    )
    assert "log_target_transform" in {node.node_id for node in log_target.cdg.nodes}

    target_encoding = ExpansionEngine([rule_set]).expand(
        _tabular_pipeline_cdg(),
        ExpansionContext(intermediates={"model_selection.requires_target_encoding": True}),
    )
    assert "smoothed_target_encoding" in {
        node.node_id for node in target_encoding.cdg.nodes
    }
