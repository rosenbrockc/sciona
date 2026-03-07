from __future__ import annotations

from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from ageom.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from ageom.cli import _resolve_retrieval_policy
from ageom.config import AgeomConfig, resolve_execution_mode


def _seed_catalog() -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    seed_builtin_primitives(catalog)
    return catalog


def _seed_medium_catalog() -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    for name, description in [
        ("dijkstra", "Single-source shortest path in weighted graph"),
        ("bellman_ford", "Shortest path distances with negative edge support"),
        ("relax_path_edges", "Relax weighted graph edges to improve tentative distances"),
        ("build_shortest_path_tree", "Construct shortest path tree from predecessor map"),
    ]:
        catalog.add(
            AlgorithmicPrimitive(
                name=name,
                source="test",
                category=ConceptType.GRAPH_OPTIMIZATION,
                description=description,
                inputs=[IOSpec(name="graph", type_desc="Graph")],
                outputs=[IOSpec(name="distances", type_desc="dict[node, float]")],
            )
        )
    return catalog


def test_catalog_confidence_is_high_for_exact_builtin_alias_match():
    catalog = _seed_catalog()

    confidence = catalog.estimate_confidence("Apply Filter to ECG signal")

    assert confidence.score >= 0.70
    assert "apply_iir_filter" in confidence.exact_matches


def test_catalog_confidence_is_low_for_greenfield_text():
    catalog = _seed_catalog()

    confidence = catalog.estimate_confidence(
        "Invent a new social-choice consensus protocol for adversarial committees"
    )

    assert confidence.score < 0.40
    assert confidence.exact_matches == ()


def test_retrieval_policy_disables_heavy_retrieval_for_low_confidence_verified_mode():
    config = AgeomConfig()
    mode = resolve_execution_mode(config, "verified")
    catalog = _seed_catalog()

    policy = _resolve_retrieval_policy(
        mode_settings=mode,
        catalog=catalog,
        texts=["Invent a new social-choice consensus protocol for adversarial committees"],
    )

    assert policy.confidence_band == "low"
    assert policy.skill_index_enabled is False
    assert policy.graph_retrieval_enabled is False
    assert policy.semantic_index_backend_override == "lexical"
    assert policy.hunter_mode == "standard"


def test_retrieval_policy_keeps_heavier_retrieval_for_high_confidence_verified_mode(
    monkeypatch,
):
    monkeypatch.setenv("AGEOM_GRAPH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("AGEOM_HUNTER_MODE", "speculative_local")
    config = AgeomConfig()
    mode = resolve_execution_mode(config, "verified")
    catalog = _seed_catalog()

    policy = _resolve_retrieval_policy(
        mode_settings=mode,
        catalog=catalog,
        texts=["Apply Filter to ECG signal"],
    )

    assert policy.confidence_band == "high"
    assert policy.skill_index_enabled is True
    assert policy.graph_retrieval_enabled is True
    assert policy.semantic_index_backend_override is None
    assert policy.hunter_mode == "speculative_local"


def test_retrieval_policy_uses_lexical_without_disabling_skill_index_for_medium_confidence():
    config = AgeomConfig()
    mode = resolve_execution_mode(config, "structured")
    catalog = _seed_medium_catalog()

    policy = _resolve_retrieval_policy(
        mode_settings=mode,
        catalog=catalog,
        texts=["Find shortest path distances from a source node in a weighted graph"],
    )

    assert policy.confidence_band == "medium"
    assert policy.skill_index_enabled is True
    assert policy.graph_retrieval_enabled is False
    assert policy.semantic_index_backend_override == "lexical"
    assert policy.hunter_mode == "standard"
