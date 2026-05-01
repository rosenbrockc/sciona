from __future__ import annotations

from sciona.physics_ingest.retrieval import (
    SymbolicRetrievalQuery,
    build_symbolic_retrieval_report,
    candidates_from_rows,
    rank_symbolic_candidates,
    suggest_raw_candidate_external_knowledge,
)


def test_symbolic_ranker_prefers_reviewed_topology_dimension_and_mechanism_match() -> None:
    candidates = candidates_from_rows(
        [
            {
                "artifact_id": "raw",
                "fqdn": "physics.raw.force",
                "expression_id": "expr-raw",
                "topology_hash": "topo-force",
                "dimensional_hash": "dim-force",
                "dim_signatures": ["M1", "L1T-2", "M1L1T-2"],
                "mechanism_tags": ["transport"],
                "behavioral_archetypes": ["flow"],
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
            },
            {
                "artifact_id": "reviewed",
                "fqdn": "physics.reviewed.force",
                "expression_id": "expr-reviewed",
                "topology_hash": "topo-force",
                "dimensional_hash": "dim-force",
                "dim_signatures": ["M1", "L1T-2", "M1L1T-2"],
                "mechanism_tags": ["transport", "constitutive response"],
                "behavioral_archetypes": ["flow"],
                "review_status": "human_reviewed",
                "validation_status": "passed",
                "publish_status": "published",
                "is_publishable": True,
                "relationships": [
                    {
                        "relationship_kind": "uses_constant",
                        "verified": True,
                    }
                ],
                "validity_bounds": [
                    {
                        "variable_name": "m",
                        "lower_value": 0,
                        "review_status": "human_reviewed",
                    }
                ],
            },
        ]
    )

    results = rank_symbolic_candidates(
        SymbolicRetrievalQuery(
            topology_hashes=("topo-force",),
            dimensional_hashes=("dim-force",),
            dim_signatures=("M1", "L1T-2", "M1L1T-2"),
            mechanism_tags=("transport", "constitutive response"),
            behavioral_archetypes=("flow",),
            relationship_kinds=("uses_constant",),
            require_validity_bounds=True,
            require_reviewed_bounds=True,
        ),
        candidates,
    )

    assert [result.candidate.fqdn for result in results] == [
        "physics.reviewed.force",
        "physics.raw.force",
    ]
    winner = results[0]
    assert winner.eligible is True
    assert winner.components["topology_hash"] == 4.0
    assert winner.components["dimensional_hash"] == 2.0
    assert winner.components["mechanism_tags"] == 1.2
    assert winner.components["published"] == 2.0
    assert winner.components["reviewed_validity_bounds"] == 0.4
    assert "requested_relationships_verified" in winner.reasons

    raw = results[1]
    assert raw.eligible is False
    assert raw.score == 0.0
    assert "missing_required_validity_bounds" in raw.reasons
    assert "raw_penalty" in raw.reasons


def test_reviewed_only_trust_policy_excludes_raw_candidates() -> None:
    results = rank_symbolic_candidates(
        {
            "topology_hash": "same-topology",
            "raw_trust_policy": "reviewed_only",
        },
        [
            {
                "fqdn": "physics.raw",
                "topology_hash": "same-topology",
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
            },
            {
                "fqdn": "physics.reviewed",
                "topology_hash": "same-topology",
                "review_status": "human_reviewed",
            },
        ],
    )

    assert results[0].candidate.fqdn == "physics.reviewed"
    assert results[0].eligible is True
    assert results[1].candidate.fqdn == "physics.raw"
    assert results[1].eligible is False
    assert "raw_excluded_by_policy" in results[1].reasons


def test_allow_raw_policy_keeps_raw_candidate_score_when_no_reviewed_match_exists() -> None:
    results = rank_symbolic_candidates(
        SymbolicRetrievalQuery(
            topology_hashes=("topo",),
            raw_trust_policy="allow_raw",
        ),
        [
            {
                "fqdn": "physics.raw",
                "topology_hash": "topo",
                "review_status": "unreviewed",
            }
        ],
    )

    assert results[0].eligible is True
    assert results[0].score == 4.0
    assert "raw_penalty" not in results[0].reasons


def test_candidates_from_artifact_document_scopes_expression_metadata() -> None:
    document = {
        "artifact": {
            "artifact_id": "artifact-1",
            "fqdn": "physics.force_bundle",
            "artifact_kind": "physics_atom",
            "is_publishable": True,
        },
        "symbolic_expressions": [
            {
                "expression_id": "expr-force",
                "version_id": "v1",
                "topology_hash": "force-topology",
                "dimensional_hash": "force-dim",
                "mechanism_tags": ["transport"],
                "behavioral_archetypes": ["flow"],
                "review_status": "human_reviewed",
                "validation_status": "passed",
            },
            {
                "expression_id": "expr-energy",
                "version_id": "v1",
                "topology_hash": "energy-topology",
                "dimensional_hash": "energy-dim",
                "mechanism_tags": ["storage"],
                "review_status": "human_reviewed",
            },
        ],
        "symbolic_variables": [
            {
                "expression_id": "expr-force",
                "symbol_name": "F",
                "dim_signature": "M1L1T-2",
            },
            {
                "expression_id": "expr-energy",
                "symbol_name": "E",
                "dim_signature": "M1L2T-2",
            },
        ],
        "validity_bounds": [
            {
                "expression_id": "expr-force",
                "variable_name": "m",
                "lower_value": 0,
                "review_status": "human_reviewed",
            },
            {
                "expression_id": "expr-energy",
                "variable_name": "v",
                "lower_value": 0,
                "review_status": "human_reviewed",
            },
        ],
        "relationships": [
            {
                "source_expression_id": "expr-force",
                "relationship_kind": "derives_from",
                "verified": True,
            }
        ],
    }

    candidates = candidates_from_rows([document])
    results = rank_symbolic_candidates(
        SymbolicRetrievalQuery(
            topology_hashes=("force-topology",),
            dimensional_hashes=("force-dim",),
            dim_signatures=("M1L1T-2",),
            relationship_kinds=("derives_from",),
            require_reviewed_bounds=True,
        ),
        candidates,
    )

    assert len(candidates) == 2
    assert results[0].candidate.expression_id == "expr-force"
    assert results[0].candidate.dim_signatures == ("M1L1T-2",)
    assert [bound.variable_name for bound in results[0].candidate.validity_bounds] == ["m"]
    assert results[0].components["verified_relationships"] == 0.4
    assert results[1].candidate.expression_id == "expr-energy"


def test_blocked_candidate_is_ineligible_even_with_exact_hash_match() -> None:
    results = rank_symbolic_candidates(
        SymbolicRetrievalQuery(topology_hashes=("topo",)),
        [
            {
                "fqdn": "physics.blocked",
                "topology_hash": "topo",
                "review_status": "blocked",
                "validation_status": "passed",
            }
        ],
    )

    assert results[0].eligible is False
    assert results[0].score == 0.0
    assert "blocked_status" in results[0].reasons


def test_needs_human_candidate_is_explicitly_flagged_but_ranked() -> None:
    results = rank_symbolic_candidates(
        SymbolicRetrievalQuery(topology_hashes=("topo",)),
        [
            {
                "fqdn": "physics.needs_human",
                "topology_hash": "topo",
                "review_status": "needs_human",
                "validation_status": "passed",
            }
        ],
    )

    assert results[0].candidate.trust_status == "needs_human"
    assert results[0].eligible is True
    assert "needs_human_review" in results[0].reasons


def test_raw_candidate_external_knowledge_suggestions_are_side_effect_free() -> None:
    candidates = candidates_from_rows(
        [
            {
                "artifact_id": "raw-force",
                "fqdn": "physics.raw.force",
                "raw_formula": "F = m a",
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
                "mechanism_tags": ["transport"],
            },
            {
                "artifact_id": "reviewed-force",
                "fqdn": "physics.reviewed.force",
                "raw_formula": "F = m a",
                "review_status": "human_reviewed",
            },
        ]
    )

    suggestions = suggest_raw_candidate_external_knowledge(candidates)

    assert len(suggestions) == 1
    assert suggestions[0].candidate_key == "raw-force"
    assert suggestions[0].trust_status == "needs_human"
    assert suggestions[0].reason == "raw_candidate_needs_external_knowledge"
    assert suggestions[0].suggested_relationship_kinds == (
        "physical_grounding_of",
        "derives_from",
        "uses_constant",
    )
    assert suggestions[0].suggested_reference_queries == (
        "F = m a",
        "physics raw force",
        "transport",
    )


def test_symbolic_retrieval_report_includes_trust_and_raw_suggestions() -> None:
    report = build_symbolic_retrieval_report(
        {"topology_hash": "topo"},
        [
            {
                "artifact_id": "raw",
                "topology_hash": "topo",
                "review_status": "unreviewed",
                "raw_formula": "E = m c^2",
            }
        ],
    )

    assert report["result_count"] == 1
    assert report["results"][0]["trust_status"] == "needs_human"
    assert report["raw_candidate_external_knowledge_suggestions"][0]["candidate_key"] == "raw"
