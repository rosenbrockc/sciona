from __future__ import annotations

import json

from sciona.physics_ingest.retrieval import (
    SymbolicArtifactCandidate,
    SymbolicRetrievalQuery,
    SymbolicValidityBound,
    build_symbolic_retrieval_report,
    build_symbolic_synthesis_retrieval_report,
    candidates_from_rows,
    rank_symbolic_candidates,
    suggest_raw_candidate_external_knowledge,
)
from sciona.physics_ingest.sources.opb import build_opb_wave0_bundle


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


def test_symbolic_ranker_scores_requested_validity_regime_and_bounds() -> None:
    results = rank_symbolic_candidates(
        SymbolicRetrievalQuery(
            topology_hashes=("topo-drag",),
            validity_regimes=("low_reynolds",),
            validity_variables=("Re",),
            validity_bounds=(
                SymbolicValidityBound(
                    variable_name="Re",
                    lower_value=0,
                    upper_value=0.5,
                ),
            ),
            require_validity_matches=True,
        ),
        [
            {
                "artifact_id": "high-re",
                "fqdn": "physics.drag.turbulent",
                "topology_hash": "topo-drag",
                "review_status": "human_reviewed",
                "validity_bounds": [
                    {
                        "variable_name": "Re",
                        "regime_label": "high_reynolds",
                        "lower_value": 1000,
                        "review_status": "human_reviewed",
                    }
                ],
            },
            {
                "artifact_id": "low-re",
                "fqdn": "physics.drag.stokes",
                "topology_hash": "topo-drag",
                "review_status": "human_reviewed",
                "validity_bounds": [
                    {
                        "variable_name": "Re",
                        "regime_label": "low_reynolds",
                        "lower_value": 0,
                        "upper_value": 1,
                        "review_status": "human_reviewed",
                    }
                ],
            },
        ],
    )

    winner = results[0]
    assert winner.candidate.fqdn == "physics.drag.stokes"
    assert winner.eligible is True
    assert winner.components["validity_regimes"] == 0.5
    assert winner.components["validity_variables"] == 0.4
    assert winner.components["validity_bound_ranges"] == 0.8
    assert "requested_validity_bounds_match" in winner.reasons

    miss = results[1]
    assert miss.candidate.fqdn == "physics.drag.turbulent"
    assert miss.eligible is False
    assert "requested_validity_regime_missing" in miss.reasons
    assert "requested_validity_bounds_missing" in miss.reasons
    assert "missing_required_validity_match" in miss.reasons


def test_symbolic_validity_queries_accept_mapping_aliases() -> None:
    results = rank_symbolic_candidates(
        {
            "topology_hash": "topo-pendulum",
            "regime_label": "small_angle",
            "validity_bounds": [
                {
                    "variable_name": "theta",
                    "upper_bound": 0.05,
                }
            ],
        },
        [
            {
                "artifact_id": "pendulum-small-angle",
                "fqdn": "physics.pendulum.small_angle",
                "topology_hash": "topo-pendulum",
                "review_status": "human_reviewed",
                "validity_bounds": [
                    {
                        "variable": "theta",
                        "regime": "small-angle",
                        "min_value": 0,
                        "max_value": 0.1,
                        "review_status": "human_reviewed",
                    }
                ],
            }
        ],
    )

    assert results[0].eligible is True
    assert results[0].components["validity_regimes"] == 0.5
    assert results[0].components["validity_bound_ranges"] == 0.8
    assert results[0].candidate.validity_bounds[0].lower_value == 0.0


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


def test_raw_candidate_external_knowledge_suggests_data_validity_and_reference_work() -> None:
    candidates = candidates_from_rows(
        [
            {
                "artifact_id": "raw-force-data",
                "fqdn": "physics.raw.force.data",
                "raw_formula": "F = m a",
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
                "source_payload": {
                    "future_data_artifact": {"artifact_id": "opb.record.opb.newton-2"}
                },
            }
        ]
    )
    query = SymbolicRetrievalQuery.from_mapping(
        {
            "data_artifact_dependencies": ["opb.record.opb.newton-2"],
            "require_data_artifact_dependencies": True,
            "require_validity_bounds": True,
            "validity_variables": ["t"],
        }
    )

    suggestions = suggest_raw_candidate_external_knowledge(candidates, query)

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.suggested_relationship_kinds == (
        "physical_grounding_of",
        "derives_from",
        "uses_constant",
        "uses_data_artifact",
    )
    assert suggestion.suggested_reference_queries == (
        "F = m a",
        "physics raw force data",
        "opb.record.opb.newton-2",
    )
    assert suggestion.suggested_source_systems == (
        "manual",
        "physics_derivation_graph",
        "theoria",
        "nist_dlmf",
        "opb",
        "materials_project",
        "hitran",
        "phy_srbench",
    )
    assert suggestion.suggested_review_tasks == (
        "locate_primary_source_reference",
        "record_source_provenance",
        "define_validity_regime_and_bounds",
        "check_requested_validity_bounds_or_regime",
        "verify_data_artifact_dependency_provenance",
        "add_uses_data_artifact_relationship",
        "classify_mechanism_metadata",
    )
    assert suggestion.to_dict()["suggested_review_tasks"] == list(
        suggestion.suggested_review_tasks
    )


def test_raw_candidate_external_knowledge_distinguishes_provenance_from_reference() -> None:
    candidates = candidates_from_rows(
        [
            {
                "artifact_id": "raw-adapter-force",
                "fqdn": "physics.raw.adapter.force",
                "raw_formula": "F = m a",
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
                "source_system": "web_seed",
                "source_kind": "raw_adapter",
            }
        ]
    )

    suggestions = suggest_raw_candidate_external_knowledge(candidates)

    assert suggestions[0].suggested_review_tasks == (
        "locate_primary_source_reference",
        "define_validity_regime_and_bounds",
        "classify_mechanism_metadata",
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


def test_symbolic_retrieval_report_includes_json_safe_dashboard_summary() -> None:
    report = build_symbolic_retrieval_report(
        {
            "topology_hash": "topo-dashboard",
            "data_artifact_dependencies": ["data-1"],
            "require_data_artifact_dependencies": True,
        },
        [
            {
                "artifact_id": "reviewed-data",
                "topology_hash": "topo-dashboard",
                "dimensional_hash": "dim-dashboard",
                "mechanism_tags": ["dispersion"],
                "source_system": "theoria",
                "source_kind": "curated_publication",
                "source_domains": ["fluid_dynamics"],
                "data_artifact_dependencies": ["data-1"],
                "relationships": [
                    {
                        "relationship_kind": "uses_data_artifact",
                        "verified": True,
                    }
                ],
                "review_status": "human_reviewed",
            },
            {
                "artifact_id": "raw-no-data",
                "topology_hash": "topo-dashboard",
                "source_system": "web_seed",
                "source_kind": "raw_adapter",
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
            },
            {
                "artifact_id": "blocked-no-data",
                "topology_hash": "topo-dashboard",
                "relationships": [{"relationship_kind": "derives_from"}],
                "review_status": "blocked",
            },
        ],
    )

    summary = report["dashboard_summary"]
    assert json.loads(json.dumps(summary)) == summary
    assert summary == {
        "result_count": 3,
        "eligibility_counts": {"eligible": 1, "ineligible": 2},
        "trust_status_counts": {
            "blocked": 1,
            "human_reviewed": 1,
            "needs_human": 1,
        },
        "source_system_counts": {
            "<missing>": 1,
            "theoria": 1,
            "web_seed": 1,
        },
        "source_kind_counts": {
            "<missing>": 1,
            "curated_publication": 1,
            "raw_adapter": 1,
        },
        "presence_counts": {
            "mechanism": {"present": 1, "missing": 2},
            "source_domain": {"present": 1, "missing": 2},
            "data_artifact": {"present": 1, "missing": 2},
            "relationship": {"present": 2, "missing": 1},
        },
        "blocker_counts": {
            "blocked_status": 1,
            "missing_required_data_artifact_dependencies": 2,
        },
    }
    query_coverage = report["query_coverage_summary"]
    assert json.loads(json.dumps(query_coverage)) == query_coverage
    assert query_coverage == {
        "requested": {
            "topology_hash": True,
            "dimensional_hash": False,
            "dim_signature": False,
            "mechanism": False,
            "behavioral_archetype": False,
            "relationship": False,
            "validity": False,
            "source": False,
            "known_analogue": False,
            "data_artifact": True,
        },
        "candidate_match_counts": {
            "topology_hash": 3,
            "dimensional_hash": 0,
            "dim_signature": 0,
            "mechanism": 0,
            "behavioral_archetype": 0,
            "relationship": 0,
            "validity": 0,
            "source": 0,
            "known_analogue": 0,
            "data_artifact": 1,
        },
        "requested_feature_count": 2,
        "matched_requested_feature_count": 2,
        "unmatched_requested_features": [],
    }


def test_symbolic_synthesis_report_separates_executable_and_external_knowledge() -> None:
    report = build_symbolic_synthesis_retrieval_report(
        {
            "topology_hash": "topo-wave",
            "dimensional_hash": "dim-wave",
            "dim_signatures": ["L1", "T-1"],
            "mechanism_tags": ["dispersion"],
            "relationship_kinds": ["derives_from"],
            "source_system": "theoria",
            "source_kind": "curated_publication",
            "require_reviewed_bounds": True,
        },
        [
            {
                "artifact_id": "reviewed-wave",
                "expression_id": "expr-reviewed-wave",
                "fqdn": "physics.wave.reviewed",
                "raw_formula": "v = f lambda",
                "topology_hash": "topo-wave",
                "dimensional_hash": "dim-wave",
                "dim_signatures": ["L1", "T-1"],
                "mechanism_tags": ["dispersion"],
                "source_system": "theoria",
                "source_kind": "curated_publication",
                "review_status": "human_reviewed",
                "validation_status": "passed",
                "publish_status": "published",
                "relationships": [
                    {
                        "relationship_kind": "derives_from",
                        "relationship_label": "wave identity",
                        "confidence": 0.9,
                        "verified": True,
                        "source_kind": "publication_edge",
                    }
                ],
                "validity_bounds": [
                    {
                        "variable_name": "f",
                        "lower_value": 0,
                        "review_status": "human_reviewed",
                    }
                ],
            },
            {
                "artifact_id": "raw-wave",
                "fqdn": "physics.wave.raw",
                "raw_formula": "v = f lambda",
                "topology_hash": "topo-wave",
                "dimensional_hash": "dim-wave",
                "dim_signatures": ["L1", "T-1"],
                "mechanism_tags": ["dispersion"],
                "source_system": "web_seed",
                "source_kind": "raw_adapter",
                "review_status": "unreviewed",
                "candidate_status": "raw_imported",
            },
        ],
    )

    assert report["report_kind"] == "symbolic_synthesis_retrieval"
    assert report["executable_candidate_count"] == 1
    assert report["external_knowledge_suggestion_count"] == 1
    assert json.loads(json.dumps(report["dashboard_summary"])) == report[
        "dashboard_summary"
    ]
    assert report["dashboard_summary"]["synthesis_candidate_counts"] == {
        "executable": 1,
        "external": 1,
        "blocked": 0,
    }
    assert report["dashboard_summary"]["blocker_counts"] == {
        "missing_reviewed_validity_bounds": 1,
        "not_published_or_reviewed": 1,
    }
    assert report["query_coverage_summary"] == {
        "requested": {
            "topology_hash": True,
            "dimensional_hash": True,
            "dim_signature": True,
            "mechanism": True,
            "behavioral_archetype": False,
            "relationship": True,
            "validity": True,
            "source": True,
            "known_analogue": False,
            "data_artifact": False,
        },
        "candidate_match_counts": {
            "topology_hash": 2,
            "dimensional_hash": 2,
            "dim_signature": 2,
            "mechanism": 2,
            "behavioral_archetype": 0,
            "relationship": 1,
            "validity": 1,
            "source": 1,
            "known_analogue": 0,
            "data_artifact": 0,
        },
        "requested_feature_count": 7,
        "matched_requested_feature_count": 7,
        "unmatched_requested_features": [],
    }

    executable = report["executable_candidates"][0]
    assert executable["candidate_key"] == "expr-reviewed-wave"
    assert executable["score_components"]["source_system"] == 0.5
    assert executable["score_components"]["source_kind"] == 0.3
    assert executable["score_components"]["provenance_present"] == 0.2
    assert executable["compiler_contract"]["can_compile"] is True
    assert executable["compiler_contract"]["blockers"] == []
    assert "verify_candidate_dimensional_hash" in executable["compiler_contract"][
        "required_dimensional_checks"
    ]
    assert executable["relationship_edges"] == [
        {
            "relationship_kind": "derives_from",
            "relationship_label": "wave identity",
            "confidence": 0.9,
            "verified": True,
            "source_kind": "publication_edge",
        }
    ]

    suggestion = report["external_knowledge_suggestions"][0]
    assert suggestion["candidate_key"] == "raw-wave"
    assert suggestion["suggestion"]["reason"] == "raw_candidate_needs_external_knowledge"
    assert suggestion["compiler_contract"]["can_compile"] is False
    assert "not_published_or_reviewed" in suggestion["compiler_contract"]["blockers"]
    assert suggestion["dimensions"]["dimensionally_usable"] is True


def test_symbolic_synthesis_report_blocks_reviewed_candidate_without_dimensions() -> None:
    report = build_symbolic_synthesis_retrieval_report(
        {"topology_hash": "topo"},
        [
            {
                "artifact_id": "reviewed-no-dimensions",
                "topology_hash": "topo",
                "review_status": "human_reviewed",
                "validation_status": "passed",
            }
        ],
    )

    assert report["executable_candidates"] == []
    blocked = report["blocked_candidates"][0]
    assert blocked["candidate_key"] == "reviewed-no-dimensions"
    assert blocked["compiler_contract"]["can_compile"] is False
    assert blocked["compiler_contract"]["blockers"] == ["missing_dimensional_metadata"]


def test_symbolic_synthesis_report_exposes_unverified_relationship_diagnostics() -> None:
    report = build_symbolic_synthesis_retrieval_report(
        {
            "topology_hash": "topo-wave",
            "relationship_kind": "derived from",
            "relationship_label": "wave identity",
        },
        [
            {
                "artifact_id": "reviewed-wave-missing-dimensions",
                "topology_hash": "topo-wave",
                "review_status": "human_reviewed",
                "relationships": [
                    {
                        "relationship_kind": "derives-from",
                        "relationship_label": "Wave Identity",
                        "confidence": 0.75,
                        "verified": False,
                        "source_kind": "adapter_edge",
                    }
                ],
            }
        ],
    )

    assert report["executable_candidates"] == []
    blocked = report["blocked_candidates"][0]
    assert blocked["candidate_key"] == "reviewed-wave-missing-dimensions"
    assert blocked["compiler_contract"]["blockers"] == ["missing_dimensional_metadata"]
    diagnostics = blocked["compiler_contract"]["relationship_request_diagnostics"]
    assert diagnostics["requested"] == ["derives_from", "wave identity"]
    assert diagnostics["verified"] == []
    assert diagnostics["unverified"] == ["derives_from", "wave identity"]
    assert diagnostics["missing"] == []
    assert diagnostics["matched_edges"] == [
        {
            "requested_relationship": "derives_from",
            "relationship_kind": "derives_from",
            "relationship_label": "Wave Identity",
            "confidence": 0.75,
            "verified": False,
            "source_kind": "adapter_edge",
        },
        {
            "requested_relationship": "wave identity",
            "relationship_kind": "derives_from",
            "relationship_label": "Wave Identity",
            "confidence": 0.75,
            "verified": False,
            "source_kind": "adapter_edge",
        },
    ]


def test_symbolic_synthesis_report_exposes_missing_required_validity_diagnostics() -> None:
    report = build_symbolic_synthesis_retrieval_report(
        {
            "topology_hash": "topo-validity",
            "validity_regimes": ["small_angle"],
            "validity_variables": ["theta"],
            "requested_validity_bounds": [
                {"variable_name": "theta", "lower_value": 0.0, "upper_value": 0.1}
            ],
            "require_validity_matches": True,
        },
        [
            {
                "artifact_id": "reviewed-large-angle",
                "topology_hash": "topo-validity",
                "dimensional_hash": "dim-validity",
                "review_status": "human_reviewed",
                "validity_bounds": [
                    {
                        "variable_name": "theta",
                        "regime_label": "large_angle",
                        "lower_value": 0.5,
                        "upper_value": 1.0,
                        "review_status": "human_reviewed",
                    }
                ],
            }
        ],
    )

    assert report["executable_candidates"] == []
    blocked = report["blocked_candidates"][0]
    assert blocked["candidate_key"] == "reviewed-large-angle"
    assert blocked["compiler_contract"]["blockers"] == [
        "missing_required_validity_match"
    ]
    assert blocked["compiler_contract"]["can_compile"] is False

    diagnostics = blocked["compiler_contract"]["validity_request_diagnostics"]
    assert diagnostics["requested"]["regimes"] == ["small_angle"]
    assert diagnostics["requested"]["variables"] == ["theta"]
    assert diagnostics["requested"]["bounds"] == [
        {
            "variable_name": "theta",
            "regime_label": "",
            "validity_statement": "",
            "lower_value": 0.0,
            "upper_value": 0.1,
            "review_status": "",
            "reviewed": False,
        }
    ]
    assert diagnostics["available_bound_count"] == 1
    assert diagnostics["reviewed_bound_count"] == 1
    assert diagnostics["unreviewed_bound_count"] == 0
    assert diagnostics["matched"]["regimes"] == []
    assert diagnostics["matched"]["variables"] == ["theta"]
    assert diagnostics["matched"]["bounds"] == []
    assert diagnostics["missing"]["regimes"] == ["small_angle"]
    assert diagnostics["missing"]["variables"] == []
    assert diagnostics["missing"]["bounds"] == diagnostics["requested"]["bounds"]


def test_symbolic_synthesis_report_exposes_partial_validity_diagnostics() -> None:
    report = build_symbolic_synthesis_retrieval_report(
        {
            "topology_hash": "topo-validity-partial",
            "validity_regimes": ["small_angle", "long_wave"],
            "validity_variables": ["theta", "omega"],
            "requested_validity_bounds": [
                {"variable_name": "theta", "upper_value": 0.1},
                {"variable_name": "omega", "lower_value": 1.0, "upper_value": 2.0},
            ],
        },
        [
            {
                "artifact_id": "reviewed-partial-validity",
                "topology_hash": "topo-validity-partial",
                "dimensional_hash": "dim-validity",
                "review_status": "human_reviewed",
                "validity_bounds": [
                    {
                        "variable_name": "theta",
                        "regime_label": "small_angle",
                        "lower_value": 0.0,
                        "upper_value": 0.2,
                        "review_status": "human_reviewed",
                    },
                    {
                        "variable_name": "k",
                        "regime_label": "long_wave",
                        "lower_value": 0.0,
                        "review_status": "unreviewed",
                    },
                ],
            }
        ],
    )

    assert report["executable_candidate_count"] == 1
    executable = report["executable_candidates"][0]
    assert executable["compiler_contract"]["can_compile"] is True
    assert executable["compiler_contract"]["blockers"] == []

    diagnostics = executable["compiler_contract"]["validity_request_diagnostics"]
    assert diagnostics["available_bound_count"] == 2
    assert diagnostics["reviewed_bound_count"] == 1
    assert diagnostics["unreviewed_bound_count"] == 1
    assert diagnostics["matched"]["regimes"] == ["small_angle", "long_wave"]
    assert diagnostics["matched"]["variables"] == ["theta"]
    assert diagnostics["matched"]["bounds"] == [
        {
            "requested_bound": {
                "variable_name": "theta",
                "regime_label": "",
                "validity_statement": "",
                "lower_value": None,
                "upper_value": 0.1,
                "review_status": "",
                "reviewed": False,
            },
            "available_bound": {
                "variable_name": "theta",
                "regime_label": "small_angle",
                "validity_statement": "",
                "lower_value": 0.0,
                "upper_value": 0.2,
                "review_status": "human_reviewed",
                "reviewed": True,
            },
        }
    ]
    assert diagnostics["missing"]["regimes"] == []
    assert diagnostics["missing"]["variables"] == ["omega"]
    assert diagnostics["missing"]["bounds"] == [
        {
            "variable_name": "omega",
            "regime_label": "",
            "validity_statement": "",
            "lower_value": 1.0,
            "upper_value": 2.0,
            "review_status": "",
            "reviewed": False,
        }
    ]


def test_symbolic_ranker_scores_source_domain_analogues_and_data_artifacts() -> None:
    results = rank_symbolic_candidates(
        {
            "topology_hash": "topo-fluid",
            "source_domain": "fluid_dynamics",
            "known_analogues": ["navier-stokes"],
            "data_artifact_dependencies": ["opb-wave-benchmark"],
        },
        [
            {
                "artifact_id": "without-phase6-metadata",
                "fqdn": "physics.fluid.generic",
                "topology_hash": "topo-fluid",
                "dimensional_hash": "dim-fluid",
                "review_status": "human_reviewed",
            },
            {
                "artifact_id": "with-phase6-metadata",
                "fqdn": "physics.fluid.benchmark_backed",
                "topology_hash": "topo-fluid",
                "dimensional_hash": "dim-fluid",
                "source_domains": ["fluid_dynamics", "continuum_mechanics"],
                "known_analogues": [{"artifact_fqdn": "navier-stokes"}],
                "data_artifact_dependencies": [
                    {"artifact_key": "opb-wave-benchmark"}
                ],
                "review_status": "human_reviewed",
                "validation_status": "passed",
            },
        ],
    )

    assert results[0].candidate.artifact_id == "with-phase6-metadata"
    assert results[0].components["source_domains"] == 0.4
    assert results[0].components["known_analogues"] == 0.7
    assert results[0].components["data_artifact_dependencies"] == 0.7
    assert "source_domain_overlap" in results[0].reasons
    assert "known_analogue_overlap" in results[0].reasons
    assert "data_artifact_dependency_overlap" in results[0].reasons


def test_source_domain_ranking_normalizes_label_variants_and_nested_payloads() -> None:
    results = rank_symbolic_candidates(
        {
            "topology_hash": "topo-domain",
            "source_domain": "fluid dynamics",
        },
        [
            {
                "artifact_id": "materials-candidate",
                "topology_hash": "topo-domain",
                "source_payload": {"physics_domain": "materials_science"},
                "review_status": "human_reviewed",
            },
            {
                "artifact_id": "fluid-candidate",
                "topology_hash": "topo-domain",
                "source_payload": {"physics_domain": "Fluid-Dynamics"},
                "review_status": "human_reviewed",
            },
        ],
    )

    assert results[0].candidate.artifact_id == "fluid-candidate"
    assert results[0].candidate.source_domains == ("Fluid-Dynamics",)
    assert results[0].components["source_domains"] == 0.4
    assert "source_domain_overlap" in results[0].reasons
    assert "source_domains" not in results[1].components


def test_phase6_reference_ranking_normalizes_analogue_and_data_artifact_aliases() -> None:
    results = rank_symbolic_candidates(
        {
            "topology_hash": "topo-reference",
            "known_analogues": [{"artifact_id": "navier_stokes"}],
            "data_artifact_dependencies": [
                {"artifact_id": "opb.record.wave-benchmark"}
            ],
            "require_data_artifact_dependencies": True,
        },
        [
            {
                "artifact_id": "no-reference-aliases",
                "topology_hash": "topo-reference",
                "review_status": "human_reviewed",
            },
            {
                "artifact_id": "reference-aliases",
                "topology_hash": "topo-reference",
                "relationships": [
                    {
                        "relationship_kind": "mechanism_analogue_of",
                        "target_artifact_label": "Navier-Stokes",
                        "verified": True,
                    }
                ],
                "source_payload": {
                    "future_data_artifact": {
                        "artifact_label": "OPB Record Wave Benchmark"
                    }
                },
                "review_status": "human_reviewed",
            },
        ],
    )

    assert results[0].candidate.artifact_id == "reference-aliases"
    assert results[0].candidate.known_analogues == ("Navier-Stokes",)
    assert results[0].candidate.data_artifact_dependencies == (
        "OPB Record Wave Benchmark",
    )
    assert results[0].components["known_analogues"] == 0.7
    assert results[0].components["data_artifact_dependencies"] == 0.7
    assert "known_analogue_overlap" in results[0].reasons
    assert "data_artifact_dependency_overlap" in results[0].reasons

    assert results[1].candidate.artifact_id == "no-reference-aliases"
    assert results[1].eligible is False
    assert "missing_required_data_artifact_dependencies" in results[1].reasons


def test_phase6_relationship_matching_accepts_aliases_labels_and_unverified_edges() -> None:
    results = rank_symbolic_candidates(
        {
            "topology_hash": "topo-relationship",
            "relationship_kind": "derived from",
            "relationship_label": "wave identity",
        },
        [
            {
                "artifact_id": "missing-relationship",
                "topology_hash": "topo-relationship",
                "review_status": "human_reviewed",
            },
            {
                "artifact_id": "unverified-relationship",
                "topology_hash": "topo-relationship",
                "relationships": [
                    {
                        "relationship_kind": "derives-from",
                        "relationship_label": "Wave Identity",
                        "verified": False,
                    }
                ],
                "review_status": "human_reviewed",
            },
        ],
    )

    winner = results[0]
    assert winner.candidate.artifact_id == "unverified-relationship"
    assert winner.candidate.relationships[0].relationship_kind == "derives_from"
    assert winner.components["relationship_kinds"] == 0.8
    assert winner.components["unverified_requested_relationships"] == 0.2
    assert "relationship_kind_overlap" in winner.reasons
    assert "requested_relationships_unverified" in winner.reasons

    miss = results[1]
    assert "requested_relationships_missing" in miss.reasons


def test_symbolic_synthesis_report_can_require_data_artifact_dependencies() -> None:
    report = build_symbolic_synthesis_retrieval_report(
        {
            "topology_hash": "topo-benchmark",
            "data_artifact_dependencies": ["artifact-dataset-1"],
            "require_data_artifact_dependencies": True,
        },
        [
            {
                "artifact_id": "reviewed-with-data",
                "topology_hash": "topo-benchmark",
                "dimensional_hash": "dim-benchmark",
                "source_domains": "materials,benchmarks",
                "data_artifacts": [{"artifact_id": "artifact-dataset-1"}],
                "relationships": [
                    {
                        "relationship_kind": "mechanism_analogue_of",
                        "relationship_label": "spring_mass_reference",
                        "verified": True,
                    }
                ],
                "review_status": "human_reviewed",
            },
            {
                "artifact_id": "reviewed-without-data",
                "topology_hash": "topo-benchmark",
                "dimensional_hash": "dim-benchmark",
                "review_status": "human_reviewed",
            },
        ],
    )

    assert report["executable_candidate_count"] == 1
    executable = report["executable_candidates"][0]
    assert executable["candidate_key"] == "reviewed-with-data"
    assert executable["data_artifact_dependencies"] == ["artifact-dataset-1"]
    assert executable["known_analogues"] == ["spring_mass_reference"]
    assert executable["provenance"]["source_domains"] == ["materials", "benchmarks"]

    blocked = report["blocked_candidates"][0]
    assert blocked["candidate_key"] == "reviewed-without-data"
    assert "missing_required_data_artifact_dependencies" in blocked[
        "compiler_contract"
    ]["blockers"]


def test_adapter_source_payload_future_data_artifact_satisfies_dependency_requirement() -> None:
    bundle = build_opb_wave0_bundle(
        [
            {
                "problem_id": "opb:newton-2",
                "title": "Newton second law",
                "latex": "F = m a",
                "data": {"fixture_rows": [{"m": 2, "a": 3, "F": 6}]},
            }
        ],
        source_version="OPB fixture",
        snapshot_id="00000000-0000-0000-0000-000000000030",
    )
    adapter_row = {
        **bundle.candidate_rows[0],
        "artifact_id": "opb-newton-2",
        "dimensional_hash": "dim-force",
        "review_status": "human_reviewed",
    }

    candidate = SymbolicArtifactCandidate.from_catalog_row(adapter_row)

    assert candidate.data_artifact_dependencies == ("opb.record.opb.newton-2",)

    report = build_symbolic_synthesis_retrieval_report(
        {
            "data_artifact_dependencies": ["opb.record.opb.newton-2"],
            "require_data_artifact_dependencies": True,
        },
        [adapter_row],
    )

    assert report["executable_candidate_count"] == 1
    executable = report["executable_candidates"][0]
    assert executable["candidate_key"] == "opb-newton-2"
    assert executable["data_artifact_dependencies"] == [
        "opb.record.opb.newton-2"
    ]
    assert "missing_required_data_artifact_dependencies" not in executable[
        "compiler_contract"
    ]["blockers"]
