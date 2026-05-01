from __future__ import annotations

from sciona.physics_ingest.retrieval import (
    SymbolicRetrievalQuery,
    build_symbolic_retrieval_report,
    build_symbolic_synthesis_retrieval_report,
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
