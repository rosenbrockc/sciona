"""Tests for baseline-core scoring graph assembly."""

from __future__ import annotations

from sciona.architect.models import ConceptType
from sciona.architect.skeletons import get_skeleton, instantiate_baseline_scoring


def _baseline_scoring_skeleton():
    skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS, variant="baseline_scoring")
    assert skeleton is not None
    return skeleton


def _edge_pairs(nodes, edges):
    id_to_name = {node.node_id: node.name for node in nodes}
    return {(id_to_name[edge.source_id], id_to_name[edge.target_id]) for edge in edges}


class TestBaselineScoringAssembly:
    def test_named_variant_is_registered(self):
        skeleton = _baseline_scoring_skeleton()

        assert skeleton.name == "Baseline Scoring"
        assert len(skeleton.template_nodes) == 15
        assert len(skeleton.template_edges) == 20

    def test_wires_analyzer_alias_inputs_into_scores(self):
        nodes, edges = instantiate_baseline_scoring("AHI baseline core scoring")

        names = [node.name for node in nodes]
        assert names.count("Analyzer Output: sqi_events") == 1
        assert names.count("Analyzer Output: sqi_probability") == 1
        assert names.count("Analyzer Output: combined_events") == 1
        assert names.count("Analyzer Output: pat_events") == 1
        assert names.count("Analyzer Output: pat_probability") == 1
        assert names.count("Analyzer Output: spo2_probability") == 1
        assert names.count("Analyzer Anchor") == 1
        assert names.count("Analyzer Sleep Mask") == 1
        assert names.count("Analyzer BMI") == 1

        assert names.count("Score sAHI") == 1
        assert names.count("Score bAHI") == 1
        assert names.count("Score pAHI") == 1

        pairs = _edge_pairs(nodes, edges)
        assert ("Analyzer Sleep Mask", "Compute Analyzed Sleep Time") in pairs
        assert ("Analyzer Anchor", "Compute Analyzed Sleep Time") in pairs
        assert ("Analyzer Output: sqi_probability", "Compute SQI Density") in pairs
        assert ("Analyzer Output: pat_probability", "Compute PAT Density") in pairs
        assert ("Analyzer Output: sqi_events", "Score sAHI") in pairs
        assert ("Analyzer Output: combined_events", "Score sAHI") in pairs
        assert ("Analyzer Output: spo2_probability", "Score sAHI") in pairs
        assert ("Analyzer Output: sqi_events", "Score bAHI") in pairs
        assert ("Analyzer Output: combined_events", "Score bAHI") in pairs
        assert ("Analyzer Output: spo2_probability", "Score bAHI") in pairs
        assert ("Analyzer BMI", "Score bAHI") in pairs
        assert ("Analyzer Output: pat_events", "Score pAHI") in pairs

    def test_score_nodes_expose_baseline_result_aliases(self):
        nodes, _edges = instantiate_baseline_scoring("AHI baseline core scoring")

        by_name = {node.name: node for node in nodes}
        assert by_name["Score sAHI"].outputs[0].name == "sAHI"
        assert by_name["Score bAHI"].outputs[0].name == "bAHI"
        assert by_name["Score pAHI"].outputs[0].name == "pAHI"

    def test_pat_branch_uses_its_own_density_node(self):
        nodes, edges = instantiate_baseline_scoring("AHI baseline core scoring")

        pairs = _edge_pairs(nodes, edges)
        assert ("Analyzer Output: pat_probability", "Compute PAT Density") in pairs
        assert ("Compute PAT Density", "Score pAHI") in pairs
        assert ("Compute SQI Density", "Score pAHI") not in pairs
