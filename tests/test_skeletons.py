"""Tests for sciona.architect.skeletons — paradigm templates."""

import pytest

from sciona.architect.models import (
    ConceptType,
    NodeStatus,
)
from sciona.architect.skeletons import (
    SKELETON_TEMPLATES,
    get_skeleton,
    instantiate_skeleton,
)


class TestSkeletonRegistry:
    def test_all_templates_present(self):
        expected = {
            ConceptType.DIVIDE_AND_CONQUER,
            ConceptType.DYNAMIC_PROGRAMMING,
            ConceptType.GREEDY,
            ConceptType.GRAPH_TRAVERSAL,
            ConceptType.GRAPH_OPTIMIZATION,
            ConceptType.SORTING,
            ConceptType.STRING_MATCHING,
            ConceptType.SEARCHING,
            ConceptType.GEOMETRY,
            ConceptType.NUMBER_THEORY,
            ConceptType.SIGNAL_TRANSFORM,
            ConceptType.SIGNAL_FILTER,
            ConceptType.GRAPH_SIGNAL_PROCESSING,
            ConceptType.MCMC_KERNEL,
            ConceptType.VI_ELBO,
            ConceptType.SEQUENTIAL_FILTER,
            ConceptType.MESSAGE_PASSING,
            ConceptType.ALGEBRA,
            ConceptType.OPTIMIZATION,
            ConceptType.COMBINATORICS,
            ConceptType.NEURAL_NETWORK,
            ConceptType.CLUSTERING,
            ConceptType.DIMENSIONALITY_REDUCTION,
            ConceptType.ODE_SOLVER,
            ConceptType.QUADRATURE,
            ConceptType.RANDOMIZED,
            ConceptType.INFORMATION_THEORY,
            ConceptType.COMPRESSION,
            ConceptType.FIXED_POINT,
            ConceptType.MAP_OVER,
            ConceptType.BASELINE_ANALYSIS,
        }
        assert set(SKELETON_TEMPLATES.keys()) == expected

    def test_get_skeleton_exists(self):
        skel = get_skeleton(ConceptType.DIVIDE_AND_CONQUER)
        assert skel is not None
        assert skel.paradigm == ConceptType.DIVIDE_AND_CONQUER

    def test_get_skeleton_missing(self):
        assert get_skeleton(ConceptType.CUSTOM) is None


class TestSkeletonWellFormedness:
    """Every template must be structurally valid."""

    @pytest.mark.parametrize("concept_type", list(SKELETON_TEMPLATES.keys()))
    def test_has_nodes_and_edges(self, concept_type):
        skel = SKELETON_TEMPLATES[concept_type]
        assert len(skel.template_nodes) >= 2, f"{skel.name} has fewer than 2 nodes"
        assert len(skel.template_edges) >= 1, f"{skel.name} has no edges"

    @pytest.mark.parametrize("concept_type", list(SKELETON_TEMPLATES.keys()))
    def test_no_dangling_edges(self, concept_type):
        skel = SKELETON_TEMPLATES[concept_type]
        node_ids = {n.node_id for n in skel.template_nodes}
        for edge in skel.template_edges:
            assert (
                edge.source_id in node_ids
            ), f"Edge source {edge.source_id} not in nodes of {skel.name}"
            assert (
                edge.target_id in node_ids
            ), f"Edge target {edge.target_id} not in nodes of {skel.name}"

    @pytest.mark.parametrize("concept_type", list(SKELETON_TEMPLATES.keys()))
    def test_unique_node_ids(self, concept_type):
        skel = SKELETON_TEMPLATES[concept_type]
        ids = [n.node_id for n in skel.template_nodes]
        assert len(ids) == len(set(ids)), f"Duplicate node IDs in {skel.name}"

    # Bayesian skeletons intentionally mix concept types to enforce
    # Oracle Isolation (stateless oracle nodes) and Conjugate Update
    # semantics within a parent paradigm skeleton.
    _ALLOWED_HETEROGENEOUS: dict[ConceptType, set[ConceptType]] = {
        ConceptType.MCMC_KERNEL: {
            ConceptType.MCMC_KERNEL,
            ConceptType.PROBABILISTIC_ORACLE,
        },
        ConceptType.VI_ELBO: {ConceptType.VI_ELBO, ConceptType.PROBABILISTIC_ORACLE},
        ConceptType.SEQUENTIAL_FILTER: {
            ConceptType.SEQUENTIAL_FILTER,
            ConceptType.PROBABILISTIC_ORACLE,
            ConceptType.CONJUGATE_UPDATE,
        },
        ConceptType.MESSAGE_PASSING: {ConceptType.MESSAGE_PASSING},
        ConceptType.FIXED_POINT: {
            ConceptType.FIXED_POINT,
            ConceptType.STATE_INIT,
            ConceptType.CUSTOM,
        },
        ConceptType.MAP_OVER: {
            ConceptType.MAP_OVER,
            ConceptType.STATE_INIT,
            ConceptType.CUSTOM,
        },
        ConceptType.BASELINE_ANALYSIS: {
            ConceptType.BASELINE_ANALYSIS,
            ConceptType.MAP_OVER,
        },
    }

    @pytest.mark.parametrize("concept_type", list(SKELETON_TEMPLATES.keys()))
    def test_nodes_match_paradigm(self, concept_type):
        skel = SKELETON_TEMPLATES[concept_type]
        assert skel.paradigm == concept_type
        allowed = self._ALLOWED_HETEROGENEOUS.get(concept_type, {concept_type})
        for node in skel.template_nodes:
            assert node.concept_type in allowed, (
                f"Node {node.name} has type {node.concept_type} "
                f"but skeleton paradigm {concept_type} allows {allowed}"
            )

    @pytest.mark.parametrize("concept_type", list(SKELETON_TEMPLATES.keys()))
    def test_all_nodes_pending(self, concept_type):
        skel = SKELETON_TEMPLATES[concept_type]
        for node in skel.template_nodes:
            if concept_type == ConceptType.BASELINE_ANALYSIS:
                if node.name == "Qualify Events":
                    assert (
                        node.status == NodeStatus.PENDING
                    ), f"Baseline node {node.name} should stay PENDING, got {node.status}"
                elif node.matched_primitive:
                    assert (
                        node.status == NodeStatus.ATOMIC
                    ), f"Baseline node {node.name} should be ATOMIC, got {node.status}"
                else:
                    assert (
                        node.status == NodeStatus.PENDING
                    ), f"Baseline node {node.name} should be PENDING, got {node.status}"
            else:
                assert (
                    node.status == NodeStatus.PENDING
                ), f"Template node {node.name} should be PENDING, got {node.status}"

    @pytest.mark.parametrize("concept_type", list(SKELETON_TEMPLATES.keys()))
    def test_has_description_and_name(self, concept_type):
        skel = SKELETON_TEMPLATES[concept_type]
        assert skel.name, f"Skeleton for {concept_type} has empty name"
        assert skel.description, f"Skeleton for {concept_type} has empty description"


class TestInstantiateSkeleton:
    def test_produces_fresh_ids(self):
        skel = get_skeleton(ConceptType.DIVIDE_AND_CONQUER)
        assert skel is not None

        nodes1, edges1 = instantiate_skeleton(skel, "sort integers")
        nodes2, edges2 = instantiate_skeleton(skel, "sort strings")

        ids1 = {n.node_id for n in nodes1}
        ids2 = {n.node_id for n in nodes2}
        assert ids1.isdisjoint(ids2), "Instantiations should produce unique IDs"

    def test_preserves_structure(self):
        skel = get_skeleton(ConceptType.DIVIDE_AND_CONQUER)
        assert skel is not None

        nodes, edges = instantiate_skeleton(skel, "merge sort")
        assert len(nodes) == len(skel.template_nodes)
        assert len(edges) == len(skel.template_edges)

    def test_edges_reference_valid_nodes(self):
        skel = get_skeleton(ConceptType.DYNAMIC_PROGRAMMING)
        assert skel is not None

        nodes, edges = instantiate_skeleton(skel, "longest common subsequence")
        node_ids = {n.node_id for n in nodes}
        for edge in edges:
            assert edge.source_id in node_ids
            assert edge.target_id in node_ids

    def test_goal_in_description(self):
        skel = get_skeleton(ConceptType.GREEDY)
        assert skel is not None

        nodes, _edges = instantiate_skeleton(skel, "activity selection")
        for node in nodes:
            assert "activity selection" in node.description

    def test_parent_id_set(self):
        skel = get_skeleton(ConceptType.SORTING)
        assert skel is not None

        nodes, _edges = instantiate_skeleton(skel, "sort", parent_id="root_0")
        for node in nodes:
            assert node.parent_id == "root_0"

    def test_depth_offset(self):
        skel = get_skeleton(ConceptType.SEARCHING)
        assert skel is not None

        nodes, _edges = instantiate_skeleton(skel, "find", base_depth=3)
        for node in nodes:
            assert node.depth >= 4  # base_depth + template depth (1)
