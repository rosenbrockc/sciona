"""Tests for baseline-analysis multi-component instantiation."""

from __future__ import annotations

from sciona.architect.models import ConceptType, NodeStatus
from sciona.architect.skeletons import (
    get_skeleton,
    instantiate_baseline_multi_component,
    instantiate_skeleton,
)


def _baseline_skeleton():
    skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS)
    assert skeleton is not None
    return skeleton


class TestBaselineMultiComponent:
    def test_single_component_matches_baseline_counts(self):
        skeleton = _baseline_skeleton()
        nodes_ref, edges_ref = instantiate_skeleton(skeleton, "baseline fan-out")
        nodes, edges = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            1,
        )

        assert len(nodes) == len(nodes_ref)
        assert len(edges) == len(edges_ref)
        assert len(nodes) == 12
        assert len(edges) == 10

    def test_two_components_have_expected_counts(self):
        skeleton = _baseline_skeleton()
        nodes, edges = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            2,
        )

        assert len(nodes) == 21
        assert len(edges) == 19

        node_ids = {node.node_id for node in nodes}
        assert len(node_ids) == len(nodes)
        assert all(edge.source_id in node_ids for edge in edges)
        assert all(edge.target_id in node_ids for edge in edges)

    def test_three_components_have_expected_count(self):
        skeleton = _baseline_skeleton()
        nodes, edges = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            3,
        )

        assert len(nodes) == 30
        assert len(edges) == 28

        node_ids = {node.node_id for node in nodes}
        assert all(edge.source_id in node_ids for edge in edges)
        assert all(edge.target_id in node_ids for edge in edges)

    def test_fresh_ids_across_calls(self):
        skeleton = _baseline_skeleton()
        nodes1, _edges1 = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            2,
        )
        nodes2, _edges2 = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            2,
        )

        ids1 = {node.node_id for node in nodes1}
        ids2 = {node.node_id for node in nodes2}
        assert ids1.isdisjoint(ids2)

    def test_parent_id_and_depth(self):
        skeleton = _baseline_skeleton()
        nodes, _edges = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            3,
            parent_id="baseline_root",
            base_depth=4,
        )

        assert {node.parent_id for node in nodes} == {"baseline_root"}
        assert all(node.depth >= 5 for node in nodes)

    def test_component_naming_and_shared_nodes(self):
        skeleton = _baseline_skeleton()
        nodes, _edges = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            3,
        )

        names = {node.name for node in nodes}
        assert "Acquire Data" in names
        assert "Combine" in names
        assert "Regionize" in names

        for idx in range(1, 4):
            suffix = f" (Component {idx})"
            assert f"Windowed Analysis{suffix}" in names
            assert f"Mask{suffix}" in names
            assert f"Resample{suffix}" in names
            assert f"Scale{suffix}" in names
            assert f"Per-Window Fit{suffix}" in names
            assert f"Output Transform{suffix}" in names
            assert f"Qualify Events{suffix}" in names
            assert f"Pad{suffix}" in names
            assert f"Normalize{suffix}" in names

    def test_goal_in_all_descriptions(self):
        skeleton = _baseline_skeleton()
        goal = "baseline fan-out"
        nodes, _edges = instantiate_baseline_multi_component(skeleton, goal, 2)

        assert all(goal in node.description for node in nodes)

    def test_component_metadata_is_preserved(self):
        skeleton = _baseline_skeleton()
        nodes, _edges = instantiate_baseline_multi_component(
            skeleton,
            "baseline fan-out",
            2,
        )

        nodes_by_name = {node.name: node for node in nodes}
        windowed = nodes_by_name["Windowed Analysis (Component 1)"]
        qualify = nodes_by_name["Qualify Events (Component 1)"]
        assert windowed.map_window_size == 1024
        assert windowed.map_hop_size == 512
        assert len(windowed.children) == 5
        assert qualify.is_opaque is True
        assert qualify.matched_primitive == "baseline_fit_stack"
        assert qualify.status == NodeStatus.PENDING

        mask = nodes_by_name["Mask (Component 1)"]
        assert mask.matched_primitive == "baseline_mask"
        assert mask.status == NodeStatus.ATOMIC

        pad = nodes_by_name["Pad (Component 1)"]
        assert pad.matched_primitive == "baseline_pad_constant"
        assert pad.status == NodeStatus.ATOMIC

        normalize = nodes_by_name["Normalize (Component 1)"]
        assert normalize.matched_primitive == "baseline_normalize_max"
        assert normalize.status == NodeStatus.ATOMIC
