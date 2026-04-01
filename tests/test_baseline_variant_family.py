"""Tests for baseline-analysis variant family routing and mutations."""

from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import ConceptType, NodeStatus
from sciona.architect.skeletons import get_skeleton, instantiate_skeleton
from sciona.principal.variant_mutation import (
    BaselineAnalysisVariantFamily,
    VARIANT_FAMILIES,
    maybe_apply_bottleneck_variant,
)


def _baseline_cdg() -> CDGExport:
    skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS)
    assert skeleton is not None
    nodes, edges = instantiate_skeleton(skeleton, "test variant")
    return CDGExport(nodes=list(nodes), edges=list(edges), metadata={})


class TestBaselineAnalysisVariantFamily:
    def test_matches_baseline_cdg(self):
        family = BaselineAnalysisVariantFamily()
        assert family.matches(_baseline_cdg()) is True

    def test_no_match_without_atomic_nodes(self):
        family = BaselineAnalysisVariantFamily()
        assert family.matches(CDGExport(nodes=[], edges=[], metadata={})) is False

    def test_mutate_swaps_pad(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Pad")
        assert result.applied is True
        assert result.family == "baseline_analysis"
        assert result.variant_name == "baseline_pad_exponential"
        pad_nodes = [node for node in result.cdg.nodes if node.name == "Pad"]
        assert pad_nodes
        assert all(
            node.matched_primitive == "baseline_pad_exponential"
            for node in pad_nodes
        )

    def test_mutate_swaps_scale(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Scale")
        assert result.applied is True
        assert result.variant_name == "baseline_scale_wavelet"

    def test_mutate_swaps_fit(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Per-Window Fit")
        assert result.applied is True
        assert result.variant_name == "baseline_fit_exp_fall"

    def test_mutate_swaps_normalize(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Normalize")
        assert result.applied is True
        assert result.variant_name == "baseline_normalize_constant"

    def test_mutate_swaps_combine(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Combine")
        assert result.applied is True
        assert result.variant_name == "baseline_combine_convolve"

    def test_mutate_swaps_output_transform(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Output Transform")
        assert result.applied is True
        assert result.variant_name == "baseline_output_clipshift"

    def test_mutate_no_op_for_mask(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Mask")
        assert result.applied is False
        assert result.family == "baseline_analysis"
        assert result.allow_redecompose is False

    def test_mutate_no_op_for_resample(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Resample")
        assert result.applied is False
        assert result.family == "baseline_analysis"
        assert result.allow_redecompose is False

    def test_mutate_no_op_for_regionize(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Regionize")
        assert result.applied is False
        assert result.family == "baseline_analysis"
        assert result.allow_redecompose is False

    def test_mutate_wrong_bottleneck(self):
        family = BaselineAnalysisVariantFamily()
        result = family.mutate(_baseline_cdg(), bottleneck_name="Nonexistent Node")
        assert result.applied is False

    def test_mutate_rejects_non_atomic_targets(self):
        family = BaselineAnalysisVariantFamily()
        cdg = _baseline_cdg()
        for node in cdg.nodes:
            if node.name == "Acquire Data":
                node.status = NodeStatus.DECOMPOSED
                break
        result = family.mutate(cdg, bottleneck_name="Acquire Data")
        assert result.applied is False

    def test_registered_in_variant_families(self):
        names = {family.name for family in VARIANT_FAMILIES}
        assert "baseline_analysis" in names

    def test_maybe_apply_routes_to_baseline_family(self):
        result = maybe_apply_bottleneck_variant(
            _baseline_cdg(),
            bottleneck_name="Pad",
        )
        assert result.applied is True
        assert result.family == "baseline_analysis"
        assert result.variant_name == "baseline_pad_exponential"

