"""Tests for heterogeneous baseline analyzer assembly."""

from __future__ import annotations

from sciona.architect.models import (
    BaselineAnalyzerComponentSpec,
    BaselineAnalyzerSpec,
    BaselineComponentOutputRef,
    BaselineComponentShape,
    BaselinePredictorAliasSpec,
    BaselineStageSpec,
    BaselineWindowSpec,
    ConceptType,
)
from sciona.architect.skeletons import get_skeleton, instantiate_baseline_analyzer


def _baseline_skeleton():
    skeleton = get_skeleton(ConceptType.BASELINE_ANALYSIS)
    assert skeleton is not None
    return skeleton


def _ahi_like_spec() -> BaselineAnalyzerSpec:
    region_output = "list[tuple[int,int]]"
    return BaselineAnalyzerSpec(
        preprocessors=[
            BaselineStageSpec(
                key="invert_pat",
                name="Invert PAT Signal",
                description="Invert the PAT SQI trace before windowed baseline processing.",
                matched_primitive="baseline_invert_signal",
            )
        ],
        components=[
            BaselineAnalyzerComponentSpec(
                name="sma_excl",
                shape=BaselineComponentShape.WINDOWED,
                window=BaselineWindowSpec(size=120, hop=120),
                window_stages=[
                    BaselineStageSpec(
                        key="threshold",
                        name="Threshold Function",
                        description="Apply the SMA exclusion threshold function.",
                        matched_primitive="baseline_function_threshold",
                        input_name="window",
                        output_name="thresholded",
                    )
                ],
                post_stages=[
                    BaselineStageSpec(
                        key="output",
                        name="Output Transform",
                        template_name="Output Transform",
                        matched_primitive="baseline_output_clipshift",
                        input_name="thresholded",
                        output_name="onsets",
                    ),
                    BaselineStageSpec(
                        key="pad",
                        name="Pad",
                        template_name="Pad",
                        matched_primitive="baseline_pad_constant",
                        input_name="onsets",
                        output_name="padded",
                    ),
                    BaselineStageSpec(
                        key="normalize",
                        name="Normalize",
                        template_name="Normalize",
                        matched_primitive="baseline_normalize_max",
                        input_name="padded",
                        output_name="normalized",
                    ),
                    BaselineStageSpec(
                        key="regions",
                        name="Regionize",
                        template_name="Regionize",
                        matched_primitive="baseline_regionize",
                        input_name="normalized",
                        output_name="regions",
                        output_type=region_output,
                    ),
                ],
                default_output_stage="regions",
            ),
            BaselineAnalyzerComponentSpec(
                name="irsqi",
                shape=BaselineComponentShape.WINDOWED,
                window=BaselineWindowSpec(size=120, hop=120),
                window_stages=[
                    BaselineStageSpec(
                        key="rise",
                        name="Exp Rise Step",
                        description="Detect SQI rises directly from the per-window signal.",
                        matched_primitive="baseline_fit_exp_rise",
                        input_name="window",
                        output_name="rise_signal",
                    )
                ],
                post_stages=[
                    BaselineStageSpec(
                        key="output",
                        name="Output Transform",
                        template_name="Output Transform",
                        matched_primitive="baseline_output_nonzero",
                        input_name="rise_signal",
                        output_name="onsets",
                    ),
                    BaselineStageSpec(
                        key="pad",
                        name="Pad",
                        template_name="Pad",
                        matched_primitive="baseline_pad_exponential",
                        input_name="onsets",
                        output_name="probability",
                    ),
                    BaselineStageSpec(
                        key="regions",
                        name="Regionize",
                        template_name="Regionize",
                        matched_primitive="baseline_regionize",
                        input_name="probability",
                        output_name="regions",
                        output_type=region_output,
                    ),
                ],
                default_output_stage="regions",
            ),
            BaselineAnalyzerComponentSpec(
                name="spo2",
                shape=BaselineComponentShape.WINDOWED,
                window=BaselineWindowSpec(size=120, hop=120),
                window_stages=[
                    BaselineStageSpec(
                        key="mask",
                        name="Mask",
                        template_name="Mask",
                        matched_primitive="baseline_mask",
                        input_name="window",
                        output_name="masked",
                    ),
                    BaselineStageSpec(
                        key="fit",
                        name="Per-Window Fit",
                        template_name="Per-Window Fit",
                        matched_primitive="baseline_fit_stack_spo2",
                        input_name="masked",
                        output_name="fit_internals",
                        output_type="BaselineFitStackInternals",
                    ),
                ],
                post_stages=[
                    BaselineStageSpec(
                        key="qualify",
                        name="Qualify Events",
                        template_name="Qualify Events",
                        matched_primitive="baseline_fit_stack",
                        status="pending",
                        is_opaque=True,
                        input_name="fit_internals",
                        input_type="BaselineFitStackInternals",
                        output_name="probability",
                    ),
                    BaselineStageSpec(
                        key="regions",
                        name="Regionize",
                        template_name="Regionize",
                        matched_primitive="baseline_regionize",
                        input_name="probability",
                        output_name="regions",
                        output_type=region_output,
                    ),
                ],
                default_output_stage="regions",
            ),
            BaselineAnalyzerComponentSpec(
                name="patsqi",
                shape=BaselineComponentShape.WINDOWED,
                source_key="invert_pat",
                window=BaselineWindowSpec(size=120, hop=120),
                window_stages=[
                    BaselineStageSpec(
                        key="rise",
                        name="Exp Rise Step",
                        description="Detect PAT SQI rises from the shared inverted PAT signal.",
                        matched_primitive="baseline_fit_exp_rise",
                        input_name="window",
                        output_name="rise_signal",
                    )
                ],
                post_stages=[
                    BaselineStageSpec(
                        key="output",
                        name="Output Transform",
                        template_name="Output Transform",
                        matched_primitive="baseline_output_nonzero",
                        input_name="rise_signal",
                        output_name="onsets",
                    ),
                    BaselineStageSpec(
                        key="pad",
                        name="Pad",
                        template_name="Pad",
                        matched_primitive="baseline_pad_exponential",
                        input_name="onsets",
                        output_name="probability",
                    ),
                    BaselineStageSpec(
                        key="regions",
                        name="Regionize",
                        template_name="Regionize",
                        matched_primitive="baseline_regionize",
                        input_name="probability",
                        output_name="regions",
                        output_type=region_output,
                    ),
                ],
                default_output_stage="regions",
            ),
            BaselineAnalyzerComponentSpec(
                name="combined",
                shape=BaselineComponentShape.COMBINER,
                combine_stage=BaselineStageSpec(
                    key="combine",
                    name="Combine",
                    template_name="Combine",
                    matched_primitive="baseline_combine_product",
                    input_name="probability",
                    output_name="combined_probability",
                ),
                combine_inputs=[
                    BaselineComponentOutputRef(component="sma_excl", stage_key="normalize"),
                    BaselineComponentOutputRef(component="irsqi", stage_key="pad"),
                ],
                post_stages=[
                    BaselineStageSpec(
                        key="regions",
                        name="Regionize",
                        template_name="Regionize",
                        matched_primitive="baseline_regionize",
                        input_name="combined_probability",
                        output_name="regions",
                        output_type=region_output,
                    )
                ],
                default_output_stage="regions",
            ),
        ],
        predictor_aliases=[
            BaselinePredictorAliasSpec(
                alias="sqi",
                source=BaselineComponentOutputRef(
                    component="irsqi",
                    stage_key="regions",
                ),
            ),
            BaselinePredictorAliasSpec(
                alias="combined",
                source=BaselineComponentOutputRef(
                    component="combined",
                    stage_key="regions",
                ),
            ),
            BaselinePredictorAliasSpec(
                alias="pat",
                source=BaselineComponentOutputRef(
                    component="patsqi",
                    stage_key="regions",
                ),
            ),
        ],
    )


def _edge_pairs(nodes, edges):
    id_to_name = {node.node_id: node.name for node in nodes}
    return {(id_to_name[edge.source_id], id_to_name[edge.target_id]) for edge in edges}


class TestBaselineAnalyzerAssembly:
    def test_shared_preprocessor_is_singleton_and_fans_out(self):
        skeleton = _baseline_skeleton()
        nodes, edges = instantiate_baseline_analyzer(
            skeleton,
            "AHI baseline core",
            _ahi_like_spec(),
        )

        names = [node.name for node in nodes]
        assert names.count("Invert PAT Signal") == 1

        pairs = _edge_pairs(nodes, edges)
        assert ("Acquire Data", "Invert PAT Signal") in pairs
        assert ("Invert PAT Signal", "Windowed Analysis (patsqi)") in pairs

    def test_predictor_alias_nodes_attach_to_expected_components(self):
        skeleton = _baseline_skeleton()
        nodes, edges = instantiate_baseline_analyzer(
            skeleton,
            "AHI baseline core",
            _ahi_like_spec(),
        )

        pairs = _edge_pairs(nodes, edges)
        assert ("Regionize (irsqi)", "Predictor Alias: sqi") in pairs
        assert ("Regionize (combined)", "Predictor Alias: combined") in pairs
        assert ("Regionize (patsqi)", "Predictor Alias: pat") in pairs

    def test_combiner_only_component_has_no_windowed_node(self):
        skeleton = _baseline_skeleton()
        nodes, edges = instantiate_baseline_analyzer(
            skeleton,
            "AHI baseline core",
            _ahi_like_spec(),
        )

        names = {node.name for node in nodes}
        assert "Windowed Analysis (combined)" not in names
        assert "Combine (combined)" in names
        assert "Regionize (combined)" in names

        pairs = _edge_pairs(nodes, edges)
        assert ("Normalize (sma_excl)", "Combine (combined)") in pairs
        assert ("Pad (irsqi)", "Combine (combined)") in pairs
        assert ("Combine (combined)", "Regionize (combined)") in pairs

    def test_heterogeneous_component_shapes_are_preserved(self):
        skeleton = _baseline_skeleton()
        nodes, _edges = instantiate_baseline_analyzer(
            skeleton,
            "AHI baseline core",
            _ahi_like_spec(),
        )

        names = {node.name for node in nodes}
        assert "Threshold Function (sma_excl)" in names
        assert "Normalize (sma_excl)" in names
        assert "Exp Rise Step (irsqi)" in names
        assert "Qualify Events (spo2)" in names
        assert "Exp Rise Step (patsqi)" in names
        assert "Per-Window Fit (spo2)" in names

        by_name = {node.name: node for node in nodes}
        patsqi_window = by_name["Windowed Analysis (patsqi)"]
        spo2_window = by_name["Windowed Analysis (spo2)"]
        assert patsqi_window.map_window_size == 120
        assert patsqi_window.map_hop_size == 120
        assert len(patsqi_window.children) == 1
        assert len(spo2_window.children) == 2
        assert by_name["Qualify Events (spo2)"].is_opaque is True
