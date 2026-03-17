"""Tests for the Conceptual Abstraction Agent step in the semantic chunker.

Covers: ConceptualProfile model, prompt templates, abstract_atoms node,
description enrichment, graph wiring, and LLM failure fallback.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.runnables import RunnableConfig

from ageom.architect.models import ConceptType, IOSpec
from ageom.ingester.chunker import (
    ChunkerDeps,
    ChunkerState,
    abstract_atoms,
    build_chunker_graph,
    _format_io_specs,
    _parse_conceptual_profile,
)
from ageom.ingester.models import (
    ConceptualProfile,
    MacroAtomSpec,
    ProposedMacroPlan,
    ValidatedMacroPlan,
    RawDataFlowGraph,
)
from ageom.ingester.prompts import (
    CONCEPTUAL_ABSTRACT_SYSTEM,
    CONCEPTUAL_ABSTRACT_USER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atom(
    name: str = "Signal Conditioner",
    description: str = "Applies Butterworth low-pass filter to ECG signal",
    concept_type: ConceptType = ConceptType.SIGNAL_FILTER,
) -> MacroAtomSpec:
    return MacroAtomSpec(
        name=name,
        description=description,
        concept_type=concept_type,
        method_names=["condition"],
        inputs=[IOSpec(name="signal", type_desc="ndarray", constraints="1D float64")],
        outputs=[IOSpec(name="filtered", type_desc="ndarray")],
    )


def _make_validated_plan(
    atoms: list[MacroAtomSpec] | None = None,
) -> ValidatedMacroPlan:
    if atoms is None:
        atoms = [_make_atom()]
    return ValidatedMacroPlan(
        plan=ProposedMacroPlan(macro_atoms=atoms),
        all_attrs_accounted=True,
    )


_SAMPLE_LLM_RESPONSE = json.dumps(
    {
        "abstract_name": "Frequency-Band Attenuator",
        "conceptual_transform": (
            "Suppresses spectral components above a cutoff frequency in a "
            "uniformly sampled 1D real-valued sequence, preserving low-frequency "
            "structure while attenuating high-frequency oscillations."
        ),
        "abstract_inputs": [
            "A uniformly sampled 1D array of floats representing a continuous physical measurement"
        ],
        "abstract_outputs": [
            "A 1D array of the same length with high-frequency components attenuated"
        ],
        "algorithmic_properties": [
            "linear",
            "causal",
            "stateless",
            "lossy-compression",
        ],
        "cross_disciplinary_applications": [
            "Smoothing telemetry data in aerospace",
            "De-noising seismic waveforms in geophysics",
            "Filtering high-frequency noise in audio processing",
            "Removing motion artifacts from accelerometer data in wearables",
        ],
    }
)


# ---------------------------------------------------------------------------
# ConceptualProfile model
# ---------------------------------------------------------------------------


class TestConceptualProfile:
    def test_defaults(self):
        p = ConceptualProfile()
        assert p.abstract_name == ""
        assert p.algorithmic_properties == []
        assert p.cross_disciplinary_applications == []

    def test_full_construction(self):
        p = ConceptualProfile(
            abstract_name="Frequency-Band Attenuator",
            conceptual_transform="Suppresses spectral components.",
            abstract_inputs=["1D float array"],
            abstract_outputs=["1D float array, attenuated"],
            algorithmic_properties=["linear", "causal"],
            cross_disciplinary_applications=["aerospace", "geophysics"],
        )
        assert p.abstract_name == "Frequency-Band Attenuator"
        assert len(p.cross_disciplinary_applications) == 2

    def test_on_macro_atom_spec(self):
        atom = _make_atom()
        assert atom.conceptual_profile is None

        profile = ConceptualProfile(abstract_name="Test")
        enriched = atom.model_copy(update={"conceptual_profile": profile})
        assert enriched.conceptual_profile is not None
        assert enriched.conceptual_profile.abstract_name == "Test"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_system_prompt_contains_key_instructions(self):
        assert "domain-agnostic" in CONCEPTUAL_ABSTRACT_SYSTEM.lower()
        assert "Eradicate Domain Jargon" in CONCEPTUAL_ABSTRACT_SYSTEM
        assert "Conceptual Transform" in CONCEPTUAL_ABSTRACT_SYSTEM
        assert "Structural Properties" in CONCEPTUAL_ABSTRACT_SYSTEM
        assert "Isomorphic Use Cases" in CONCEPTUAL_ABSTRACT_SYSTEM

    def test_user_prompt_has_placeholders(self):
        assert "{atom_name}" in CONCEPTUAL_ABSTRACT_USER
        assert "{atom_description}" in CONCEPTUAL_ABSTRACT_USER
        assert "{concept_type}" in CONCEPTUAL_ABSTRACT_USER
        assert "{inputs_spec}" in CONCEPTUAL_ABSTRACT_USER
        assert "{outputs_spec}" in CONCEPTUAL_ABSTRACT_USER
        assert "{method_names}" in CONCEPTUAL_ABSTRACT_USER

    def test_user_prompt_formats(self):
        result = CONCEPTUAL_ABSTRACT_USER.format(
            atom_name="Signal Conditioner",
            atom_description="Filters noise",
            concept_type="signal_filter",
            inputs_spec="  - signal: ndarray (1D float64)",
            outputs_spec="  - filtered: ndarray",
            method_names="condition",
        )
        assert "Signal Conditioner" in result
        assert "signal_filter" in result

    def test_system_prompt_requests_json(self):
        assert "Return valid JSON only" in CONCEPTUAL_ABSTRACT_SYSTEM


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_format_io_specs_empty(self):
        assert _format_io_specs([]) == "(none)"

    def test_format_io_specs_with_constraints(self):
        specs = [IOSpec(name="x", type_desc="ndarray", constraints="sorted")]
        result = _format_io_specs(specs)
        assert "x: ndarray (sorted)" in result

    def test_format_io_specs_without_constraints(self):
        specs = [IOSpec(name="x", type_desc="int")]
        result = _format_io_specs(specs)
        assert "x: int" in result
        assert "()" not in result  # no empty parens

    def test_parse_conceptual_profile(self):
        raw = {
            "abstract_name": "Temporal Smoother",
            "conceptual_transform": "Smooths a sequence.",
            "abstract_inputs": ["1D float array"],
            "abstract_outputs": ["1D float array"],
            "algorithmic_properties": ["linear"],
            "cross_disciplinary_applications": ["physics", "finance"],
        }
        p = _parse_conceptual_profile(raw)
        assert p.abstract_name == "Temporal Smoother"
        assert p.algorithmic_properties == ["linear"]

    def test_parse_conceptual_profile_missing_keys(self):
        p = _parse_conceptual_profile({})
        assert p.abstract_name == ""
        assert p.cross_disciplinary_applications == []


# ---------------------------------------------------------------------------
# abstract_atoms node
# ---------------------------------------------------------------------------


class TestAbstractAtomsNode:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.complete.return_value = _SAMPLE_LLM_RESPONSE
        return llm

    @pytest.fixture
    def config(self, mock_llm):
        deps = ChunkerDeps(llm=mock_llm)
        return RunnableConfig(configurable={"deps": deps})

    @pytest.mark.asyncio
    async def test_stores_profile_without_modifying_description(self, mock_llm, config):
        validated = _make_validated_plan()
        state: ChunkerState = {
            "raw_dfg": RawDataFlowGraph(class_name="Test"),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }

        result = await abstract_atoms(state, config)
        new_plan = result["validated_plan"]
        atom = new_plan.plan.macro_atoms[0]

        # Description should NOT contain the profile JSON
        assert "<!-- conceptual_profile -->" not in atom.description
        assert atom.description == "Applies Butterworth low-pass filter to ECG signal"

        # Profile should be stored on the atom
        assert atom.conceptual_profile is not None
        assert atom.conceptual_profile.abstract_name == "Frequency-Band Attenuator"
        assert len(atom.conceptual_profile.cross_disciplinary_applications) == 4

    @pytest.mark.asyncio
    async def test_calls_llm_per_atom(self, mock_llm, config):
        atoms = [
            _make_atom("Atom A"),
            _make_atom("Atom B"),
            _make_atom("Atom C"),
        ]
        validated = _make_validated_plan(atoms)
        state: ChunkerState = {
            "raw_dfg": RawDataFlowGraph(class_name="Test"),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }

        await abstract_atoms(state, config)
        assert mock_llm.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, config):
        """On LLM failure, atom gets a fallback profile with just the name."""
        llm = config["configurable"]["deps"].llm
        llm.complete.side_effect = RuntimeError("LLM down")

        validated = _make_validated_plan()
        state: ChunkerState = {
            "raw_dfg": RawDataFlowGraph(class_name="Test"),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }

        result = await abstract_atoms(state, config)
        atom = result["validated_plan"].plan.macro_atoms[0]
        assert atom.conceptual_profile is not None
        assert atom.conceptual_profile.abstract_name == "Signal Conditioner"

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self, config):
        """On invalid JSON from LLM, atom gets a fallback profile."""
        llm = config["configurable"]["deps"].llm
        llm.complete.return_value = "not valid json {"

        validated = _make_validated_plan()
        state: ChunkerState = {
            "raw_dfg": RawDataFlowGraph(class_name="Test"),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }

        result = await abstract_atoms(state, config)
        atom = result["validated_plan"].plan.macro_atoms[0]
        assert atom.conceptual_profile.abstract_name == "Signal Conditioner"

    @pytest.mark.asyncio
    async def test_preserves_other_atom_fields(self, mock_llm, config):
        """Enrichment should not clobber non-description fields."""
        atom = _make_atom()
        validated = _make_validated_plan([atom])
        state: ChunkerState = {
            "raw_dfg": RawDataFlowGraph(class_name="Test"),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }

        result = await abstract_atoms(state, config)
        enriched = result["validated_plan"].plan.macro_atoms[0]
        assert enriched.name == "Signal Conditioner"
        assert enriched.concept_type == ConceptType.SIGNAL_FILTER
        assert enriched.method_names == ["condition"]
        assert len(enriched.inputs) == 1
        assert len(enriched.outputs) == 1


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


class TestGraphWiring:
    def test_abstract_atoms_node_in_graph(self):
        graph = build_chunker_graph()
        compiled = graph.compile()
        # The node should exist in the graph
        assert "abstract_atoms" in compiled.get_graph().nodes

    def test_critic_routes_to_abstract_atoms(self):
        graph = build_chunker_graph()
        compiled = graph.compile()
        graph_dict = compiled.get_graph()
        # critic_validate routes through decompose_complex_atoms to abstract_atoms
        critic_edges = [e for e in graph_dict.edges if e.source == "critic_validate"]
        critic_targets = {e.target for e in critic_edges}
        assert "decompose_complex_atoms" in critic_targets
        decompose_edges = [e for e in graph_dict.edges if e.source == "decompose_complex_atoms"]
        decompose_targets = {e.target for e in decompose_edges}
        assert "abstract_atoms" in decompose_targets


# ---------------------------------------------------------------------------
# Prompt concept_type list coverage
# ---------------------------------------------------------------------------


def test_all_concept_types_in_prompt():
    """Every ConceptType value must appear in the ingester prompt."""
    from ageom.ingester.prompts import SEMANTIC_CHUNK_SYSTEM, DECOMPOSE_ATOM_SYSTEM

    for ct in ConceptType:
        assert ct.value in SEMANTIC_CHUNK_SYSTEM, (
            f"ConceptType.{ct.name} ({ct.value}) missing from SEMANTIC_CHUNK_SYSTEM"
        )
        assert ct.value in DECOMPOSE_ATOM_SYSTEM, (
            f"ConceptType.{ct.name} ({ct.value}) missing from DECOMPOSE_ATOM_SYSTEM"
        )
