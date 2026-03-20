from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.runnables import RunnableConfig

from sciona.architect.models import ConceptType, IOSpec
from sciona.commands._helpers import _create_llm_router
from sciona.ingester.chunker import ChunkerDeps, ChunkerState, abstract_atoms
from sciona.ingester.models import ConceptualProfile, MacroAtomSpec, ProposedMacroPlan, RawDataFlowGraph, ValidatedMacroPlan
from sciona.ingester.prompts import CONCEPTUAL_ABSTRACT_SYSTEM, CONCEPTUAL_ABSTRACT_USER
from sciona.ingester.template_abstractor import TemplateAbstractor, _parse_abstract_prompt
from sciona.llm_router import INGESTER_ABSTRACT, LLMRouter


def _user() -> str:
    return CONCEPTUAL_ABSTRACT_USER.format(
        atom_name="ECG Bandpass Filter",
        atom_description="Applies Butterworth bandpass filtering to an ECG waveform",
        concept_type="signal_filter",
        inputs_spec="- signal: ndarray\n- fs: float",
        outputs_spec="- filtered: ndarray",
        method_names="filter_signal, stabilize_response",
    )


def _validated_plan() -> ValidatedMacroPlan:
    atom = MacroAtomSpec(
        name="ECG Bandpass Filter",
        description="Applies Butterworth bandpass filtering to an ECG waveform",
        method_names=["filter_signal", "stabilize_response"],
        inputs=[IOSpec(name="signal", type_desc="ndarray"), IOSpec(name="fs", type_desc="float")],
        outputs=[IOSpec(name="filtered", type_desc="ndarray")],
        concept_type=ConceptType.SIGNAL_FILTER,
    )
    return ValidatedMacroPlan(plan=ProposedMacroPlan(macro_atoms=[atom]))


def test_parse_abstract_prompt_extracts_fields():
    parsed = _parse_abstract_prompt(_user())

    assert parsed["atom_name"] == "ECG Bandpass Filter"
    assert parsed["concept_type"] == "signal_filter"
    assert parsed["inputs"] == ["signal: ndarray", "fs: float"]
    assert parsed["outputs"] == ["filtered: ndarray"]


@pytest.mark.asyncio
async def test_template_abstractor_generates_signal_filter_profile():
    fallback = AsyncMock()
    abstractor = TemplateAbstractor(fallback)
    response = await abstractor.complete(CONCEPTUAL_ABSTRACT_SYSTEM, _user())

    payload = json.loads(response)
    assert payload["abstract_name"] == "Signal Conditioner"
    assert "filtering step" in payload["conceptual_transform"]
    assert "signal_processing" in payload["algorithmic_properties"]
    assert len(payload["cross_disciplinary_applications"]) == 3
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_template_abstractor_falls_back_for_weak_metadata():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    abstractor = TemplateAbstractor(fallback)
    user = CONCEPTUAL_ABSTRACT_USER.format(
        atom_name="x",
        atom_description="",
        concept_type="custom",
        inputs_spec="",
        outputs_spec="",
        method_names="",
    )

    response = await abstractor.complete(CONCEPTUAL_ABSTRACT_SYSTEM, user)

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_abstract_atoms_uses_deterministic_abstractor():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    llm = LLMRouter(default=fallback, overrides={INGESTER_ABSTRACT: TemplateAbstractor(fallback)})
    config = RunnableConfig(configurable={"deps": ChunkerDeps(llm=llm)})
    state: ChunkerState = {
        "raw_dfg": RawDataFlowGraph(class_name="ECGProcessor"),
        "proposed_plan": ProposedMacroPlan(),
        "validated_plan": _validated_plan(),
        "critique_passed": True,
        "critique_reason": "",
        "retry_count": 0,
        "missing_attrs": [],
        "done": False,
    }

    result = await abstract_atoms(state, config)

    profile = result["validated_plan"].plan.macro_atoms[0].conceptual_profile
    assert isinstance(profile, ConceptualProfile)
    assert profile.abstract_name == "Signal Conditioner"


def test_create_llm_router_wraps_ingester_abstract_deterministically(monkeypatch):
    created: list[tuple[str, str]] = []

    def _fake_create_llm_client(*, provider, model, **kwargs):
        created.append((provider, model))
        client = AsyncMock()
        client.provider = provider
        client.model = model
        return client

    monkeypatch.setattr("sciona.hunter.llm.create_llm_client", _fake_create_llm_client)

    args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None)
    config = SimpleNamespace(
        execution_mode="verified",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250929",
        llm_max_tokens=4096,
        anthropic_api_key="",
        openai_api_key="",
        openai_base_url="",
        llama_cpp_base_url="http://127.0.0.1:8080/v1",
        llama_cpp_api_key="local",
        use_agent_layer=False,
        ingester_llm_provider="",
        ingester_llm_model="",
        ingester_abstract_llm_provider="llama_cpp",
        ingester_abstract_llm_model="qwen3:14b",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "ingester", [INGESTER_ABSTRACT])

    assert isinstance(router, LLMRouter)
    assert created == [("anthropic", "claude-sonnet-4-5-20250929")]
    assert isinstance(router.for_prompt(INGESTER_ABSTRACT), TemplateAbstractor)
