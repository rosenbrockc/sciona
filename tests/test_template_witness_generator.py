from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ageom.architect.models import ConceptType, IOSpec
from ageom.commands._helpers import _create_llm_router
from ageom.ingester.emitter import generate_opaque_witnesses
from ageom.ingester.models import MacroAtomSpec, MethodFact, RawDataFlowGraph
from ageom.ingester.prompts import DRAFT_OPAQUE_WITNESS_SYSTEM, DRAFT_OPAQUE_WITNESS_USER
from ageom.ingester.template_witness_generator import (
    TemplateWitnessGenerator,
    _parse_opaque_prompt,
)
from ageom.llm_router import INGESTER_OPAQUE_WITNESS, LLMRouter


def _prompt(
    *,
    class_name: str,
    method_name: str = "forward",
    params: str = "x",
    return_type: str = "Tensor",
    docstring: str = "(none)",
    fn_name: str = "layer",
) -> str:
    return DRAFT_OPAQUE_WITNESS_USER.format(
        class_name=class_name,
        base_classes="nn.Module",
        method_name=method_name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        fn_name=fn_name,
        param_specs='"x: AbstractArray"',
        return_type_spec="AbstractArray",
    )


def test_parse_opaque_prompt_extracts_fields():
    parsed = _parse_opaque_prompt(
        _prompt(class_name="Linear", params="self, x", fn_name="linear_layer")
    )

    assert parsed.class_name == "Linear"
    assert parsed.method_name == "forward"
    assert parsed.params == ["x"]
    assert parsed.fn_name == "linear_layer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("class_name", "needle"),
    [
        ("Linear", "out_features"),
        ("Conv2d", 'shape=(batch, "c_out", height, width)'),
        ("BatchNorm1d", "x.shape"),
        ("Flatten", "flat_dim"),
        ("Embedding", "embed_dim"),
    ],
)
async def test_template_witness_generator_emits_known_layer_templates(
    class_name: str,
    needle: str,
):
    fallback = AsyncMock()
    generator = TemplateWitnessGenerator(fallback)

    response = await generator.complete(
        DRAFT_OPAQUE_WITNESS_SYSTEM,
        _prompt(class_name=class_name, fn_name=class_name.lower()),
    )

    payload = json.loads(response)
    assert needle in payload["witness_body"]
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_template_witness_generator_falls_back_for_unknown_layer():
    fallback = AsyncMock()
    fallback.complete.return_value = "fallback"
    generator = TemplateWitnessGenerator(fallback)

    response = await generator.complete(
        DRAFT_OPAQUE_WITNESS_SYSTEM,
        _prompt(class_name="FeatureExtractor", docstring="Custom feature extractor block."),
    )

    assert response == "fallback"
    fallback.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_opaque_witnesses_uses_template_generator():
    fallback = AsyncMock()
    fallback.complete.side_effect = AssertionError("fallback should not be used")
    llm = LLMRouter(
        default=fallback,
        overrides={INGESTER_OPAQUE_WITNESS: TemplateWitnessGenerator(fallback)},
    )
    atom = MacroAtomSpec(
        name="Linear",
        method_names=["forward"],
        inputs=[IOSpec(name="x", type_desc="Any")],
        outputs=[IOSpec(name="output", type_desc="Any")],
        concept_type=ConceptType.NEURAL_NETWORK,
        is_opaque=True,
    )
    dfg = RawDataFlowGraph(
        class_name="Linear",
        methods=[
            MethodFact(
                name="forward",
                params=["x"],
                return_type="Tensor",
                docstring="Linear layer forward pass.",
                is_opaque=True,
            )
        ],
        is_opaque=True,
        opaque_base_classes=["nn.Module"],
    )

    witness_source, name_map = await generate_opaque_witnesses([atom], dfg, llm)

    assert "witness_linear" in name_map.values()
    assert "out_features" in witness_source


def test_create_llm_router_wraps_ingester_opaque_witness_deterministically(monkeypatch):
    created: list[tuple[str, str]] = []

    def _fake_create_llm_client(*, provider, model, **kwargs):
        created.append((provider, model))
        client = AsyncMock()
        client.provider = provider
        client.model = model
        return client

    monkeypatch.setattr("ageom.hunter.llm.create_llm_client", _fake_create_llm_client)

    args = SimpleNamespace(llm_provider=None, llm_model=None, llm_max_tokens=None, mode="verified")
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
        ingester_opaque_witness_llm_provider="codex_shim",
        ingester_opaque_witness_llm_model="gpt-5.4-codex",
        allow_legacy_subprocess_providers=False,
    )

    router = _create_llm_router(args, config, "ingester", [INGESTER_OPAQUE_WITNESS])

    assert isinstance(router, LLMRouter)
    assert created == [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("codex_shim", "gpt-5.4-codex"),
    ]
    assert isinstance(router.for_prompt(INGESTER_OPAQUE_WITNESS), TemplateWitnessGenerator)
