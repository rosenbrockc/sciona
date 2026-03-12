"""Deterministic wrapper for ingester_opaque_witness."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpaquePrompt:
    class_name: str
    base_classes: str
    method_name: str
    params: list[str]
    return_type: str
    docstring: str
    fn_name: str


_MODULE_RE = re.compile(
    r"Module:\s*(?P<class_name>.*)\n"
    r"Base classes:\s*(?P<base_classes>.*)\n"
    r"Entry method:\s*(?P<method_name>\w+)\((?P<params>.*)\)\n"
    r"Return type annotation:\s*(?P<return_type>.*)\n"
    r"Docstring:\s*(?P<docstring>.*)\n",
    re.DOTALL,
)
_FN_NAME_RE = re.compile(r'"witness_name":\s*"witness_(?P<fn_name>[^"]+)"')


def _parse_opaque_prompt(user: str) -> OpaquePrompt:
    match = _MODULE_RE.search(user)
    if match is None:
        return OpaquePrompt("", "", "", [], "", "", "")
    params = [
        param.strip()
        for param in match.group("params").split(",")
        if param.strip() and param.strip() != "self"
    ]
    fn_match = _FN_NAME_RE.search(user)
    return OpaquePrompt(
        class_name=match.group("class_name").strip(),
        base_classes=match.group("base_classes").strip(),
        method_name=match.group("method_name").strip(),
        params=params,
        return_type=match.group("return_type").strip(),
        docstring=match.group("docstring").strip(),
        fn_name=fn_match.group("fn_name").strip() if fn_match is not None else "",
    )


def _layer_signature(prompt: OpaquePrompt) -> str:
    return " ".join(
        [
            prompt.class_name,
            prompt.base_classes,
            prompt.method_name,
            prompt.return_type,
            prompt.docstring,
        ]
    ).lower()


def _identity_body(param: str) -> str:
    return f'return AbstractArray(shape={param}.shape, dtype={param}.dtype)'


def _linear_body(param: str) -> str:
    return "\n".join(
        [
            'out_features = "out_features"',
            f"return AbstractArray(shape=(*{param}.shape[:-1], out_features), dtype={param}.dtype)",
        ]
    )


def _conv1d_body(param: str) -> str:
    return "\n".join(
        [
            f'batch = {param}.shape[0] if len({param}.shape) >= 1 else "N"',
            f'length = {param}.shape[-1] if {param}.shape else "L"',
            f'return AbstractArray(shape=(batch, "c_out", length), dtype={param}.dtype)',
        ]
    )


def _conv2d_body(param: str) -> str:
    return "\n".join(
        [
            f'batch = {param}.shape[0] if len({param}.shape) >= 1 else "N"',
            f'height = {param}.shape[-2] if len({param}.shape) >= 2 else "H"',
            f'width = {param}.shape[-1] if len({param}.shape) >= 1 else "W"',
            f'return AbstractArray(shape=(batch, "c_out", height, width), dtype={param}.dtype)',
        ]
    )


def _pool_body(param: str) -> str:
    return "\n".join(
        [
            f'batch = {param}.shape[0] if len({param}.shape) >= 1 else "N"',
            f'channels = {param}.shape[1] if len({param}.shape) >= 2 else "C"',
            f'height = {param}.shape[-2] if len({param}.shape) >= 2 else "H"',
            f'width = {param}.shape[-1] if len({param}.shape) >= 1 else "W"',
            'height_out = max(1, height // 2) if isinstance(height, int) else "H_out"',
            'width_out = max(1, width // 2) if isinstance(width, int) else "W_out"',
            f"return AbstractArray(shape=(batch, channels, height_out, width_out), dtype={param}.dtype)",
        ]
    )


def _flatten_body(param: str) -> str:
    return "\n".join(
        [
            f'leading = {param}.shape[0] if len({param}.shape) >= 1 else "N"',
            'flat_dim = "flat_dim"',
            f"return AbstractArray(shape=(leading, flat_dim), dtype={param}.dtype)",
        ]
    )


def _embedding_body(param: str) -> str:
    return "\n".join(
        [
            'embed_dim = "embed_dim"',
            f"return AbstractArray(shape=(*{param}.shape, embed_dim), dtype={param}.dtype)",
        ]
    )


def _template_witness(prompt: OpaquePrompt) -> tuple[str, str] | None:
    if not prompt.params:
        return None
    param = prompt.params[0]
    signature = _layer_signature(prompt)
    if any(token in signature for token in ("linear", "dense", " fc", "fully connected")):
        return "(*, in_features) -> (*, out_features)", _linear_body(param)
    if "conv2d" in signature:
        return "(N, C_in, H, W) -> (N, C_out, H, W)", _conv2d_body(param)
    if "conv1d" in signature:
        return "(N, C_in, L) -> (N, C_out, L)", _conv1d_body(param)
    if any(token in signature for token in ("maxpool", "avgpool", "pooling")):
        return "(N, C, H, W) -> (N, C, H_out, W_out)", _pool_body(param)
    if any(token in signature for token in ("batchnorm", "layernorm", "relu", "gelu", "sigmoid", "dropout")):
        return "identity", _identity_body(param)
    if "flatten" in signature:
        return "(N, *dims) -> (N, flat_dim)", _flatten_body(param)
    if "embedding" in signature:
        return "(*,) -> (*, embed_dim)", _embedding_body(param)
    return None


class TemplateWitnessGenerator:
    """Deterministic opaque witness generator with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "opaque_witness_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        prompt = _parse_opaque_prompt(user)
        template = _template_witness(prompt)
        if template is None:
            self._last_completion_metadata = {"opaque_witness_source": "fallback"}
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        shape_transform, witness_body = template
        self._last_completion_metadata = {
            "opaque_witness_source": "deterministic",
            "opaque_witness_class": prompt.class_name,
            "opaque_witness_shape_transform": shape_transform,
        }
        self._last_error_metadata = {}
        return json.dumps(
            {
                "witness_name": f"witness_{prompt.fn_name or prompt.class_name.lower()}",
                "params": [f"{param}: AbstractArray" for param in prompt.params],
                "return_type": "AbstractArray",
                "shape_transform": shape_transform,
                "witness_body": witness_body,
            }
        )

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
