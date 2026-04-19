"""Template-based witness generation and conceptual abstraction.

Deterministic pattern matching for common DL layer shapes and
domain-specific conceptual profiles. No LLM calls.
"""

from __future__ import annotations

from typing import Any


def match_witness_template(
    class_name: str,
    base_classes: str,
    method_name: str,
    params: list[str],
    return_type: str,
    docstring: str = "",
) -> tuple[str, str] | None:
    """Match a DL module signature against known witness templates.

    Recognizes common layer types: Linear, Conv1D, Conv2D, Pooling,
    Activation/Norm, Flatten, Embedding. Returns a (shape_transform,
    witness_body) tuple for the matching template.

    Args:
        class_name: Name of the DL module class.
        base_classes: Comma-separated base class names.
        method_name: Entry method name (usually "forward").
        params: Parameter names (excluding self).
        return_type: Return type annotation string.
        docstring: Optional class/method docstring.

    Returns:
        (shape_transform, witness_body) if a template matches, else None.
    """
    from sciona.ingester.template_witness_generator import (
        OpaquePrompt,
        _template_witness,
    )

    prompt = OpaquePrompt(
        class_name=class_name,
        base_classes=base_classes,
        method_name=method_name,
        params=params,
        return_type=return_type,
        docstring=docstring,
        fn_name=class_name.lower(),
    )
    return _template_witness(prompt)


def generate_abstract_profile(
    atom_name: str,
    concept_type: str,
    inputs: list[str],
    outputs: list[str],
    methods: list[str],
) -> dict[str, Any] | None:
    """Generate a cross-domain conceptual profile for an atom.

    Uses domain prefix stripping, token matching, and template lookup
    to produce an abstract profile with: abstract_name,
    conceptual_transform, algorithmic_properties, and
    cross_disciplinary_applications.

    Args:
        atom_name: The atom's identifier.
        concept_type: The concept type (e.g., "signal_filter").
        inputs: List of input parameter names.
        outputs: List of output names.
        methods: List of method names in the atom.

    Returns:
        Profile dict or None if no template matches.
    """
    from sciona.ingester.template_abstractor import _generate_abstract

    return _generate_abstract(atom_name, concept_type, inputs, outputs, methods)


__all__ = ["match_witness_template", "generate_abstract_profile"]
