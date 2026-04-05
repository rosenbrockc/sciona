"""Explicit structured-return allowlist for known safe ingest targets."""

from __future__ import annotations

from dataclasses import dataclass

from sciona.ingester.models import IOSpec, MethodFact, OutputBindingSpec


@dataclass(frozen=True)
class StructuredReturnField:
    """One explicitly allowlisted structured return binding."""

    output_name: str
    field_key: str
    type_desc: str = "Any"


@dataclass(frozen=True)
class StructuredReturnShape:
    """Matcher-local structured return knowledge for one subject/method pair."""

    subject_name: str
    source_method: str
    fields: tuple[StructuredReturnField, ...]


_STRUCTURED_RETURN_ALLOWLIST: tuple[StructuredReturnShape, ...] = (
    StructuredReturnShape(
        subject_name="PeakDetector",
        source_method="detect",
        fields=(
            StructuredReturnField(
                output_name="rpeaks",
                field_key="rpeaks",
                type_desc="list[int]",
            ),
            StructuredReturnField(
                output_name="quality",
                field_key="quality",
                type_desc="float",
            ),
        ),
    ),
    StructuredReturnShape(
        subject_name="OnsetDetector",
        source_method="detect_events",
        fields=(
            StructuredReturnField(
                output_name="onsets",
                field_key="onsets",
                type_desc="list[int]",
            ),
            StructuredReturnField(
                output_name="confidence",
                field_key="confidence",
                type_desc="float",
            ),
        ),
    ),
    StructuredReturnShape(
        subject_name="SQIDetector",
        source_method="evaluate",
        fields=(
            StructuredReturnField(
                output_name="accepted",
                field_key="accepted",
                type_desc="bool",
            ),
            StructuredReturnField(
                output_name="quality",
                field_key="quality",
                type_desc="float",
            ),
        ),
    ),
)


def resolve_structured_return_bindings(
    *,
    subject_name: str,
    methods: list[MethodFact],
    legacy_outputs: list[IOSpec],
) -> list[OutputBindingSpec] | None:
    """Return explicit bindings for one allowlisted structured-return case.

    The matcher only activates this layer for exact subject/method matches, and
    only when the requested outputs exactly match the allowlisted output set.
    Any mismatch falls back to the existing conservative behavior.
    """

    if not legacy_outputs:
        return None

    methods_by_name = {method.name: method for method in methods}
    for shape in _STRUCTURED_RETURN_ALLOWLIST:
        if shape.subject_name != subject_name:
            continue
        method = methods_by_name.get(shape.source_method)
        if method is None:
            continue

        allowlisted_names = {field.output_name for field in shape.fields}
        legacy_names = {output.name for output in legacy_outputs}
        if legacy_names != allowlisted_names:
            return None

        field_by_name = {field.output_name: field for field in shape.fields}
        bindings: list[OutputBindingSpec] = []
        for output in legacy_outputs:
            field = field_by_name.get(output.name)
            if field is None:
                return None
            bindings.append(
                OutputBindingSpec(
                    output_name=output.name,
                    type_desc=output.type_desc or field.type_desc,
                    binding_kind="dict_field",
                    source_method=shape.source_method,
                    source_attr=field.field_key,
                )
            )
        return bindings

    return None
