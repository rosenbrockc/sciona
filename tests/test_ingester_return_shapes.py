"""Tests for the explicit structured-return allowlist."""

from __future__ import annotations

from sciona.architect.models import ConceptType, IOSpec
from sciona.ingester.chunker import _infer_output_bindings
from sciona.ingester.emitter import generate_atom_wrappers, generate_ghost_witnesses
from sciona.ingester.models import (
    IngestIRPlan,
    MacroAtomSpec,
    MethodBinding,
    MethodFact,
    OperationSpec,
    OutputBindingSpec,
    ParameterFact,
    ProposedMacroPlan,
    ReturnFact,
    SourceSpan,
    FactProvenance,
    ValidatedMacroPlan,
)
from sciona.ingester.return_shapes import resolve_structured_return_bindings


def _provenance() -> FactProvenance:
    return FactProvenance(
        rule_id="test",
        span=SourceSpan(file_path="detector.py", line_start=1, line_end=1),
    )


def _detect_method() -> MethodFact:
    prov = _provenance()
    return MethodFact(
        name="detect",
        signature=[
            ParameterFact(name="self", provenance=prov),
            ParameterFact(name="signal", provenance=prov),
        ],
        return_type="dict[str, object]",
        return_facts=[ReturnFact(kind="unknown", provenance=prov)],
        provenance=[prov],
    )


def _structured_outputs(*names: str) -> list[IOSpec]:
    by_name = {
        "rpeaks": IOSpec(name="rpeaks", type_desc="list[int]"),
        "quality": IOSpec(name="quality", type_desc="float"),
        "count": IOSpec(name="count", type_desc="int"),
    }
    return [by_name[name] for name in names]


def test_return_shape_helper_resolves_allowlisted_detector_case():
    bindings = resolve_structured_return_bindings(
        subject_name="PeakDetector",
        methods=[_detect_method()],
        legacy_outputs=_structured_outputs("rpeaks", "quality"),
    )

    assert bindings is not None
    assert [binding.output_name for binding in bindings] == ["rpeaks", "quality"]
    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.source_attr for binding in bindings] == ["rpeaks", "quality"]


def test_return_shape_helper_fails_closed_on_partial_output_mismatch():
    bindings = resolve_structured_return_bindings(
        subject_name="PeakDetector",
        methods=[_detect_method()],
        legacy_outputs=_structured_outputs("rpeaks", "quality", "count"),
    )

    assert bindings is None


def test_chunker_uses_allowlisted_dict_field_bindings_for_detector_case():
    bindings = _infer_output_bindings(
        "PeakDetector",
        [_detect_method()],
        _structured_outputs("rpeaks", "quality"),
    )

    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.output_name for binding in bindings] == ["rpeaks", "quality"]
    assert [binding.source_method for binding in bindings] == ["detect", "detect"]


def test_chunker_preserves_conservative_fallback_for_non_allowlisted_case():
    bindings = _infer_output_bindings(
        "OtherDetector",
        [_detect_method()],
        _structured_outputs("rpeaks", "quality"),
    )

    assert [binding.binding_kind for binding in bindings] == ["unknown", "unknown"]


def test_emitter_renders_dict_field_extraction_for_allowlisted_case():
    atom = MacroAtomSpec(
        name="Peak Detector",
        method_names=["detect"],
        inputs=[IOSpec(name="signal", type_desc="list[float]")],
        outputs=_structured_outputs("rpeaks", "quality"),
        concept_type=ConceptType.CUSTOM,
    )
    operation = OperationSpec(
        operation_id="detect",
        display_name="Peak Detector",
        role="query",
        method_bindings=[
            MethodBinding(
                method_name="detect",
                signature=[
                    ParameterFact(name="self", provenance=_provenance()),
                    ParameterFact(name="signal", provenance=_provenance()),
                ],
            )
        ],
        direct_inputs=list(atom.inputs),
        emitted_outputs=[
            OutputBindingSpec(
                output_name="rpeaks",
                type_desc="list[int]",
                binding_kind="dict_field",
                source_method="detect",
                source_attr="rpeaks",
            ),
            OutputBindingSpec(
                output_name="quality",
                type_desc="float",
                binding_kind="dict_field",
                source_method="detect",
                source_attr="quality",
            ),
        ],
    )
    plan = ValidatedMacroPlan(
        plan=ProposedMacroPlan(
            macro_atoms=[atom],
            canonical_ir=IngestIRPlan(
                subject_name="PeakDetector",
                source_language="python",
                operations=[operation],
            ),
        ),
        all_attrs_accounted=True,
    )
    _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)

    source = generate_atom_wrappers(
        plan.plan.macro_atoms,
        plan.plan.state_models,
        witness_names,
        class_name="PeakDetector",
        source_file="detector.py",
        plan=plan,
    )

    assert "_ret_0 = obj.detect(signal)" in source
    assert "['rpeaks']" in source
    assert "['quality']" in source
