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
    return _method_fact("detect")


def _detect_events_method() -> MethodFact:
    return _method_fact("detect_events")


def _evaluate_method() -> MethodFact:
    return _method_fact("evaluate")


def _method_fact(name: str) -> MethodFact:
    prov = _provenance()
    return MethodFact(
        name=name,
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
        "onsets": IOSpec(name="onsets", type_desc="list[int]"),
        "confidence": IOSpec(name="confidence", type_desc="float"),
        "accepted": IOSpec(name="accepted", type_desc="bool"),
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


def test_return_shape_helper_resolves_allowlisted_onset_detector_case():
    bindings = resolve_structured_return_bindings(
        subject_name="OnsetDetector",
        methods=[_detect_events_method()],
        legacy_outputs=_structured_outputs("onsets", "confidence"),
    )

    assert bindings is not None
    assert [binding.output_name for binding in bindings] == ["onsets", "confidence"]
    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.source_method for binding in bindings] == ["detect_events", "detect_events"]
    assert [binding.source_attr for binding in bindings] == ["onsets", "confidence"]


def test_chunker_uses_allowlisted_dict_field_bindings_for_detector_case():
    bindings = _infer_output_bindings(
        "PeakDetector",
        [_detect_method()],
        _structured_outputs("rpeaks", "quality"),
    )

    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.output_name for binding in bindings] == ["rpeaks", "quality"]
    assert [binding.source_method for binding in bindings] == ["detect", "detect"]


def test_chunker_uses_allowlisted_dict_field_bindings_for_onset_detector_case():
    bindings = _infer_output_bindings(
        "OnsetDetector",
        [_detect_events_method()],
        _structured_outputs("onsets", "confidence"),
    )

    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.output_name for binding in bindings] == ["onsets", "confidence"]
    assert [binding.source_method for binding in bindings] == ["detect_events", "detect_events"]


def test_chunker_preserves_conservative_fallback_for_non_allowlisted_case():
    bindings = _infer_output_bindings(
        "OtherDetector",
        [_detect_method()],
        _structured_outputs("rpeaks", "quality"),
    )

    assert [binding.binding_kind for binding in bindings] == ["unknown", "unknown"]


def test_chunker_preserves_conservative_fallback_for_non_allowlisted_onset_subject():
    bindings = _infer_output_bindings(
        "OtherOnsetDetector",
        [_detect_events_method()],
        _structured_outputs("onsets", "confidence"),
    )

    assert [binding.binding_kind for binding in bindings] == ["unknown", "unknown"]
    assert [binding.output_name for binding in bindings] == ["onsets", "confidence"]


def test_return_shape_helper_resolves_allowlisted_sqi_detector_case():
    bindings = resolve_structured_return_bindings(
        subject_name="SQIDetector",
        methods=[_evaluate_method()],
        legacy_outputs=_structured_outputs("accepted", "quality"),
    )

    assert bindings is not None
    assert [binding.output_name for binding in bindings] == ["accepted", "quality"]
    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.source_attr for binding in bindings] == ["accepted", "quality"]


def test_chunker_uses_allowlisted_dict_field_bindings_for_sqi_detector_case():
    bindings = _infer_output_bindings(
        "SQIDetector",
        [_evaluate_method()],
        _structured_outputs("accepted", "quality"),
    )

    assert [binding.binding_kind for binding in bindings] == ["dict_field", "dict_field"]
    assert [binding.output_name for binding in bindings] == ["accepted", "quality"]
    assert [binding.source_method for binding in bindings] == ["evaluate", "evaluate"]


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


def test_emitter_renders_dict_field_extraction_for_allowlisted_onset_case():
    atom = MacroAtomSpec(
        name="Onset Detector",
        method_names=["detect_events"],
        inputs=[IOSpec(name="signal", type_desc="list[float]")],
        outputs=_structured_outputs("onsets", "confidence"),
        concept_type=ConceptType.CUSTOM,
    )
    operation = OperationSpec(
        operation_id="detect_events",
        display_name="Onset Detector",
        role="query",
        method_bindings=[
            MethodBinding(
                method_name="detect_events",
                signature=[
                    ParameterFact(name="self", provenance=_provenance()),
                    ParameterFact(name="signal", provenance=_provenance()),
                ],
            )
        ],
        direct_inputs=list(atom.inputs),
        emitted_outputs=[
            OutputBindingSpec(
                output_name="onsets",
                type_desc="list[int]",
                binding_kind="dict_field",
                source_method="detect_events",
                source_attr="onsets",
            ),
            OutputBindingSpec(
                output_name="confidence",
                type_desc="float",
                binding_kind="dict_field",
                source_method="detect_events",
                source_attr="confidence",
            ),
        ],
    )
    plan = ValidatedMacroPlan(
        plan=ProposedMacroPlan(
            macro_atoms=[atom],
            canonical_ir=IngestIRPlan(
                subject_name="OnsetDetector",
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
        class_name="OnsetDetector",
        source_file="detector.py",
        plan=plan,
    )

    assert "_ret_0 = obj.detect_events(signal)" in source
    assert "['onsets']" in source
    assert "['confidence']" in source


def test_emitter_renders_dict_field_extraction_for_allowlisted_sqi_case():
    atom = MacroAtomSpec(
        name="SQI Detector",
        method_names=["evaluate"],
        inputs=[IOSpec(name="signal", type_desc="list[float]")],
        outputs=_structured_outputs("accepted", "quality"),
        concept_type=ConceptType.CUSTOM,
    )
    operation = OperationSpec(
        operation_id="evaluate",
        display_name="SQI Detector",
        role="query",
        method_bindings=[
            MethodBinding(
                method_name="evaluate",
                signature=[
                    ParameterFact(name="self", provenance=_provenance()),
                    ParameterFact(name="signal", provenance=_provenance()),
                ],
            )
        ],
        direct_inputs=list(atom.inputs),
        emitted_outputs=[
            OutputBindingSpec(
                output_name="accepted",
                type_desc="bool",
                binding_kind="dict_field",
                source_method="evaluate",
                source_attr="accepted",
            ),
            OutputBindingSpec(
                output_name="quality",
                type_desc="float",
                binding_kind="dict_field",
                source_method="evaluate",
                source_attr="quality",
            ),
        ],
    )
    plan = ValidatedMacroPlan(
        plan=ProposedMacroPlan(
            macro_atoms=[atom],
            canonical_ir=IngestIRPlan(
                subject_name="SQIDetector",
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
        class_name="SQIDetector",
        source_file="detector.py",
        plan=plan,
    )

    assert "_ret_0 = obj.evaluate(signal)" in source
    assert "['accepted']" in source
    assert "['quality']" in source
