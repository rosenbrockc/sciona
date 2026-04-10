from __future__ import annotations

import pytest

from sciona.usability_registries import (
    UsabilityFamilyRegistry,
    UsabilityRegistryAudit,
    UsabilityRegistryEntry,
    known_usability_reason_codes,
    load_local_usability_registries,
    load_local_usability_registries_by_family,
    resolve_local_usability_registry,
    usability_registry_summary,
)
from sciona.usability import (
    UsabilityProvenanceKind,
    UsabilityReasonKind,
    UsabilityScope,
)


def test_load_local_usability_registries_includes_signal_and_neutral_fixtures() -> None:
    registries = load_local_usability_registries()
    families = {registry.family for registry in registries}

    assert "signal_processing" in families
    assert "divide_and_conquer" in families


def test_signal_registry_stays_cross_family() -> None:
    registry = resolve_local_usability_registry("signal_processing")
    assert registry is not None

    assert registry.asset_id == "family.signal_processing.usability.v1"
    assert registry.audit.review_status == "transitional"
    assert set(registry.family_aliases) == {"signal_event_rate", "signal_detect_measure"}
    assert {entry.reason_code for entry in registry.entries} == {
        "required_input_missing",
        "quality_instability",
        "timing_context_incoherent",
    }
    assert all(entry.reason_code in known_usability_reason_codes() for entry in registry.entries)
    assert all(entry.family_notes for entry in registry.entries)

    summary = usability_registry_summary(registry)
    assert summary["family"] == "signal_processing"
    assert summary["reason_count"] == 3


def test_neutral_registry_proves_portability() -> None:
    registry = resolve_local_usability_registry("recursive_split_merge")
    assert registry is not None
    assert registry.family == "divide_and_conquer"
    assert registry.audit.review_status == "draft"
    assert {entry.reason_code for entry in registry.entries} == {
        "coverage_insufficient",
        "required_reference_missing",
        "review_recommended",
    }
    assert all(entry.family_notes for entry in registry.entries)

    by_family = load_local_usability_registries_by_family()
    assert by_family["graph_partition"].family == "divide_and_conquer"


def test_usability_registry_summary_handles_dict_payloads() -> None:
    summary = usability_registry_summary(
        {
            "asset_id": "family.signal_processing.usability.v1",
            "asset_version": "phase2.v1",
            "family": "signal_processing",
            "family_aliases": ["signal_event_rate"],
            "entries": [{}, {}],
            "review_status": "transitional",
            "source_kind": "local_asset",
        }
    )
    assert summary["reason_count"] == 2
    assert summary["family_aliases"] == ["signal_event_rate"]


def test_registry_entry_rejects_family_local_redefinition_of_shared_meaning() -> None:
    with pytest.raises(ValueError, match="redefine shared canonical meaning"):
        UsabilityRegistryEntry(
            reason_code="required_input_missing",
            reason_kind=UsabilityReasonKind.BLOCKING,
            supported_scopes=[UsabilityScope.GUIDANCE],
            sanctioned_provenance_kinds=[UsabilityProvenanceKind.RUNTIME_ASSESSOR],
            admissibility_notes=["Use when required context is unavailable."],
            escalation_conditions=["Escalate when the candidate cannot be evaluated."],
            family_notes=[
                "This entry redefines the shared meaning so it means only missing recursion depth."
            ],
            references=[{"title": "Registry reference"}],
        )


def test_registry_entry_requires_supported_scopes_provenance_and_references() -> None:
    with pytest.raises(ValueError, match="supported scopes"):
        UsabilityRegistryEntry(
            reason_code="coverage_insufficient",
            reason_kind=UsabilityReasonKind.BLOCKING,
            supported_scopes=[],
            sanctioned_provenance_kinds=[UsabilityProvenanceKind.RUNTIME_ASSESSOR],
            admissibility_notes=["Use when coverage is too limited."],
            escalation_conditions=["Escalate when the result would be untrustworthy."],
            family_notes=["In this family the issue is limited branch coverage."],
            references=[{"title": "Registry reference"}],
        )

    with pytest.raises(ValueError, match="sanctioned provenance kinds"):
        UsabilityRegistryEntry(
            reason_code="coverage_insufficient",
            reason_kind=UsabilityReasonKind.BLOCKING,
            supported_scopes=[UsabilityScope.GUIDANCE],
            sanctioned_provenance_kinds=[],
            admissibility_notes=["Use when coverage is too limited."],
            escalation_conditions=["Escalate when the result would be untrustworthy."],
            family_notes=["In this family the issue is limited branch coverage."],
            references=[{"title": "Registry reference"}],
        )

    with pytest.raises(ValueError, match="at least one reference"):
        UsabilityRegistryEntry(
            reason_code="coverage_insufficient",
            reason_kind=UsabilityReasonKind.BLOCKING,
            supported_scopes=[UsabilityScope.GUIDANCE],
            sanctioned_provenance_kinds=[UsabilityProvenanceKind.RUNTIME_ASSESSOR],
            admissibility_notes=["Use when coverage is too limited."],
            escalation_conditions=["Escalate when the result would be untrustworthy."],
            family_notes=["In this family the issue is limited branch coverage."],
            references=[],
        )


def test_registry_audit_requires_rationale_provenance_and_uncertainty_for_transitional_states() -> None:
    with pytest.raises(ValueError, match="include provenance"):
        UsabilityRegistryAudit(
            source_kind="local_asset",
            review_status="draft",
            rationale="Cross-family fixture rationale.",
            dejargonized_summary="Cross-family fixture summary.",
            uncertainty_notes=["Registry is still under review."],
            references=[{"title": "Registry reference"}],
            maintainers=["ageo-matcher"],
        )

    with pytest.raises(ValueError, match="uncertainty notes"):
        UsabilityRegistryAudit(
            provenance="repo_local_transitional_asset",
            source_kind="local_asset",
            review_status="transitional",
            rationale="Cross-family fixture rationale.",
            dejargonized_summary="Cross-family fixture summary.",
            references=[{"title": "Registry reference"}],
            maintainers=["ageo-matcher"],
        )


def test_registry_rejects_duplicate_family_aliases() -> None:
    with pytest.raises(ValueError, match="duplicate family aliases"):
        UsabilityFamilyRegistry(
            asset_id="family.generic.usability.v1",
            asset_version="v1",
            family="generic",
            family_aliases=["shared_family", "shared_family"],
            name="Generic Usability Registry",
            summary="Family-local interpretation of canonical usability reasons.",
            dejargonized_summary="This registry explains generic usability decisions.",
            entries=[
                UsabilityRegistryEntry(
                    reason_code="review_recommended",
                    reason_kind=UsabilityReasonKind.WARNING,
                    supported_scopes=[UsabilityScope.GUIDANCE],
                    sanctioned_provenance_kinds=[
                        UsabilityProvenanceKind.MANUAL_REVIEW
                    ],
                    admissibility_notes=["Use when review is prudent but not mandatory."],
                    escalation_conditions=["Escalate when support remains thin."],
                    family_notes=[
                        "In this family the artifact may still be useful but a review step is prudent."
                    ],
                    references=[{"title": "Registry reference"}],
                )
            ],
            audit=UsabilityRegistryAudit(
                provenance="repo_local_transitional_asset",
                source_kind="local_asset",
                review_status="draft",
                rationale="Cross-family fixture rationale.",
                dejargonized_summary="Cross-family fixture summary.",
                uncertainty_notes=["Registry is still under review."],
                references=[{"title": "Registry reference"}],
                maintainers=["ageo-matcher"],
            ),
        )
