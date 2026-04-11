"""Tests for local skeleton-family assets and compatibility resolution."""

from __future__ import annotations

from sciona.asset_migration import MigrationReadinessAsset
from sciona.architect.models import ConceptType
from sciona.architect.skeleton_assets import (
    load_local_skeleton_assets,
    load_local_skeleton_graphs,
    skeleton_asset_summary,
)
from sciona.architect.skeletons import get_skeleton


class TestLocalSkeletonAssets:
    def test_loads_local_assets_with_audit_metadata(self):
        assets = load_local_skeleton_assets()
        by_id = {asset.asset_id: asset for asset in assets}

        assert "family.divide_and_conquer.v1" in by_id
        assert "family.sequential_filter.v1" in by_id
        assert "signal_detect_measure" in by_id
        assert by_id["family.divide_and_conquer.v1"].audit.review_status == "transitional"
        assert by_id["family.sequential_filter.v1"].family == "sequential_filter"
        assert by_id["family.sequential_filter.v1"].variant_hints == [
            "kalman_filter",
            "particle_filter",
        ]
        assert by_id["signal_detect_measure"].audit.rationale
        readiness = by_id["family.divide_and_conquer.v1"].audit.migration_readiness
        assert readiness.status == "in_progress"
        assert readiness.target_repository == "../sciona-atoms"
        assert readiness.required_check_count() == 3
        assert readiness.completed_required_check_count() == 2
        assert readiness.is_ready_for_migration() is False

    def test_skeleton_asset_summary_includes_migration_readiness(self):
        assets = load_local_skeleton_assets()
        asset = next(asset for asset in assets if asset.asset_id == "family.divide_and_conquer.v1")

        summary = skeleton_asset_summary(asset)

        assert summary["migration_readiness_status"] == "in_progress"
        assert summary["migration_readiness_ready"] is False
        assert summary["migration_readiness_required_check_count"] == 3
        assert "cross_family_portability" in summary["migration_readiness_check_ids"]

    def test_migration_readiness_asset_requires_satisfied_checks_when_ready(self):
        readiness = MigrationReadinessAsset.model_validate(
            {
                "status": "ready_for_migration",
                "target_repository": "../sciona-atoms",
                "checklist": [
                    {
                        "check_id": "schema",
                        "description": "Stable schema",
                        "required": True,
                        "satisfied": True,
                    },
                    {
                        "check_id": "docs",
                        "description": "Docs complete",
                        "required": True,
                        "satisfied": True,
                    },
                ],
            }
        )

        assert readiness.is_ready_for_migration() is True

    def test_canonical_family_asset_can_override_paradigm_default(self):
        skeleton = get_skeleton(ConceptType.DIVIDE_AND_CONQUER, variant="merge_sort")

        assert skeleton is not None
        assert skeleton.metadata["asset"]["asset_id"] == "family.divide_and_conquer.v1"
        assert skeleton.metadata["asset"]["asset_version"] == "v1"

    def test_subfamily_asset_requires_explicit_variant(self):
        default_signal = get_skeleton(ConceptType.SIGNAL_FILTER)
        hr_signal = get_skeleton(
            ConceptType.SIGNAL_FILTER,
            variant="bandpass_hr_detection",
        )

        assert default_signal is not None
        assert "asset" not in default_signal.metadata
        assert hr_signal is not None
        assert hr_signal.metadata["asset"]["asset_id"] == "signal_detect_measure"

    def test_sequential_filter_asset_is_discoverable_by_paradigm_and_variant_hint(self):
        by_paradigm, by_name = load_local_skeleton_graphs()

        sequential = by_paradigm[ConceptType.SEQUENTIAL_FILTER]

        assert sequential.metadata["asset"]["asset_id"] == "family.sequential_filter.v1"
        assert by_name["kalman_filter"].metadata["asset"]["asset_id"] == "family.sequential_filter.v1"
        assert by_name["particle_filter"].metadata["asset"]["asset_id"] == "family.sequential_filter.v1"

    def test_sequential_filter_asset_is_first_class_skeleton_lookup(self):
        default_filter = get_skeleton(ConceptType.SEQUENTIAL_FILTER)
        kalman_hint = get_skeleton(ConceptType.SEQUENTIAL_FILTER, variant="kalman_filter")
        particle_hint = get_skeleton(
            ConceptType.SEQUENTIAL_FILTER,
            variant="particle_filter",
        )

        assert default_filter is not None
        assert kalman_hint is not None
        assert particle_hint is not None
        assert default_filter.metadata["asset"]["asset_id"] == "family.sequential_filter.v1"
        assert kalman_hint.metadata["asset"]["asset_id"] == "family.sequential_filter.v1"
        assert particle_hint.metadata["asset"]["asset_id"] == "family.sequential_filter.v1"
