"""Tests for local skeleton-family assets and compatibility resolution."""

from __future__ import annotations

from sciona.architect.models import ConceptType
from sciona.architect.skeleton_assets import load_local_skeleton_assets
from sciona.architect.skeletons import get_skeleton


class TestLocalSkeletonAssets:
    def test_loads_local_assets_with_audit_metadata(self):
        assets = load_local_skeleton_assets()
        by_id = {asset.asset_id: asset for asset in assets}

        assert "family.divide_and_conquer.v1" in by_id
        assert "signal_detect_measure" in by_id
        assert by_id["family.divide_and_conquer.v1"].audit.review_status == "transitional"
        assert by_id["signal_detect_measure"].audit.rationale

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
