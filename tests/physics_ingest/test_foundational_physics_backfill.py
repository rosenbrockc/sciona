from __future__ import annotations

import re

from sciona.physics_ingest.staging import stage_source_rows
from sciona.physics_ingest.sources.foundational_physics import (
    FOUNDATIONAL_LAW_SEEDS,
    build_foundational_physics_backfill_bundle,
    foundational_seed_domains,
)


SNAPSHOT_ID = "00000000-0000-0000-0000-000000000020"
EXPECTED_DOMAINS = (
    "diffusion_transport",
    "electromagnetism",
    "mechanics",
    "scaling_laws",
    "thermodynamics",
    "waves",
)


def test_foundational_physics_bundle_retains_all_seed_records() -> None:
    bundle = build_foundational_physics_backfill_bundle(
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id=SNAPSHOT_ID,
    )
    bundle_again = build_foundational_physics_backfill_bundle(
        retrieved_at="2026-04-30T00:00:00Z",
        snapshot_id=SNAPSHOT_ID,
    )

    seed_ids = [seed.source_id for seed in FOUNDATIONAL_LAW_SEEDS]
    candidate_ids = [row["source_candidate_id"] for row in bundle.candidate_rows]
    payload_ids = [
        record["source_id"] for record in bundle.snapshot_row["payload"]["records"]
    ]

    assert len(seed_ids) == 18
    assert candidate_ids == seed_ids
    assert payload_ids == seed_ids
    assert len(set(candidate_ids)) == len(candidate_ids)
    assert bundle.snapshot_row["source_system"] == "manual"
    assert (
        bundle.snapshot_row["adapter_name"]
        == "sciona.physics_ingest.sources.foundational_physics"
    )
    assert bundle.snapshot_row["payload"]["record_count"] == len(FOUNDATIONAL_LAW_SEEDS)
    assert bundle.snapshot_row["payload"]["domains"] == list(EXPECTED_DOMAINS)
    assert bundle.snapshot_row["payload"]["domain_counts"] == {
        "diffusion_transport": 3,
        "electromagnetism": 3,
        "mechanics": 3,
        "scaling_laws": 3,
        "thermodynamics": 3,
        "waves": 3,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", str(bundle.snapshot_row["payload_sha256"]))
    assert bundle.snapshot_row["payload_sha256"] == bundle_again.snapshot_row[
        "payload_sha256"
    ]


def test_foundational_physics_candidates_have_required_curated_metadata() -> None:
    bundle = build_foundational_physics_backfill_bundle(snapshot_id=SNAPSHOT_ID)

    for seed, candidate in zip(FOUNDATIONAL_LAW_SEEDS, bundle.candidate_rows):
        assert candidate["snapshot_id"] == SNAPSHOT_ID
        assert candidate["raw_formula"] == seed.raw_formula
        assert candidate["raw_formula_format"] == "plain_text"
        assert candidate["candidate_status"] == "raw_imported"
        assert candidate["parse_confidence"] == 0.0
        assert candidate["mechanism_tags"] == list(seed.mechanism_tags)
        assert candidate["behavioral_archetypes"] == list(
            seed.behavioral_archetypes
        )
        assert candidate["source_payload"]["variable_dimension_hints"] == dict(
            seed.variable_dimension_hints
        )
        assert candidate["source_payload"]["references"]
        assert candidate["source_payload"]["provenance"]["curation_method"] == (
            "manual_seed"
        )


def test_stage_source_rows_validates_foundational_physics_candidates() -> None:
    bundle = build_foundational_physics_backfill_bundle()

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )

    assert staged_snapshot.source_system == "manual"
    assert len(staged_candidates) == len(FOUNDATIONAL_LAW_SEEDS)
    assert [candidate.source_candidate_id for candidate in staged_candidates] == [
        seed.source_id for seed in FOUNDATIONAL_LAW_SEEDS
    ]
    assert staged_candidates[0].source_payload["domain"] == "mechanics"
    assert staged_candidates[-1].source_payload["domain"] == "scaling_laws"


def test_foundational_seed_domains_reports_expected_coverage() -> None:
    assert foundational_seed_domains() == EXPECTED_DOMAINS
