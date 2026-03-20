"""Tests for the provenance graph schema and models."""

from __future__ import annotations

import pytest

from sciona.provenance.models import Bounty, CDGSubmission, Originator
from sciona.provenance.schema import (
    PROVENANCE_CONSTRAINTS,
    PROVENANCE_INDEXES,
    build_authored_by_params,
    build_bounty_params,
    build_depends_on_params,
    build_derives_from_params,
    build_originator_params,
    build_solved_by_params,
    build_submission_params,
)


class TestModels:
    def test_originator_round_trip(self):
        orig = Originator(originator_id="org1", github_username="alice")
        data = orig.model_dump()
        restored = Originator.model_validate(data)
        assert restored == orig

    def test_originator_defaults(self):
        orig = Originator(originator_id="org1")
        assert orig.github_username == ""
        assert orig.affiliation == ""

    def test_bounty_round_trip(self):
        b = Bounty(bounty_id="b1", escrow_amount=100.0, status="open")
        data = b.model_dump()
        restored = Bounty.model_validate(data)
        assert restored == b

    def test_bounty_defaults(self):
        b = Bounty(bounty_id="b1")
        assert b.escrow_amount == 0.0
        assert b.status == "open"
        assert b.verification_budget == 5

    def test_submission_round_trip(self):
        s = CDGSubmission(cdg_id="cdg1", topo_hash="abc")
        data = s.model_dump()
        restored = CDGSubmission.model_validate(data)
        assert restored == s

    def test_submission_defaults(self):
        s = CDGSubmission(cdg_id="cdg1", topo_hash="abc")
        assert s.verified is False
        assert s.created_at == ""


class TestParamBuilders:
    def test_originator_params(self):
        params = build_originator_params("org1", github_username="alice")
        assert params["originator_id"] == "org1"
        assert params["github_username"] == "alice"
        assert "affiliation" in params

    def test_bounty_params(self):
        params = build_bounty_params("b1", escrow_amount=50.0, status="submitted")
        assert params["bounty_id"] == "b1"
        assert params["escrow_amount"] == 50.0
        assert params["status"] == "submitted"

    def test_submission_params(self):
        params = build_submission_params("cdg1", topo_hash="hash1", verified=True)
        assert params["cdg_id"] == "cdg1"
        assert params["topo_hash"] == "hash1"
        assert params["verified"] is True

    def test_authored_by_params(self):
        params = build_authored_by_params("repo.atom1", "org1", 0.5)
        assert params["atom_fqn"] == "repo.atom1"
        assert params["originator_id"] == "org1"
        assert params["contribution_share"] == 0.5

    def test_depends_on_params(self):
        params = build_depends_on_params("cdg1", "repo.atom1", "sha256abc")
        assert params["cdg_id"] == "cdg1"
        assert params["atom_fqn"] == "repo.atom1"
        assert params["content_hash"] == "sha256abc"

    def test_solved_by_params(self):
        params = build_solved_by_params("b1", "cdg1", 0.95)
        assert params["bounty_id"] == "b1"
        assert params["cdg_id"] == "cdg1"
        assert params["metric_value"] == 0.95

    def test_derives_from_params(self):
        params = build_derives_from_params("cdg2", "cdg1")
        assert params["child_cdg_id"] == "cdg2"
        assert params["parent_cdg_id"] == "cdg1"


class TestConstraintsAndIndexes:
    def test_constraints_are_create_constraint(self):
        for stmt in PROVENANCE_CONSTRAINTS:
            assert stmt.startswith("CREATE CONSTRAINT")

    def test_indexes_are_create_index(self):
        for stmt in PROVENANCE_INDEXES:
            assert stmt.startswith("CREATE INDEX")

    def test_constraint_count(self):
        assert len(PROVENANCE_CONSTRAINTS) == 3

    def test_index_count(self):
        assert len(PROVENANCE_INDEXES) == 4
