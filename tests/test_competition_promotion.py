"""Synthetic intake contracts; no competition records or dataset fixtures."""

import json
import os
from dataclasses import replace
from uuid import uuid4

import pytest

from sciona.competition_promotion import (
    assess_implementation_intake,
    load_competition_candidates,
    assess_keyword_call_contract,
)
from sciona.competition_import import (
    build_competition_import_plan, import_competition_drafts, verify_competition_import,
)
from sciona.competition_graph import restore_competition_graph, load_competition_graph


def candidate(tmp_path, *, status="active", action="replace_stage", target="synthetic.scale", pin="v1"):
    template = {
        "asset_id": "solution.synthetic", "asset_version": "1.0.0",
        "family": "synthetic", "paradigm": "arithmetic", "name": "Scale",
        "summary": "Scale a vector", "dejargonized_summary": "Multiply values",
        "stages": [{"stage_id": "scale", "name": "Scale", "description": "Multiply values", "concept_type": "arithmetic"}],
        "edges": [], "audit": {"references": [{"title": "Synthetic contract"}]},
        "planning_constraints": [
            {"category": category, "statement": "Synthetic requirement"}
            for category in ["data", "validation", "resource", "metric"]
        ],
        "applicability": {"use_when": ["synthetic test"]},
    }
    bindings = {"bindings": [{
        "stage_id": "scale", "status": status, "action_class": action,
        "bound_artifact_fqdn": target, "bound_version_content_hash": pin,
    }]}
    (tmp_path / "synthetic.json").write_text(json.dumps(template))
    (tmp_path / "synthetic_bindings.json").write_text(json.dumps(bindings))
    return load_competition_candidates(tmp_path)[0]


CATALOG = {"synthetic.scale": {"status": "approved", "is_publishable": True, "content_hash": "v1"}}


def test_keyword_contract_requires_real_parameter_mapping():
    def operation(values, *, scale=2):
        return values * scale

    assert assess_keyword_call_contract(["values"], operation)["keyword_compatible"]
    report = assess_keyword_call_contract(["features"], operation)
    assert report["missing_required_parameters"] == ["values"]
    assert report["unexpected_parameters"] == ["features"]
    assert not report["keyword_compatible"]


def test_keyword_contract_honors_kwargs_but_rejects_positional_only_requirements():
    def extensible(values, **kwargs):
        return values

    def positional(values, /, **kwargs):
        return values

    assert assess_keyword_call_contract(["values", "context"], extensible)["keyword_compatible"]
    assert not assess_keyword_call_contract(["values"], positional)["keyword_compatible"]
    assert not assess_keyword_call_contract(["values", "values"], extensible)["keyword_compatible"]


def test_intake_preserves_constraints_and_original_fields(tmp_path):
    c = candidate(tmp_path)
    assert [x.category.value for x in c.asset.planning_constraints] == ["data", "validation", "resource", "metric"]
    assert c.original_template["applicability"] == {"use_when": ["synthetic test"]}
    report = assess_implementation_intake(c, CATALOG)
    assert report["dependency_intake_complete"] is True
    assert "publishable" not in report


@pytest.mark.parametrize("action", ["orchestration", "trivial_inline", "external_tool", "external_knowledge"])
def test_noncomputational_labels_do_not_resolve_gaps(tmp_path, action):
    report = assess_implementation_intake(candidate(tmp_path, status="gap", action=action), CATALOG)
    assert report["implementation_coverage"] == 0
    assert report["blockers"] == {"unresolved_binding": 1}


@pytest.mark.parametrize("kwargs,blocker", [
    ({"status": "approximate"}, "approximate_binding"),
    ({"target": ""}, "missing_implementation"),
    ({"target": "unknown"}, "target_not_in_catalog"),
    ({"pin": ""}, "dependency_version_unpinned"),
    ({"pin": "old"}, "dependency_version_not_current"),
])
def test_incomplete_dependencies_cannot_pass_intake(tmp_path, kwargs, blocker):
    report = assess_implementation_intake(candidate(tmp_path, **kwargs), CATALOG)
    assert report["dependency_intake_complete"] is False
    assert report["blockers"] == {blocker: 1}


@pytest.mark.parametrize("state", [
    {"status": "draft", "is_publishable": True},
    {"status": "approved", "is_publishable": False},
])
def test_dependency_must_be_approved_and_publishable(tmp_path, state):
    report = assess_implementation_intake(candidate(tmp_path), {"synthetic.scale": state})
    assert report["blockers"] == {"target_not_served": 1}


def test_duplicate_bindings_fail_instead_of_last_writer_winning(tmp_path):
    c = candidate(tmp_path)
    c.original_bindings["bindings"] *= 2
    assert assess_implementation_intake(c, CATALOG)["blockers"] == {"ambiguous_binding": 1}


def test_import_snapshot_preserves_unrepresentable_binding_states(tmp_path):
    c = candidate(tmp_path, status="approximate", action="orchestration")
    plan = build_competition_import_plan(c)
    snapshot = plan.rows["artifact_audit_evidence"][0]["details"]["snapshot"]
    assert snapshot["template"] == c.original_template
    assert snapshot["bindings"] == c.original_bindings
    assert plan.rows["artifact_cdg_bindings"] == []
    assert plan.artifact["is_publishable"] is False
    assert plan.artifact["verified_leaf_coverage"] == 0
    assert build_competition_import_plan(c) == plan


@pytest.fixture
def promotion_database():
    url = os.environ.get("SCIONA_PROMOTION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires explicit promotion integration-test database")
    import psycopg
    from psycopg.rows import dict_row
    connection = psycopg.connect(url, row_factory=dict_row)
    # Open an outer transaction; importer transactions become savepoints.
    connection.execute("SELECT 1")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def unique_candidate(tmp_path):
    c = candidate(tmp_path)
    raw = {**c.original_template, "asset_id": "solution.synthetic." + uuid4().hex}
    return replace(c, original_template=raw, asset=c.asset.model_validate(raw))


def test_database_intake_roundtrip_idempotence_and_history(tmp_path, promotion_database):
    c = unique_candidate(tmp_path)
    plan = build_competition_import_plan(c)
    db = promotion_database
    assert import_competition_drafts(db, [plan]) == {"imported": 1, "unchanged": 0}
    assert import_competition_drafts(db, [plan]) == {"imported": 0, "unchanged": 1}
    changed_raw = {**c.original_template, "summary": "Revised synthetic contract"}
    changed = replace(c, original_template=changed_raw, asset=c.asset.model_validate(changed_raw))
    second = build_competition_import_plan(changed)
    assert import_competition_drafts(db, [second])["imported"] == 1
    versions = db.execute("SELECT is_latest FROM artifact_versions WHERE artifact_id=%s", (plan.artifact["artifact_id"],)).fetchall()
    assert len(versions) == 2
    assert sum(r["is_latest"] for r in versions) == 1
    verify_competition_import(db, plan)
    verify_competition_import(db, second)
    for expected in (plan, second):
        graph = load_competition_graph(db, version_id=expected.rows["artifact_versions"][0]["version_id"])
        assert graph.metadata["artifact_content_hash"] == expected.rows["artifact_versions"][0]["content_hash"]
        assert graph.metadata["execution_ready"] is False
    assert import_competition_drafts(db, [plan])["unchanged"] == 1
    latest = db.execute("SELECT content_hash FROM artifact_versions WHERE artifact_id=%s AND is_latest", (plan.artifact["artifact_id"],)).fetchone()
    assert latest["content_hash"] == second.rows["artifact_versions"][0]["content_hash"]


def test_database_intake_rolls_back_entire_batch(tmp_path, promotion_database):
    first = build_competition_import_plan(unique_candidate(tmp_path))
    second = build_competition_import_plan(unique_candidate(tmp_path))
    plans = sorted([first, second], key=lambda p: p.artifact["fqdn"])
    plans[1].artifact["artifact_kind"] = "invalid_kind"
    import psycopg
    with pytest.raises(psycopg.errors.CheckViolation):
        import_competition_drafts(promotion_database, plans)
    rows = promotion_database.execute("SELECT artifact_id FROM artifacts WHERE artifact_id=ANY(%s::uuid[])", ([p.artifact["artifact_id"] for p in plans],)).fetchall()
    assert rows == []


def test_database_intake_refuses_to_replace_approval(tmp_path, promotion_database):
    c = unique_candidate(tmp_path)
    plan = build_competition_import_plan(c)
    db = promotion_database
    import_competition_drafts(db, [plan])
    db.execute("UPDATE artifacts SET status='approved' WHERE artifact_id=%s", (plan.artifact["artifact_id"],))
    db.execute(
        "INSERT INTO artifact_audit_evidence(artifact_id,version_id,audit_type,passed,runner_version) "
        "VALUES(%s,%s,'smoke_test',true,'synthetic-review.v1')",
        (plan.artifact["artifact_id"], plan.rows["artifact_versions"][0]["version_id"]),
    )
    # Later independent review evidence is not owned or erased by intake.
    assert import_competition_drafts(db, [plan])["unchanged"] == 1
    raw = {**c.original_template, "summary": "Synthetic new version"}
    changed = build_competition_import_plan(replace(c, original_template=raw, asset=c.asset.model_validate(raw)))
    with pytest.raises(ValueError, match="protected artifact"):
        import_competition_drafts(db, [changed])
    assert db.execute("SELECT status FROM artifacts WHERE artifact_id=%s", (plan.artifact["artifact_id"],)).fetchone()["status"] == "approved"


def test_database_intake_detects_projection_drift(tmp_path, promotion_database):
    plan = build_competition_import_plan(unique_candidate(tmp_path))
    db = promotion_database
    import_competition_drafts(db, [plan])
    db.execute("UPDATE artifact_cdg_nodes SET description='unexpected edit' WHERE version_id=%s", (plan.rows["artifact_versions"][0]["version_id"],))
    with pytest.raises(ValueError, match="field mismatch"):
        import_competition_drafts(db, [plan])


def test_snapshot_reconstruction_retains_contracts_and_rejects_stale_hash(tmp_path):
    c = candidate(tmp_path)
    c.original_template["stages"][0]["inputs"] = [{"name": "values", "type_desc": "np.ndarray", "constraints": "finite"}]
    plan = build_competition_import_plan(c)
    snapshot = plan.rows["artifact_audit_evidence"][0]["details"]["snapshot"]
    digest = plan.rows["artifact_versions"][0]["content_hash"]
    graph = restore_competition_graph(snapshot, content_hash=digest)
    assert graph.nodes[0].inputs[0].constraints == "finite"
    assert graph.metadata["proposed_bindings"] == c.original_bindings
    assert graph.metadata["planning_constraints"] == c.original_template["planning_constraints"]
    assert graph.nodes[0].matched_primitive is None
    with pytest.raises(ValueError, match="requested version"):
        restore_competition_graph(snapshot, content_hash="stale")


@pytest.mark.anyio
async def test_intake_snapshot_cannot_execute_even_if_readiness_flag_is_edited(tmp_path, monkeypatch):
    from sciona.visualizer.runner import CDGExecutionSession
    monkeypatch.setattr("sciona.visualizer.runner.RUNS_DIR", tmp_path)
    plan = build_competition_import_plan(candidate(tmp_path))
    graph = restore_competition_graph(
        plan.rows["artifact_audit_evidence"][0]["details"]["snapshot"],
        content_hash=plan.rows["artifact_versions"][0]["content_hash"],
    )
    graph.metadata["execution_ready"] = True
    with pytest.raises(ValueError, match="validated implementation version"):
        await CDGExecutionSession(driver=None, repo="synthetic", run_id="intake").execute({}, cdg=graph)
