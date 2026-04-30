from __future__ import annotations

from sciona.physics_ingest.sources.wikidata import (
    DEFINING_FORMULA_PROPERTY_ID,
    HAS_USE_PROPERTY_ID,
    build_physical_equation_candidates_query,
    build_snapshot_record,
    build_wave0_candidate_records,
    entity_uri_to_id,
    parse_sparql_bindings,
    parse_sparql_response,
    property_uri_to_id,
    stable_payload_hash,
)


def _binding(value: str, kind: str = "literal") -> dict[str, str]:
    return {"type": kind, "value": value}


def _fixture_bindings() -> list[dict[str, dict[str, str]]]:
    return [
        {
            "item": _binding("http://www.wikidata.org/entity/Q1", "uri"),
            "itemLabel": _binding("Example law"),
            "itemDescription": _binding("a physical equation fixture"),
            "formulaProperty": _binding("http://www.wikidata.org/prop/direct/P2534", "uri"),
            "formula": _binding("E = m c^2"),
            "alias": _binding("mass energy relation"),
            "use": _binding("http://www.wikidata.org/entity/Q2", "uri"),
            "useLabel": _binding("relativistic mechanics"),
            "useDescription": _binding("physics use fixture"),
        },
        {
            "item": _binding("http://www.wikidata.org/entity/Q1", "uri"),
            "itemLabel": _binding("Example law"),
            "itemDescription": _binding("a physical equation fixture"),
            "formulaProperty": _binding("http://www.wikidata.org/prop/direct/P2534", "uri"),
            "formula": _binding("E = m c^2"),
            "alias": _binding("energy mass equivalence"),
            "use": _binding("http://www.wikidata.org/entity/Q3", "uri"),
            "useLabel": _binding("particle physics"),
            "useDescription": _binding("another use fixture"),
        },
        {
            "item": _binding("http://www.wikidata.org/entity/Q4", "uri"),
            "itemLabel": _binding("No use law"),
            "itemDescription": _binding("formula without use links"),
            "formulaProperty": _binding("http://www.wikidata.org/prop/direct/P2534", "uri"),
            "formula": _binding("F = m a"),
        },
    ]


def test_build_physical_equation_candidates_query_preserves_wikidata_ids() -> None:
    query = build_physical_equation_candidates_query(
        limit=25,
        item_qids=("Q1",),
        required_use_qids=("Q2",),
    )

    assert "VALUES ?formulaProperty { wdt:P2534 }" in query
    assert "VALUES ?item { wd:Q1 }" in query
    assert "VALUES ?requiredUse { wd:Q2 }" in query
    assert "?item wdt:P366 ?requiredUse" in query
    assert "?item skos:altLabel ?alias" in query
    assert "LIMIT 25" in query


def test_parse_sparql_bindings_groups_aliases_and_uses() -> None:
    candidates = parse_sparql_bindings(_fixture_bindings())

    assert len(candidates) == 2
    first = candidates[0]
    assert first.entity_id == "Q1"
    assert first.formula_property_id == DEFINING_FORMULA_PROPERTY_ID
    assert first.formula_text == "E = m c^2"
    assert first.aliases == ("energy mass equivalence", "mass energy relation")
    assert {use.entity_id for use in first.uses} == {"Q2", "Q3"}
    assert {use.property_id for use in first.uses} == {HAS_USE_PROPERTY_ID}
    assert len(first.source_rows) == 2


def test_parse_sparql_response_accepts_wikidata_json_shape() -> None:
    response = {"head": {"vars": []}, "results": {"bindings": _fixture_bindings()}}

    candidates = parse_sparql_response(response)

    assert [candidate.entity_id for candidate in candidates] == ["Q1", "Q4"]


def test_candidate_record_is_wave0_compatible() -> None:
    candidate = parse_sparql_bindings(_fixture_bindings())[0]

    record = candidate.to_wave0_candidate_record(snapshot_id="snapshot-1")

    assert record["snapshot_id"] == "snapshot-1"
    assert record["source_candidate_id"].startswith("Q1:P2534:")
    assert record["source_entity_uri"] == "http://www.wikidata.org/entity/Q1"
    assert record["source_label"] == "Example law"
    assert record["raw_formula"] == "E = m c^2"
    assert record["raw_formula_format"] == "wikidata_math"
    assert record["candidate_status"] == "raw_imported"
    assert record["parse_confidence"] == 0.0
    assert record["mechanism_tags"] == []
    assert record["behavioral_archetypes"] == []
    assert record["source_payload"]["wikidata_entity_id"] == "Q1"
    assert record["source_payload"]["formula_property_id"] == "P2534"
    assert len(record["source_payload"]["uses"]) == 2


def test_build_wave0_candidate_records_converts_response() -> None:
    response = {"head": {"vars": []}, "results": {"bindings": _fixture_bindings()}}

    records = build_wave0_candidate_records(response, snapshot_id="snapshot-1")

    assert [record["source_label"] for record in records] == ["Example law", "No use law"]
    assert {record["snapshot_id"] for record in records} == {"snapshot-1"}


def test_snapshot_record_has_stable_payload_hash() -> None:
    query = build_physical_equation_candidates_query(limit=1)
    response = {"results": {"bindings": _fixture_bindings()[:1]}}

    snapshot = build_snapshot_record(
        query=query,
        response=response,
        source_version="2026-04-30",
    )

    assert snapshot["source_system"] == "wikidata"
    assert snapshot["source_version"] == "2026-04-30"
    assert snapshot["adapter_name"] == "sciona.physics_ingest.sources.wikidata"
    assert len(snapshot["payload_sha256"]) == 64
    assert snapshot["payload_sha256"] == stable_payload_hash(snapshot["payload"])


def test_uri_helpers_accept_entity_and_property_uris_or_ids() -> None:
    assert entity_uri_to_id("http://www.wikidata.org/entity/Q42") == "Q42"
    assert entity_uri_to_id("wd:Q42") == "Q42"
    assert entity_uri_to_id("Q42") == "Q42"
    assert property_uri_to_id("http://www.wikidata.org/prop/direct/P2534") == "P2534"
    assert property_uri_to_id("http://www.wikidata.org/entity/P366") == "P366"
    assert property_uri_to_id("wdt:P2534") == "P2534"
    assert property_uri_to_id("P366") == "P366"
