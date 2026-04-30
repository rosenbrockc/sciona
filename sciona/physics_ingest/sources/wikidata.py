"""Wikidata equation discovery adapter.

This module is intentionally QUDT-independent.  It discovers and preserves
Wikidata-native equation evidence so a later normalization stage can parse
formulae, resolve dimensions, and publish reviewed symbolic artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence
from urllib import parse, request


WIKIDATA_ENTITY_BASE = "http://www.wikidata.org/entity/"
WIKIDATA_ENTITY_HTTPS_BASE = "https://www.wikidata.org/entity/"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

DEFINING_FORMULA_PROPERTY_ID = "P2534"
HAS_USE_PROPERTY_ID = "P366"
ADAPTER_NAME = "sciona.physics_ingest.sources.wikidata"
ADAPTER_VERSION = "0.1.0"


def build_physical_equation_candidates_query(
    *,
    limit: int = 500,
    language: str = "en",
    formula_property_ids: Sequence[str] = (DEFINING_FORMULA_PROPERTY_ID,),
    required_use_qids: Sequence[str] = (),
    item_qids: Sequence[str] = (),
) -> str:
    """Build a SPARQL query for Wikidata items with equation-like formulae.

    The default query is deliberately broad: it asks for items carrying a
    defining formula and preserves labels, aliases, descriptions, and
    ``has use`` links when available.  Callers can narrow discovery by passing
    explicit item QIDs or required use QIDs; low-priority results should remain
    raw candidates rather than being dropped here.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not formula_property_ids:
        raise ValueError("at least one formula property id is required")

    formula_values = _sparql_values("formulaProperty", "wdt", formula_property_ids)
    clauses = [
        formula_values,
        "?item ?formulaProperty ?formula .",
    ]
    if item_qids:
        clauses.insert(0, _sparql_values("item", "wd", item_qids))
    if required_use_qids:
        clauses.append(_sparql_values("requiredUse", "wd", required_use_qids))
        clauses.append("?item wdt:P366 ?requiredUse .")

    where = "\n  ".join(clauses)
    lang = _escape_sparql_string(language)
    return f"""SELECT ?item ?itemLabel ?itemDescription ?formulaProperty ?formula ?alias ?use ?useLabel ?useDescription WHERE {{
  {where}
  OPTIONAL {{ ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "{lang}") }}
  OPTIONAL {{ ?item wdt:P366 ?use . }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{lang},mul,en".
  }}
}} LIMIT {int(limit)}"""


def execute_sparql_query(
    query: str,
    *,
    endpoint: str = WIKIDATA_SPARQL_ENDPOINT,
    user_agent: str = "sciona-physics-ingest/0.1",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute a SPARQL query against Wikidata and return the JSON response."""

    params = parse.urlencode({"query": query, "format": "json"}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=params,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class WikidataUseRelationship:
    """A Wikidata ``has use`` target preserved from discovery rows."""

    property_id: str
    entity_id: str
    entity_uri: str
    label: str = ""
    description: str = ""

    def to_payload(self) -> dict[str, str]:
        return {
            "property_id": self.property_id,
            "entity_id": self.entity_id,
            "entity_uri": self.entity_uri,
            "label": self.label,
            "description": self.description,
        }


@dataclass(frozen=True)
class WikidataEquationCandidate:
    """A raw Wikidata equation candidate compatible with Wave 0 ingestion."""

    entity_id: str
    entity_uri: str
    label: str
    description: str
    formula_property_id: str
    formula_text: str
    formula_format: str = "wikidata_math"
    aliases: tuple[str, ...] = ()
    uses: tuple[WikidataUseRelationship, ...] = ()
    source_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple, repr=False)

    @property
    def source_candidate_id(self) -> str:
        formula_hash = sha256(self.formula_text.encode("utf-8")).hexdigest()[:16]
        return f"{self.entity_id}:{self.formula_property_id}:{formula_hash}"

    def to_wave0_candidate_record(
        self,
        *,
        snapshot_id: str | None = None,
        candidate_status: str = "raw_imported",
        parse_confidence: float = 0.0,
        priority_score: float = 0.0,
    ) -> dict[str, Any]:
        """Render this candidate as a row for ``physics_equation_candidates``."""

        record: dict[str, Any] = {
            "source_candidate_id": self.source_candidate_id,
            "source_entity_uri": self.entity_uri,
            "source_label": self.label,
            "source_description": self.description,
            "raw_formula": self.formula_text,
            "raw_formula_format": self.formula_format,
            "candidate_status": candidate_status,
            "parse_confidence": parse_confidence,
            "priority_score": priority_score,
            "mechanism_tags": [],
            "behavioral_archetypes": [],
            "source_payload": {
                "source_system": "wikidata",
                "wikidata_entity_id": self.entity_id,
                "formula_property_id": self.formula_property_id,
                "aliases": list(self.aliases),
                "uses": [use.to_payload() for use in self.uses],
                "source_rows": list(self.source_rows),
            },
            "notes": "",
        }
        if snapshot_id is not None:
            record["snapshot_id"] = snapshot_id
        return record


def parse_sparql_response(response: Mapping[str, Any]) -> list[WikidataEquationCandidate]:
    """Parse a Wikidata SPARQL JSON response into grouped equation candidates."""

    bindings = response.get("results", {}).get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("SPARQL response results.bindings must be a list")
    return parse_sparql_bindings(bindings)


def build_wave0_candidate_records(
    response: Mapping[str, Any],
    *,
    snapshot_id: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a SPARQL response directly into Wave 0 candidate row dictionaries."""

    return [
        candidate.to_wave0_candidate_record(snapshot_id=snapshot_id)
        for candidate in parse_sparql_response(response)
    ]


def parse_sparql_bindings(
    bindings: Iterable[Mapping[str, Mapping[str, str]]],
) -> list[WikidataEquationCandidate]:
    """Parse SPARQL result bindings into de-duplicated candidates."""

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in bindings:
        entity_uri = _value(row, "item")
        formula = _value(row, "formula")
        if not entity_uri or not formula:
            continue

        entity_id = entity_uri_to_id(entity_uri)
        formula_property_id = property_uri_to_id(_value(row, "formulaProperty"))
        if not formula_property_id:
            formula_property_id = DEFINING_FORMULA_PROPERTY_ID
        key = (entity_id, formula_property_id, formula)
        bucket = grouped.setdefault(
            key,
            {
                "entity_id": entity_id,
                "entity_uri": entity_uri,
                "label": _value(row, "itemLabel"),
                "description": _value(row, "itemDescription"),
                "formula_property_id": formula_property_id,
                "formula_text": formula,
                "aliases": set(),
                "uses": {},
                "source_rows": [],
            },
        )
        bucket["label"] = bucket["label"] or _value(row, "itemLabel")
        bucket["description"] = bucket["description"] or _value(row, "itemDescription")
        if alias := _value(row, "alias"):
            bucket["aliases"].add(alias)
        if use_uri := _value(row, "use"):
            use_id = entity_uri_to_id(use_uri)
            bucket["uses"][use_id] = WikidataUseRelationship(
                property_id=HAS_USE_PROPERTY_ID,
                entity_id=use_id,
                entity_uri=use_uri,
                label=_value(row, "useLabel"),
                description=_value(row, "useDescription"),
            )
        bucket["source_rows"].append(_plain_binding_row(row))

    candidates = [
        WikidataEquationCandidate(
            entity_id=bucket["entity_id"],
            entity_uri=bucket["entity_uri"],
            label=bucket["label"],
            description=bucket["description"],
            formula_property_id=bucket["formula_property_id"],
            formula_text=bucket["formula_text"],
            aliases=tuple(sorted(bucket["aliases"])),
            uses=tuple(sorted(bucket["uses"].values(), key=lambda use: use.entity_id)),
            source_rows=tuple(bucket["source_rows"]),
        )
        for bucket in grouped.values()
    ]
    return sorted(candidates, key=lambda candidate: candidate.source_candidate_id)


def build_snapshot_record(
    *,
    query: str,
    response: Mapping[str, Any],
    source_version: str = "",
    source_uri: str = WIKIDATA_SPARQL_ENDPOINT,
    license_expression: str = "CC0-1.0",
    provenance_summary: str = "Wikidata SPARQL results for physical equation candidate discovery.",
) -> dict[str, Any]:
    """Build a Wave 0 ``physics_ingest_snapshots`` row for a query response."""

    payload = {
        "query": query,
        "response": response,
    }
    return {
        "source_system": "wikidata",
        "source_version": source_version,
        "source_uri": source_uri,
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "license_expression": license_expression,
        "provenance_summary": provenance_summary,
        "payload_sha256": stable_payload_hash(payload),
        "payload": payload,
    }


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for JSON-compatible payloads."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def entity_uri_to_id(uri: str) -> str:
    """Extract a Wikidata QID from an entity URI or return the input id."""

    return _uri_tail(uri, expected_prefix="Q")


def property_uri_to_id(uri: str) -> str:
    """Extract a Wikidata PID from an entity/property URI or return the input id."""

    if "/prop/direct/" in uri:
        return uri.rsplit("/", 1)[-1]
    return _uri_tail(uri, expected_prefix="P")


def _uri_tail(value: str, *, expected_prefix: str) -> str:
    value = value.strip()
    if value.startswith(("wd:", "wdt:")):
        return value.split(":", 1)[1]
    if value.startswith((WIKIDATA_ENTITY_BASE, WIKIDATA_ENTITY_HTTPS_BASE)):
        return value.rsplit("/", 1)[-1]
    if value.startswith(expected_prefix):
        return value
    return value.rsplit("/", 1)[-1] if "/" in value else value


def _value(row: Mapping[str, Mapping[str, str]], key: str) -> str:
    cell = row.get(key)
    if not cell:
        return ""
    return cell.get("value", "")


def _plain_binding_row(row: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    return {key: value.get("value", "") for key, value in row.items()}


def _sparql_values(variable: str, namespace: str, identifiers: Sequence[str]) -> str:
    normalized = []
    for identifier in identifiers:
        bare = identifier.strip()
        if ":" in bare:
            normalized.append(bare)
        else:
            normalized.append(f"{namespace}:{bare}")
    return f"VALUES ?{variable} {{ {' '.join(normalized)} }}"


def _escape_sparql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
