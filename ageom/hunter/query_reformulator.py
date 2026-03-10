"""Deterministic lightweight replacement for the hunter reformulate prompt."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from ageom.types import Declaration

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_EXACT_COUNT_RE = re.compile(r"generate exactly (\d+)", re.IGNORECASE)
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "candidate",
    "code",
    "correct",
    "description",
    "does",
    "exactly",
    "for",
    "from",
    "function",
    "generate",
    "got",
    "highly",
    "id",
    "in",
    "into",
    "is",
    "it",
    "last",
    "library",
    "likely",
    "map",
    "maximize",
    "new",
    "not",
    "of",
    "on",
    "or",
    "over",
    "predicate",
    "previous",
    "prover",
    "queries",
    "query",
    "return",
    "returns",
    "search",
    "signal",
    "strings",
    "that",
    "the",
    "this",
    "to",
    "tried",
    "try",
    "use",
    "using",
    "with",
}
_DOMAIN_ANCHORS = {
    "add",
    "addition",
    "bandpass",
    "cholesky",
    "commutative",
    "distance",
    "distances",
    "dijkstra",
    "dynamic",
    "ecg",
    "filter",
    "graph",
    "lcs",
    "longest",
    "nat",
    "path",
    "positive",
    "shortest",
    "solve",
    "spd",
    "subsequence",
    "symmetric",
    "theorem",
    "triangular",
    "weighted",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace(".", " ").replace("-", " "))


def _normalize_query(text: str) -> str:
    return " ".join(text.split())


def _parse_prompt(user: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "predicate_id": "",
        "statement": "",
        "informal_desc": "",
        "prover": "",
        "queries_tried": [],
        "compiler_errors": "",
        "catalog_hints": [],
        "requested_count": 4,
    }
    current_section = ""
    compiler_lines: list[str] = []

    for raw_line in user.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        section_match = _SECTION_RE.match(stripped)
        if section_match:
            current_section = section_match.group(1).lower()
            continue
        if not stripped:
            continue
        if stripped.startswith("ID:"):
            fields["predicate_id"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Statement:"):
            fields["statement"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Description:"):
            fields["informal_desc"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Prover:"):
            fields["prover"] = stripped.split(":", 1)[1].strip()
            continue
        if current_section.startswith("previous queries") and stripped.startswith("- "):
            fields["queries_tried"].append(stripped[2:].strip())
            continue
        if current_section.startswith("catalog hints") and stripped.startswith("- "):
            fields["catalog_hints"].append(stripped[2:].strip())
            continue
        if current_section.startswith("compiler errors"):
            compiler_lines.append(stripped)

    count_match = _EXACT_COUNT_RE.search(user)
    if count_match:
        fields["requested_count"] = max(1, min(8, int(count_match.group(1))))
    fields["compiler_errors"] = "\n".join(compiler_lines)
    return fields


def _phrase_rules(text: str) -> list[str]:
    lower = text.lower()
    if all(term in lower for term in ("ecg", "bandpass", "filter")):
        return [
            "ecg bandpass filter",
            "stable ecg filter",
            "bandpass cardiac signal filter",
            "iir bandpass filter",
            "filtered signal bandpass filter",
        ]
    if ("shortest path" in lower or "distance" in lower) and (
        "graph" in lower or "dijkstra" in lower or "weighted" in lower
    ):
        return [
            "dijkstra shortest path",
            "weighted graph distances",
            "shortest path distance map",
            "single source shortest path",
            "relax edges distances",
        ]
    if "spd" in lower or "symmetric positive definite" in lower:
        return [
            "cholesky solve spd",
            "solve symmetric positive definite",
            "triangular solve cholesky",
            "positive definite linear solve",
            "cholesky factor solve",
        ]
    if "longest common subsequence" in lower or (
        "subsequence" in lower and "dynamic" in lower
    ) or " lcs " in f" {lower} ":
        return [
            "longest common subsequence",
            "dynamic programming lcs",
            "string subsequence recurrence",
            "lcs dp table",
            "common subsequence dynamic programming",
        ]
    if (
        ("commutat" in lower and "add" in lower)
        or ("n + m = m + n" in lower)
        or ("ℕ" in lower and "+" in lower)
    ):
        return [
            "Nat.add_comm addition commutative",
            "addition commutative natural numbers",
            "n + m = m + n theorem",
            "nat add comm theorem",
        ]
    return []


def _keyword_variants(
    statement: str,
    informal_desc: str,
    compiler_errors: str,
    prover: str,
) -> list[str]:
    priorities: list[str] = []
    for source in (statement, informal_desc, compiler_errors):
        for token in _tokenize(source):
            if token in _STOPWORDS:
                continue
            if len(token) <= 2 and token not in {"dp", "ecg"}:
                continue
            if token not in priorities:
                priorities.append(token)

    if not priorities:
        return []

    queries: list[str] = []
    if len(priorities) >= 3:
        queries.append(" ".join(priorities[:4]))
    if len(priorities) >= 5:
        queries.append(" ".join(priorities[1:5]))
    if len(priorities) >= 4:
        queries.append(" ".join(priorities[:2] + priorities[-2:]))
    if prover.lower() == "lean4":
        lean_terms = [tok for tok in priorities if tok in {"nat", "list", "theorem", "commutative", "comm"}]
        if lean_terms:
            queries.append(" ".join(["lean4"] + lean_terms[:3]))
    return [_normalize_query(query) for query in queries if query.strip()]


def _extract_declarations(index: Any) -> list[Declaration]:
    seen: set[str] = set()
    declarations: list[Declaration] = []

    def _ingest(items: list[Declaration]) -> None:
        for decl in items:
            if not isinstance(decl, Declaration) or not decl.name or decl.name in seen:
                continue
            seen.add(decl.name)
            declarations.append(decl)

    by_name = getattr(index, "_by_name", None)
    if isinstance(by_name, dict):
        _ingest(list(by_name.values()))

    raw_declarations = getattr(index, "_declarations", None)
    if isinstance(raw_declarations, list):
        _ingest(raw_declarations)
    elif isinstance(raw_declarations, dict):
        _ingest(list(raw_declarations.values()))

    store = getattr(index, "_store", None)
    store_declarations = getattr(store, "_declarations", None)
    if isinstance(store_declarations, dict):
        _ingest(list(store_declarations.values()))

    return declarations


def _decl_text_tokens(decl: Declaration) -> set[str]:
    return set(
        _tokenize(
            " ".join(
                [
                    decl.name,
                    decl.type_signature,
                    decl.docstring,
                    decl.conceptual_summary,
                    decl.source_lib,
                ]
            )
        )
    )


def _namespace_prefixes(name: str) -> list[str]:
    parts = [segment for segment in re.split(r"[:./]+", name) if segment]
    if len(parts) <= 1:
        return []
    prefixes: list[str] = []
    for idx in range(1, len(parts)):
        prefixes.append(".".join(parts[:idx]))
    return prefixes


def derive_catalog_hints(
    index: Any,
    *,
    statement: str,
    informal_desc: str,
    compiler_errors: str,
    queries_tried: list[str],
    limit: int = 6,
) -> list[str]:
    declarations = _extract_declarations(index)
    if not declarations:
        return []

    query_tokens = set(
        _tokenize(" ".join([statement, informal_desc, compiler_errors, *queries_tried]))
    )
    if not query_tokens:
        return []

    ranked: list[tuple[float, Declaration]] = []
    for decl in declarations:
        overlap = len(query_tokens & _decl_text_tokens(decl))
        score = float(overlap)
        lowered_name = decl.name.lower()
        if any(query.lower() in lowered_name for query in queries_tried if query.strip()):
            score += 2.0
        if "." in decl.name:
            score += 0.15
        if score > 0:
            ranked.append((score, decl))

    if not ranked:
        return []

    ranked.sort(key=lambda row: (-row[0], row[1].name))
    top = [decl for _score, decl in ranked[: max(limit, 4)]]

    hints: list[str] = []
    namespace_counts: Counter[str] = Counter()
    for decl in top:
        for prefix in _namespace_prefixes(decl.name):
            namespace_counts[prefix] += 1

    for namespace, _count in namespace_counts.most_common(limit):
        hints.append(f"namespace:{namespace}")
        if len(hints) >= limit:
            return hints

    for decl in top:
        hints.append(f"declaration:{decl.name}")
        if len(hints) >= limit:
            break
    return hints


def _catalog_hint_variants(
    statement: str,
    informal_desc: str,
    compiler_errors: str,
    catalog_hints: list[str],
) -> list[str]:
    if not catalog_hints:
        return []

    context_terms = [
        token
        for token in _tokenize(" ".join([statement, informal_desc, compiler_errors]))
        if token not in _STOPWORDS
    ][:4]
    queries: list[str] = []
    seen: set[str] = set()

    for hint in catalog_hints:
        raw = str(hint).strip()
        if not raw:
            continue
        kind, _, value = raw.partition(":")
        kind = kind.strip().lower()
        value = value.strip()
        if not value:
            continue

        if kind == "namespace":
            candidate_queries = [
                f"{value} {' '.join(context_terms[:2])}".strip(),
                f"{value} {' '.join(context_terms[2:4])}".strip(),
            ]
        elif kind == "declaration":
            tail = re.split(r"[:./]+", value)[-1]
            candidate_queries = [
                f"{value} {' '.join(context_terms[:2])}".strip(),
                f"{tail} {' '.join(context_terms[:3])}".strip(),
            ]
        else:
            candidate_queries = [f"{value} {' '.join(context_terms[:2])}".strip()]

        for candidate in candidate_queries:
            normalized = _normalize_query(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            queries.append(normalized)

    return queries


class HeuristicQueryReformulator:
    """Query reformulator with deterministic rules and LLM fallback on ambiguity."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "query_reformulator_v1"

    def __init__(self, fallback: Any, *, min_queries: int = 3) -> None:
        self._fallback = fallback
        self._min_queries = min_queries
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        generated = self._generate(user)
        if generated is None:
            self._last_completion_metadata = {"reformulation_source": "fallback"}
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        self._last_completion_metadata = {
            "reformulation_source": "deterministic",
            "reformulation_query_count": len(generated),
        }
        self._last_error_metadata = {}
        return json.dumps(generated)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)

    def _generate(self, user: str) -> list[str] | None:
        parsed = _parse_prompt(user)
        statement = str(parsed["statement"]).strip()
        informal_desc = str(parsed["informal_desc"]).strip()
        compiler_errors = str(parsed["compiler_errors"]).strip()
        prover = str(parsed["prover"]).strip()
        previous = {str(item).strip() for item in parsed["queries_tried"] if str(item).strip()}
        catalog_hints = [str(item).strip() for item in parsed["catalog_hints"] if str(item).strip()]
        requested_count = max(self._min_queries, int(parsed["requested_count"]))

        if not statement and not informal_desc:
            return None

        combined = " ".join(part for part in (statement, informal_desc, compiler_errors) if part)
        anchor_tokens = set(_tokenize(combined))
        candidates = _phrase_rules(combined)
        candidates.extend(
            _catalog_hint_variants(
                statement,
                informal_desc,
                compiler_errors,
                catalog_hints,
            )
        )
        if not candidates and not (anchor_tokens & _DOMAIN_ANCHORS) and not catalog_hints:
            return None
        candidates.extend(_keyword_variants(statement, informal_desc, compiler_errors, prover))

        deduped: list[str] = []
        seen = set(previous)
        for query in candidates:
            normalized = _normalize_query(query)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
            if len(deduped) >= requested_count:
                break

        if len(deduped) < min(requested_count, self._min_queries):
            return None

        return deduped[:requested_count]
