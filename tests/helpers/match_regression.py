from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ageom.architect.handoff import load_json, to_pdg_nodes
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class MatchCase:
    case_id: str
    cdg_path: Path
    witness_test_path: Path
    expected_matches: dict[str, str]
    aliases: dict[str, set[str]]
    live_runs: int
    live_thresholds: dict[str, float]


class StaticSemanticIndex:
    """Simple deterministic lexical index for regression testing Hunter behavior."""

    def __init__(self, declarations: list[Declaration]) -> None:
        self._declarations = declarations
        self._by_name = {d.name: d for d in declarations}
        self._decl_tokens = {
            d.name: self._tokens(f"{d.name} {d.type_signature} {d.docstring}")
            for d in declarations
        }

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t.lower() for t in _TOKEN_RE.findall(text)}

    def _score(self, query_text: str, decl: Declaration) -> float:
        query_tokens = self._tokens(query_text)
        decl_tokens = self._decl_tokens[decl.name]
        overlap = len(query_tokens & decl_tokens)

        leaf_name = decl.name.rsplit(".", 1)[-1].lower()
        leaf_bonus = 0.0
        if leaf_name in query_tokens:
            leaf_bonus += 3.0
        snake_parts = set(leaf_name.split("_"))
        leaf_bonus += 0.25 * len(query_tokens & snake_parts)

        # Slightly prefer shorter, specific signatures when overlap ties.
        length_penalty = 1.0 / (1.0 + len(decl_tokens))
        return float(overlap) + leaf_bonus + length_penalty

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        ranked = sorted(
            ((d, self._score(query_text, d)) for d in self._declarations),
            key=lambda item: (item[1], item[0].name),
            reverse=True,
        )
        return ranked[:k]

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        return [decl for decl, _score in self.search_by_embedding(type_signature, k=k)]

    def get_declaration(self, name: str) -> Declaration | None:
        return self._by_name.get(name)


class DeterministicHunterLLM:
    """Stable LLM stub so deterministic tests validate retrieval + graph logic only."""

    async def complete(self, system: str, user: str) -> str:
        lower = system.lower()
        if "rank" in lower or "score" in lower:
            # Over-complete index list; Hunter ignores out-of-range values.
            return json.dumps(list(range(200)))
        if "reformulate" in lower:
            return json.dumps(
                [
                    "exact function name",
                    "python callable signature",
                    "algorithmic primitive",
                ]
            )
        if "analy" in lower:
            return "Prioritize exact operator/function semantics."
        return "[]"

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


class FixtureOracle:
    """Oracle that validates against per-node alias sets from the fixture case."""

    def __init__(self, allowed_by_node_id: dict[str, set[str]]) -> None:
        self._allowed_by_node_id = allowed_by_node_id

    async def verify_candidate(
        self, pdg_node: PDGNode, candidate: CandidateMatch
    ) -> VerificationResult:
        allowed = self._allowed_by_node_id.get(pdg_node.predicate_id, set())
        verified = candidate.declaration.name in allowed
        return VerificationResult(
            candidate=candidate,
            verified=verified,
            compiler_output="ok" if verified else "type mismatch",
            proof_term=f"{candidate.declaration.name}" if verified else "",
            error_message="" if verified else "type mismatch",
            verification_level=(
                VerificationLevel.TYPE_CHECKED
                if verified
                else VerificationLevel.UNVERIFIED
            ),
        )

    async def verify_candidates(
        self, pdg_node: PDGNode, candidates: list[CandidateMatch]
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for candidate in candidates:
            result = await self.verify_candidate(pdg_node, candidate)
            results.append(result)
            if result.verified:
                break
        return results


def load_match_cases(repo_root: Path, fixture_path: Path) -> list[MatchCase]:
    data = json.loads(fixture_path.read_text())
    result: list[MatchCase] = []
    for raw in data.get("cases", []):
        aliases = {
            node_id: set(names) for node_id, names in raw.get("aliases", {}).items()
        }
        # Ensure canonical declaration is always accepted.
        for node_id, canonical in raw.get("expected_matches", {}).items():
            aliases.setdefault(node_id, set()).add(canonical)

        result.append(
            MatchCase(
                case_id=raw["id"],
                cdg_path=(repo_root / raw["cdg_path"]).resolve(),
                witness_test_path=(repo_root / raw["witness_test_path"]).resolve(),
                expected_matches=dict(raw.get("expected_matches", {})),
                aliases=aliases,
                live_runs=int(raw.get("live_runs", 3)),
                live_thresholds=dict(raw.get("live_thresholds", {})),
            )
        )
    return result


def load_case_pdg_nodes(case: MatchCase) -> list[PDGNode]:
    cdg = load_json(case.cdg_path)
    return to_pdg_nodes(cdg, prover=Prover.PYTHON, strict=True)


def build_ageo_atoms_declarations(ageo_atoms_root: Path) -> list[Declaration]:
    declarations: list[Declaration] = []
    ageoa_root = ageo_atoms_root / "ageoa"
    for py_path in sorted(ageoa_root.rglob("*.py")):
        if py_path.name == "__init__.py":
            continue
        if "witness" in py_path.name:
            continue

        rel = py_path.relative_to(ageo_atoms_root).with_suffix("")
        module_name = ".".join(rel.parts)

        try:
            tree = ast.parse(py_path.read_text(), filename=str(py_path))
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue

            params: list[str] = []
            for arg in node.args.args:
                if arg.annotation is not None:
                    params.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                else:
                    params.append(arg.arg)
            ret = ast.unparse(node.returns) if node.returns is not None else "Any"
            type_sig = f"({', '.join(params)}) -> {ret}"
            doc = ast.get_docstring(node) or ""

            declarations.append(
                Declaration(
                    name=f"{module_name}.{node.name}",
                    type_signature=type_sig,
                    docstring=doc,
                    source_lib=module_name,
                    prover=Prover.PYTHON,
                )
            )

    return declarations


def match_results_to_name_map(results: dict[str, MatchResult]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node_id, result in results.items():
        if result.verified_match is None:
            out[node_id] = ""
        else:
            out[node_id] = result.verified_match.candidate.declaration.name
    return out


def alias_hit_rate(
    name_map: dict[str, str],
    aliases: dict[str, set[str]],
) -> float:
    if not aliases:
        return 0.0
    total = len(aliases)
    hits = 0
    for node_id, allowed in aliases.items():
        if name_map.get(node_id, "") in allowed:
            hits += 1
    return hits / total


def run_stability_score(run_maps: list[dict[str, str]], node_ids: list[str]) -> float:
    if len(run_maps) < 2:
        return 1.0

    pair_scores: list[float] = []
    for i in range(len(run_maps)):
        for j in range(i + 1, len(run_maps)):
            agree = 0
            for node_id in node_ids:
                if run_maps[i].get(node_id, "") == run_maps[j].get(node_id, ""):
                    agree += 1
            pair_scores.append(agree / len(node_ids) if node_ids else 1.0)

    return sum(pair_scores) / len(pair_scores)
