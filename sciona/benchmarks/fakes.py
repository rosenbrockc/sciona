"""Deterministic benchmark doubles used by the flow harness."""

from __future__ import annotations

import json
import random

from sciona.architect.handoff import CDGExport
from sciona.architect.models import ConceptType
from sciona.benchmarks.core import FlowBenchmarkCase, _hint_matches, _slug
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


class _FlowArchitectLLM:
    def __init__(self, case: FlowBenchmarkCase) -> None:
        self._case = case
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        system_lower = system.lower()
        if "best" in system_lower and "paradigm" in system_lower:
            return json.dumps(
                {
                    "paradigm": ConceptType.CUSTOM.value,
                    "rationale": (
                        f"{self._case.case_id} is benchmarked as a direct leaf decomposition"
                    ),
                    "variant_hint": "",
                }
            )
        if "sub-nodes" in system_lower or "sub_nodes" in system_lower:
            return json.dumps(
                {
                    "sub_nodes": [
                        {
                            "name": leaf.name,
                            "description": leaf.description,
                            "concept_type": self._case.concept_type.value,
                            "inputs": [
                                {"name": name, "type_desc": type_desc}
                                for name, type_desc in leaf.inputs
                            ],
                            "outputs": [
                                {"name": name, "type_desc": type_desc}
                                for name, type_desc in leaf.outputs
                            ],
                            "type_signature": leaf.type_signature,
                            "is_atomic": True,
                            "matched_primitive": _slug(leaf.name),
                        }
                        for leaf in self._case.leaves
                    ],
                    "edges": [],
                }
            )
        if "critic" in system_lower or "evaluate" in system_lower:
            return json.dumps(
                {
                    "approved": True,
                    "reason": "Valid decomposition",
                    "io_issues": [],
                    "flagged_nodes": [],
                }
            )
        return "{}"


class _NoisyFlowArchitectLLM(_FlowArchitectLLM):
    """Architect mock that introduces controlled perturbations for stability testing."""

    def __init__(
        self,
        case: FlowBenchmarkCase,
        *,
        seed: int | None = None,
        shuffle_prob: float = 0.3,
        drop_field_prob: float = 0.1,
        alter_desc_prob: float = 0.2,
    ) -> None:
        super().__init__(case)
        self._rng = random.Random(seed)
        self._shuffle_prob = shuffle_prob
        self._drop_field_prob = drop_field_prob
        self._alter_desc_prob = alter_desc_prob

    async def complete(self, system: str, user: str) -> str:
        raw = await super().complete(system, user)
        system_lower = system.lower()
        if not ("sub-nodes" in system_lower or "sub_nodes" in system_lower):
            return raw
        parsed = json.loads(raw)
        nodes = parsed.get("sub_nodes", [])
        if self._rng.random() < self._shuffle_prob:
            self._rng.shuffle(nodes)
        for node in nodes:
            if self._rng.random() < self._alter_desc_prob:
                node["description"] = node["description"] + " (variant)"
            if self._rng.random() < self._drop_field_prob:
                node.pop("matched_primitive", None)
        parsed["sub_nodes"] = nodes
        return json.dumps(parsed)


class _BenchmarkHunterLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        lower = system.lower()
        if "rank" in lower or "score" in lower:
            return "[0, 1, 2, 3]"
        if "reformulate" in lower:
            return '["exact function name", "typed primitive query"]'
        if "analy" in lower:
            return "CAUSE: broad query\nTARGET: exact primitive\nNEXT: search by primitive name"
        return "[]"

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


class _FailFirstHunterLLM(_BenchmarkHunterLLM):
    """Hunter mock that reverses ranking on the first score call, then succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self._first_score_done = False

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        lower = system.lower()
        if ("rank" in lower or "score" in lower) and not self._first_score_done:
            self._first_score_done = True
            return "[3, 2, 1, 0]"
        return await super().complete(system, user)


class _LexicalSemanticIndex:
    def __init__(self, declarations: list[Declaration]) -> None:
        self._declarations = declarations

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in _slug(text).split("_") if token}

    def _score(self, query_text: str, decl: Declaration) -> float:
        query_tokens = self._tokens(query_text)
        decl_tokens = self._tokens(
            f"{decl.name} {decl.type_signature} {decl.docstring} {decl.conceptual_summary}"
        )
        return float(len(query_tokens & decl_tokens))

    def search_by_embedding(self, query_text: str, k: int = 10):
        ranked = sorted(
            ((decl, self._score(query_text, decl)) for decl in self._declarations),
            key=lambda item: (item[1], item[0].name),
            reverse=True,
        )
        return ranked[:k]

    def search_by_type(self, type_signature: str, k: int = 10):
        return [decl for decl, _score in self.search_by_embedding(type_signature, k=k)]


class _LeafOracle:
    def __init__(self, expected_by_query_hint: dict[str, str]) -> None:
        self._expected_by_query_hint = expected_by_query_hint

    async def verify_candidate(
        self, pdg_node: PDGNode, candidate: CandidateMatch
    ) -> VerificationResult:
        expected = ""
        node_text = f"{pdg_node.statement} {pdg_node.informal_desc}".lower()
        for hint, name in self._expected_by_query_hint.items():
            if _hint_matches(node_text, hint):
                expected = name
                break
        verified = candidate.declaration.name == expected and expected != ""
        return VerificationResult(
            candidate=candidate,
            verified=verified,
            compiler_output="ok" if verified else "type mismatch",
            proof_term=candidate.declaration.name if verified else "",
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


class _EmptySkillIndex:
    def search(self, query: str, k: int = 10):
        return []


class _LLMFromScratchMock:
    """Simulates a raw LLM identifying library functions from a goal prompt."""

    def __init__(self, case: FlowBenchmarkCase) -> None:
        self._case = case
        self.calls = 0

    async def identify(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps(
            [
                {
                    "name": leaf.declaration_name,
                    "type_signature": leaf.type_signature,
                    "description": leaf.description,
                }
                for leaf in self._case.leaves
            ]
        )


class _NoisyLLMFromScratchMock(_LLMFromScratchMock):
    """LLM-from-scratch mock with realistic error modes."""

    def __init__(
        self,
        case: FlowBenchmarkCase,
        *,
        seed: int | None = None,
        miss_leaf_prob: float = 0.2,
        hallucinate_prob: float = 0.15,
        rename_prob: float = 0.1,
    ) -> None:
        super().__init__(case)
        self._rng = random.Random(seed)
        self._miss_leaf_prob = miss_leaf_prob
        self._hallucinate_prob = hallucinate_prob
        self._rename_prob = rename_prob

    async def identify(self, prompt: str) -> str:
        self.calls += 1
        results = []
        for leaf in self._case.leaves:
            if self._rng.random() < self._miss_leaf_prob:
                continue
            name = leaf.declaration_name
            if self._rng.random() < self._rename_prob:
                name = name.replace(".", ".wrong_")
            results.append(
                {
                    "name": name,
                    "type_signature": leaf.type_signature,
                    "description": leaf.description,
                }
            )
        if self._rng.random() < self._hallucinate_prob:
            results.append(
                {
                    "name": "algorithms.hallucinated_function",
                    "type_signature": "any -> any",
                    "description": "This function does not exist.",
                }
            )
        return json.dumps(results)
