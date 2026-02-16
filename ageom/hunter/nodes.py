"""Pydantic-graph nodes forming the Hunter's search-verify-refine cycle.

Graph topology:
    InitialSearch -> RankCandidates -> VerifyTopK -> ReformulateQuery -> InitialSearch -> ...
                                           |
                                      End[MatchResult]  (on verified match or budget exhausted)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from ageom.hunter.deps import HunterDeps
from ageom.hunter.prompts import (
    ANALYZE_FAILURE_SYSTEM,
    ANALYZE_FAILURE_USER,
    REFORMULATE_QUERY_SYSTEM,
    REFORMULATE_QUERY_USER,
    SCORE_CANDIDATES_SYSTEM,
    SCORE_CANDIDATES_USER,
)
from ageom.hunter.state import HunterState
from ageom.types import CandidateMatch, MatchResult


@dataclass
class InitialSearch(BaseNode[HunterState, HunterDeps, MatchResult]):
    """Search the semantic index using embedding and type queries."""

    async def run(
        self, ctx: GraphRunContext[HunterState, HunterDeps]
    ) -> RankCandidates | End[MatchResult]:
        state = ctx.state
        deps = ctx.deps
        node = state.pdg_node

        # Build query from predicate
        if state.queries_tried:
            query = state.queries_tried[-1]
        else:
            query = f"{node.statement} {node.informal_desc}".strip()
            state.queries_tried.append(query)

        # Embedding search
        embedding_results = deps.index.search_by_embedding(query, k=state.search_k)

        # Type search
        type_results = deps.index.search_by_type(node.statement, k=state.search_k)

        # Merge and deduplicate
        seen_names: set[str] = {c.declaration.name for c in state.candidates_found}
        new_candidates: list[CandidateMatch] = []

        for decl, score in embedding_results:
            if decl.name not in seen_names:
                seen_names.add(decl.name)
                new_candidates.append(
                    CandidateMatch(declaration=decl, score=score, retrieval_method="embedding")
                )

        for decl in type_results:
            if decl.name not in seen_names:
                seen_names.add(decl.name)
                new_candidates.append(
                    CandidateMatch(declaration=decl, score=0.0, retrieval_method="type_search")
                )

        state.candidates_found.extend(new_candidates)

        if not state.candidates_found:
            # No candidates at all - end immediately
            return End(
                MatchResult(
                    pdg_node=node,
                    all_candidates=state.candidates_found,
                    all_verifications=state.verification_results,
                )
            )

        return RankCandidates()


@dataclass
class RankCandidates(BaseNode[HunterState, HunterDeps, MatchResult]):
    """Use LLM to rank candidates by likelihood of being correct."""

    async def run(
        self, ctx: GraphRunContext[HunterState, HunterDeps]
    ) -> VerifyTopK:
        state = ctx.state
        deps = ctx.deps

        # Build candidate list for the LLM
        candidates_text = "\n".join(
            f"[{i}] {c.declaration.name} : {c.declaration.type_signature}"
            for i, c in enumerate(state.candidates_found)
        )

        user_msg = SCORE_CANDIDATES_USER.format(
            statement=state.pdg_node.statement,
            informal_desc=state.pdg_node.informal_desc,
            candidates_list=candidates_text,
        )

        try:
            response = await deps.llm.complete(SCORE_CANDIDATES_SYSTEM, user_msg)
            ranked_indices = json.loads(response)
            # Reorder candidates
            reordered = []
            seen = set()
            for idx in ranked_indices:
                if isinstance(idx, int) and 0 <= idx < len(state.candidates_found):
                    if idx not in seen:
                        seen.add(idx)
                        reordered.append(state.candidates_found[idx])
            # Append any candidates the LLM didn't rank
            for i, c in enumerate(state.candidates_found):
                if i not in seen:
                    reordered.append(c)
            state.candidates_found = reordered
        except (json.JSONDecodeError, Exception):
            # If LLM ranking fails, keep original order
            pass

        return VerifyTopK()


@dataclass
class VerifyTopK(BaseNode[HunterState, HunterDeps, MatchResult]):
    """Send top-K candidates to the Verification Oracle."""

    async def run(
        self, ctx: GraphRunContext[HunterState, HunterDeps]
    ) -> ReformulateQuery | End[MatchResult]:
        state = ctx.state
        deps = ctx.deps

        # Take top-K unverified candidates
        already_verified = {vr.candidate.declaration.name for vr in state.verification_results}
        to_verify = [
            c for c in state.candidates_found if c.declaration.name not in already_verified
        ][: state.top_k_verify]

        if not to_verify:
            # Nothing new to verify
            if state.iteration >= state.max_iterations:
                return End(
                    MatchResult(
                        pdg_node=state.pdg_node,
                        all_candidates=state.candidates_found,
                        all_verifications=state.verification_results,
                    )
                )
            return ReformulateQuery()

        results = await deps.oracle.verify_candidates(state.pdg_node, to_verify)
        state.verification_results.extend(results)

        # Collect compiler feedback for potential reformulation
        for r in results:
            if not r.verified and r.compiler_output:
                state.compiler_feedback.append(r.compiler_output)

        # Check for verified match
        for r in results:
            if r.verified:
                state.verified_match = r
                return End(
                    MatchResult(
                        pdg_node=state.pdg_node,
                        verified_match=r,
                        all_candidates=state.candidates_found,
                        all_verifications=state.verification_results,
                    )
                )

        # No verified match - check budget
        if state.iteration >= state.max_iterations:
            return End(
                MatchResult(
                    pdg_node=state.pdg_node,
                    all_candidates=state.candidates_found,
                    all_verifications=state.verification_results,
                )
            )

        return ReformulateQuery()


@dataclass
class ReformulateQuery(BaseNode[HunterState, HunterDeps, MatchResult]):
    """Use LLM to analyze failures and generate new search queries."""

    async def run(
        self, ctx: GraphRunContext[HunterState, HunterDeps]
    ) -> InitialSearch:
        state = ctx.state
        deps = ctx.deps
        state.iteration += 1

        # Analyze the most recent failure if available
        recent_failures = [
            r for r in state.verification_results if not r.verified
        ]
        analysis = ""
        if recent_failures:
            last_fail = recent_failures[-1]
            analyze_msg = ANALYZE_FAILURE_USER.format(
                statement=state.pdg_node.statement,
                candidate_name=last_fail.candidate.declaration.name,
                candidate_type=last_fail.candidate.declaration.type_signature,
                compiler_output=last_fail.compiler_output,
            )
            try:
                analysis = await deps.llm.complete(ANALYZE_FAILURE_SYSTEM, analyze_msg)
            except Exception:
                analysis = ""

        # Generate new queries
        queries_text = "\n".join(f"- {q}" for q in state.queries_tried)
        errors_text = "\n".join(state.compiler_feedback[-3:])  # Last 3 errors
        if analysis:
            errors_text += f"\n\nAnalysis: {analysis}"

        reformulate_msg = REFORMULATE_QUERY_USER.format(
            predicate_id=state.pdg_node.predicate_id,
            statement=state.pdg_node.statement,
            informal_desc=state.pdg_node.informal_desc,
            prover=state.pdg_node.prover.value,
            queries_tried=queries_text,
            compiler_errors=errors_text,
        )

        try:
            response = await deps.llm.complete(REFORMULATE_QUERY_SYSTEM, reformulate_msg)
            new_queries = json.loads(response)
            if isinstance(new_queries, list) and new_queries:
                state.queries_tried.extend(str(q) for q in new_queries)
        except (json.JSONDecodeError, Exception):
            # Fallback: try a simple variant
            state.queries_tried.append(state.pdg_node.statement)

        return InitialSearch()
