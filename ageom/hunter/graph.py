"""Hunter graph assembly and RetrievalAgent implementation."""

from __future__ import annotations

from pydantic_graph import Graph

from ageom.hunter.deps import HunterDeps
from ageom.hunter.llm import LLMClient
from ageom.hunter.nodes import (
    InitialSearch,
    RankCandidates,
    ReformulateQuery,
    VerifyTopK,
)
from ageom.hunter.state import HunterState
from ageom.protocols import SemanticIndex, VerificationOracle
from ageom.types import MatchResult, PDGNode

hunter_graph: Graph[HunterState, HunterDeps, MatchResult] = Graph(
    nodes=[InitialSearch, RankCandidates, VerifyTopK, ReformulateQuery]
)


class HunterAgent:
    """Retrieval agent implementing the RetrievalAgent protocol.

    Drives the Hunter graph to find verified library matches for PDG predicates.
    """

    def __init__(
        self,
        index: SemanticIndex,
        oracle: VerificationOracle,
        llm: LLMClient,
        max_iterations: int = 5,
        top_k_verify: int = 3,
        search_k: int = 20,
        *,
        mode: str = "standard",
        use_gbnf: bool = False,
        query_batch_size: int = 40,
        top_k_per_query: int = 50,
        max_candidates_total: int = 3000,
    ) -> None:
        self._deps = HunterDeps(index=index, oracle=oracle, llm=llm)
        self._max_iterations = max_iterations
        self._top_k_verify = top_k_verify
        self._search_k = search_k
        self._mode = mode
        self._use_gbnf = use_gbnf
        self._query_batch_size = query_batch_size
        self._top_k_per_query = top_k_per_query
        self._max_candidates_total = max_candidates_total

    async def find_match(self, pdg_node: PDGNode) -> MatchResult:
        """Run the Hunter graph to find a verified match for the PDG node."""
        state = HunterState(
            pdg_node=pdg_node,
            max_iterations=self._max_iterations,
            top_k_verify=self._top_k_verify,
            search_k=self._search_k,
            mode=self._mode,
            use_gbnf=self._use_gbnf,
            query_batch_size=self._query_batch_size,
            top_k_per_query=self._top_k_per_query,
            max_candidates_total=self._max_candidates_total,
        )

        result = await hunter_graph.run(
            InitialSearch(),
            state=state,
            deps=self._deps,
        )
        return result.output
