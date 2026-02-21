"""Top-level orchestration loop: Architect -> Hunter -> (on failure) -> Architect.refine -> Hunter.

Replaces the one-shot CLI flow with a feedback loop that re-decomposes
atomic nodes when the Hunter fails to ground them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ageom.architect.handoff import CDGExport, to_pdg_nodes
from ageom.architect.models import AlgorithmicNode, NodeStatus
from ageom.llm_router import ORCHESTRATOR_REFINE, select_llm
from ageom.types import (
    FailureAction,
    MatchFailureReport,
    MatchResult,
    Prover,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result of the full orchestration loop."""

    cdg: CDGExport
    match_results: list[MatchResult]
    rounds_used: int = 0
    failures: list[MatchFailureReport] = field(default_factory=list)
    ungroundable: list[str] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return all(mr.success for mr in self.match_results)


async def refine_on_failure(
    failure: MatchFailureReport,
    cdg: CDGExport,
    llm: Any,
) -> CDGExport:
    """Refine the CDG based on a match failure.

    Uses the LLM to analyze the failure and either:
    - Split the atomic node into 2-3 finer sub-atoms
    - Replace it with an equivalent formulation
    - Mark it as UNGROUNDABLE

    Args:
        failure: The match failure report.
        cdg: The current CDG.
        llm: An LLM client for analysis.

    Returns:
        Updated CDG with the failed node refined.
    """
    node = failure.pdg_node
    action = failure.suggested_action

    if action == FailureAction.UNGROUNDABLE:
        # Mark the node and return unchanged CDG
        for n in cdg.nodes:
            if n.node_id == node.predicate_id:
                n.status = NodeStatus.REJECTED
                n.critic_notes = "Marked UNGROUNDABLE after match failure"
                break
        return cdg

    if action == FailureAction.SPLIT:
        error_context = "\n".join(failure.error_summaries[:3])
        system_prompt = (
            "You are an algorithm decomposition expert. A predicate could not be "
            "grounded to a single library function. Suggest 2-3 finer-grained "
            "sub-predicates that together implement the same functionality. "
            "Reply with a JSON array of objects: "
            '[{"name": "...", "description": "...", "type_signature": "..."}]'
        )
        user_prompt = (
            f"Predicate: {node.statement}\n"
            f"Description: {node.informal_desc}\n"
            f"Match errors:\n{error_context}\n\n"
            "Split this into 2-3 finer sub-predicates."
        )

        try:
            response = await select_llm(llm, ORCHESTRATOR_REFINE).complete(
                system_prompt, user_prompt
            )

            # Parse JSON from response
            text = response.strip()
            if "```" in text:
                lines = text.splitlines()
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```") and not in_block:
                        in_block = True
                        continue
                    if line.strip() == "```" and in_block:
                        break
                    if in_block:
                        json_lines.append(line)
                text = "\n".join(json_lines)

            sub_nodes = json.loads(text)
            if isinstance(sub_nodes, list) and sub_nodes:
                # Find the original node in the CDG
                original = None
                for n in cdg.nodes:
                    if n.node_id == node.predicate_id:
                        original = n
                        break

                if original is not None:
                    # Mark original as DECOMPOSED
                    original.status = NodeStatus.DECOMPOSED
                    original.children = []

                    # Create sub-nodes
                    for i, sub in enumerate(sub_nodes[:3]):
                        sub_id = f"{original.node_id}_sub{i}"
                        sub_node = AlgorithmicNode(
                            node_id=sub_id,
                            parent_id=original.node_id,
                            name=sub.get("name", f"sub_{i}"),
                            description=sub.get("description", ""),
                            concept_type=original.concept_type,
                            status=NodeStatus.ATOMIC,
                            depth=original.depth + 1,
                            type_signature=sub.get("type_signature", ""),
                        )
                        cdg.nodes.append(sub_node)
                        original.children.append(sub_id)

                logger.info(
                    "Split node %s into %d sub-nodes",
                    node.predicate_id,
                    len(sub_nodes[:3]),
                )
        except Exception as exc:
            logger.warning("Failed to split node %s: %s", node.predicate_id, exc)
            # Mark as ungroundable if splitting fails
            for n in cdg.nodes:
                if n.node_id == node.predicate_id:
                    n.status = NodeStatus.REJECTED
                    n.critic_notes = f"Split failed: {exc}"
                    break

    elif action == FailureAction.GENERALIZE:
        # Broaden the type signature
        for n in cdg.nodes:
            if n.node_id == node.predicate_id:
                n.critic_notes = "Re-formulated after match failure"
                break

    return cdg


async def run_orchestration(
    cdg: CDGExport,
    *,
    hunter_agent: Any,
    llm: Any,
    prover: Prover = Prover.LEAN4,
    max_rounds: int = 3,
) -> OrchestratorResult:
    """Run the full Architect -> Hunter feedback loop.

    Args:
        cdg: Initial CDG from the Architect.
        hunter_agent: A RetrievalAgent (has find_match method).
        llm: An LLM client for refinement.
        prover: Target prover.
        max_rounds: Maximum refinement rounds.

    Returns:
        OrchestratorResult with final CDG and match results.
    """
    result = OrchestratorResult(cdg=cdg, match_results=[])

    for round_num in range(max_rounds):
        result.rounds_used = round_num + 1
        logger.info("Orchestration round %d/%d", round_num + 1, max_rounds)

        # Convert current CDG to PDG nodes
        try:
            pdg_nodes = to_pdg_nodes(cdg, prover=prover, strict=False)
        except ValueError as exc:
            logger.warning("CDG conversion failed: %s", exc)
            break

        if not pdg_nodes:
            logger.info("No atomic nodes to match")
            break

        # Filter out already-matched nodes
        matched_ids = {
            mr.pdg_node.predicate_id for mr in result.match_results if mr.success
        }
        pending_nodes = [n for n in pdg_nodes if n.predicate_id not in matched_ids]

        if not pending_nodes:
            logger.info("All nodes matched")
            break

        # Run Hunter on pending nodes
        round_failures: list[MatchFailureReport] = []
        for pdg_node in pending_nodes:
            match_result = await hunter_agent.find_match(pdg_node)

            # Replace existing result for this node if any
            result.match_results = [
                mr
                for mr in result.match_results
                if mr.pdg_node.predicate_id != pdg_node.predicate_id
            ]
            result.match_results.append(match_result)

            if not match_result.success:
                failure = MatchFailureReport.from_match_result(match_result)
                round_failures.append(failure)
                result.failures.append(failure)

        if not round_failures:
            logger.info("All nodes matched in round %d", round_num + 1)
            break

        # If this is the last round, don't refine
        if round_num >= max_rounds - 1:
            for f in round_failures:
                result.ungroundable.append(f.pdg_node.predicate_id)
            break

        # Refine CDG based on failures
        for failure in round_failures:
            cdg = await refine_on_failure(failure, cdg, llm)

        result.cdg = cdg

    return result
