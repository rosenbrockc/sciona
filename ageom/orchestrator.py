"""Top-level orchestration loop: Architect -> Hunter -> (on failure) -> Architect.refine -> Hunter.

Replaces the one-shot CLI flow with a feedback loop that re-decomposes
atomic nodes when the Hunter fails to ground them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ageom.architect.handoff import CDGExport, to_pdg_nodes
from ageom.architect.models import AlgorithmicNode, NodeStatus
from ageom.llm_router import ORCHESTRATOR_REFINE, select_llm
from ageom.telemetry import log_event, telemetry_scope, telemetry_stage, update_stage
from ageom.types import (
    FailureAction,
    MatchFailureReport,
    MatchResult,
    Prover,
)

logger = logging.getLogger(__name__)

_REFINE_STOPWORDS = {
    "a",
    "an",
    "and",
    "apply",
    "by",
    "for",
    "from",
    "function",
    "helper",
    "implement",
    "into",
    "of",
    "on",
    "or",
    "return",
    "same",
    "step",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
}
_CONNECTOR_PATTERN = re.compile(r"\b(?:and then|then|with|plus|before|after)\b", re.IGNORECASE)


def _find_cdg_node(cdg: CDGExport, node_id: str) -> AlgorithmicNode | None:
    for node in cdg.nodes:
        if node.node_id == node_id:
            return node
    return None


def _apply_split_subnodes(
    cdg: CDGExport,
    original: AlgorithmicNode,
    sub_nodes: list[dict[str, str]],
) -> CDGExport:
    original.status = NodeStatus.DECOMPOSED
    original.children = []

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
    return cdg


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [tok for tok in tokens if tok not in _REFINE_STOPWORDS and len(tok) >= 3]


def _title_from_clause(clause: str) -> str:
    tokens = _content_tokens(clause)[:4]
    if not tokens:
        return clause.strip().title() or "Refined Step"
    return " ".join(token.capitalize() for token in tokens)


def _split_on_connectors(text: str) -> list[dict[str, str]] | None:
    parts = [part.strip(" ,.;") for part in _CONNECTOR_PATTERN.split(text) if part.strip(" ,.;")]
    if len(parts) < 2 or len(parts) > 3:
        return None

    refined: list[dict[str, str]] = []
    for part in parts:
        tokens = _content_tokens(part)
        if len(tokens) < 2:
            return None
        refined.append(
            {
                "name": _title_from_clause(part),
                "description": part.strip(),
                "type_signature": "",
            }
        )
    return refined


def _deterministic_split_subnodes(
    failure: MatchFailureReport,
    original: AlgorithmicNode | None,
) -> list[dict[str, str]] | None:
    description = (original.description if original is not None else "") or failure.pdg_node.informal_desc
    statement = failure.pdg_node.statement
    error_text = " ".join(failure.error_summaries[:3])
    candidate_names = " ".join(
        candidate.declaration.name for candidate in failure.best_candidates[:3]
    )
    combined = " ".join(part for part in [statement, description, error_text, candidate_names] if part)
    lowered = combined.lower()

    if all(term in lowered for term in ("ecg", "bandpass", "filter")) or (
        "filter" in lowered and "coeff" in lowered
    ):
        return [
            {
                "name": "Design Filter",
                "description": "Compute stable filter coefficients for the target signal.",
                "type_signature": "",
            },
            {
                "name": "Apply Filter",
                "description": "Apply the designed filter to produce the filtered signal.",
                "type_signature": "",
            },
        ]

    if ("shortest path" in lowered or "distance" in lowered) and (
        "graph" in lowered or "dijkstra" in lowered or "weighted" in lowered
    ):
        return [
            {
                "name": "Initialize Distances",
                "description": "Initialize the source distance and default values for remaining nodes.",
                "type_signature": "",
            },
            {
                "name": "Relax Edges",
                "description": "Iteratively relax edges to improve shortest-path distance estimates.",
                "type_signature": "",
            },
        ]

    if "spd" in lowered or "symmetric positive definite" in lowered or "cholesky" in lowered:
        return [
            {
                "name": "Cholesky Factor",
                "description": "Factor the system into a triangular representation.",
                "type_signature": "",
            },
            {
                "name": "Triangular Solve",
                "description": "Use the triangular factors to solve for the output vector.",
                "type_signature": "",
            },
        ]

    if "longest common subsequence" in lowered or (
        "subsequence" in lowered and "dynamic" in lowered
    ) or " lcs " in f" {lowered} ":
        return [
            {
                "name": "Build DP Table",
                "description": "Compute the dynamic-programming table for subsequence lengths.",
                "type_signature": "",
            },
            {
                "name": "Backtrack Subsequence",
                "description": "Recover the longest common subsequence from the DP table.",
                "type_signature": "",
            },
        ]

    # Matrix factorization / decomposition
    if any(term in lowered for term in ("matrix factori", "svd", "eigenvalue", "eigen decomp")):
        return [
            {
                "name": "Factorize Matrix",
                "description": "Compute the factorization of the input matrix.",
                "type_signature": "",
            },
            {
                "name": "Extract Components",
                "description": "Extract the desired components from the factorization result.",
                "type_signature": "",
            },
        ]

    # Optimization / minimization
    if any(term in lowered for term in ("optimi", "minimi", "maximi", "gradient descent", "loss function")):
        return [
            {
                "name": "Initialize Parameters",
                "description": "Set initial parameter values for the optimization.",
                "type_signature": "",
            },
            {
                "name": "Iterate Optimization",
                "description": "Run the iterative optimization loop until convergence.",
                "type_signature": "",
            },
            {
                "name": "Extract Solution",
                "description": "Extract the optimal parameters from the converged state.",
                "type_signature": "",
            },
        ]

    # FFT / spectral analysis
    if any(term in lowered for term in ("fft", "fourier", "spectral", "frequency spectrum")):
        return [
            {
                "name": "Compute Transform",
                "description": "Apply the forward frequency-domain transform to the signal.",
                "type_signature": "",
            },
            {
                "name": "Analyze Spectrum",
                "description": "Extract the relevant spectral features from the transform output.",
                "type_signature": "",
            },
        ]

    # Sorting algorithms
    if any(term in lowered for term in ("sort", "order", "arrange", "rank element")):
        return [
            {
                "name": "Partition Elements",
                "description": "Divide elements into sub-groups for ordered processing.",
                "type_signature": "",
            },
            {
                "name": "Merge Ordered",
                "description": "Combine partitioned sub-groups into the final sorted order.",
                "type_signature": "",
            },
        ]

    # String matching / edit distance
    if any(term in lowered for term in ("edit distance", "levenshtein", "string matching", "string align")):
        return [
            {
                "name": "Build Distance Table",
                "description": "Compute the edit distance table between the two sequences.",
                "type_signature": "",
            },
            {
                "name": "Trace Alignment",
                "description": "Recover the optimal alignment or edit operations from the table.",
                "type_signature": "",
            },
        ]

    # Signal processing: detect + compute pattern (e.g. R-peak → heart rate)
    if ("detect" in lowered or "peak" in lowered) and ("comput" in lowered or "rate" in lowered or "measur" in lowered):
        return [
            {
                "name": "Detect Features",
                "description": "Identify the key features or events in the input signal.",
                "type_signature": "",
            },
            {
                "name": "Compute Metric",
                "description": "Derive the target metric from the detected features.",
                "type_signature": "",
            },
        ]

    # Interpolation / approximation
    if any(term in lowered for term in ("interpolat", "spline", "approximat", "curve fit")):
        return [
            {
                "name": "Fit Model",
                "description": "Fit the interpolation or approximation model to the data points.",
                "type_signature": "",
            },
            {
                "name": "Evaluate Model",
                "description": "Evaluate the fitted model at the target query points.",
                "type_signature": "",
            },
        ]

    # Clustering
    if any(term in lowered for term in ("cluster", "k-means", "kmeans", "dbscan", "group similar")):
        return [
            {
                "name": "Assign Clusters",
                "description": "Assign data points to cluster centroids or groups.",
                "type_signature": "",
            },
            {
                "name": "Refine Centroids",
                "description": "Update cluster centroids or boundaries based on assignments.",
                "type_signature": "",
            },
        ]

    # Statistical inference / Bayesian
    if any(term in lowered for term in ("posterior", "bayesian", "likelihood", "prior", "inference", "sampling")):
        return [
            {
                "name": "Specify Model",
                "description": "Define the probabilistic model with prior and likelihood.",
                "type_signature": "",
            },
            {
                "name": "Fit Or Sample",
                "description": "Perform inference via sampling or optimization.",
                "type_signature": "",
            },
            {
                "name": "Summarize Results",
                "description": "Extract posterior summaries or point estimates.",
                "type_signature": "",
            },
        ]

    # Tree / graph traversal
    if any(term in lowered for term in ("travers", "bfs", "dfs", "breadth first", "depth first", "search tree")):
        return [
            {
                "name": "Initialize Traversal",
                "description": "Set up the traversal frontier with the starting node(s).",
                "type_signature": "",
            },
            {
                "name": "Explore Neighbors",
                "description": "Iteratively visit neighbors and collect results.",
                "type_signature": "",
            },
        ]

    # Convolution / correlation
    if any(term in lowered for term in ("convolv", "convolution", "correlat", "cross-correlat")):
        return [
            {
                "name": "Prepare Kernel",
                "description": "Set up the convolution or correlation kernel.",
                "type_signature": "",
            },
            {
                "name": "Apply Convolution",
                "description": "Apply the kernel operation across the input signal.",
                "type_signature": "",
            },
        ]

    # Normalization / standardization
    if any(term in lowered for term in ("normaliz", "standardiz", "scale feature", "feature scal")):
        return [
            {
                "name": "Compute Statistics",
                "description": "Compute the normalization statistics (mean, std, min, max).",
                "type_signature": "",
            },
            {
                "name": "Apply Transform",
                "description": "Apply the normalization transform to the data.",
                "type_signature": "",
            },
        ]

    return _split_on_connectors(description or statement)


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
        original = _find_cdg_node(cdg, node.predicate_id)
        deterministic_sub_nodes = _deterministic_split_subnodes(failure, original)
        if deterministic_sub_nodes and original is not None:
            _apply_split_subnodes(cdg, original, deterministic_sub_nodes)
            logger.info(
                "Deterministically split node %s into %d sub-nodes",
                node.predicate_id,
                len(deterministic_sub_nodes),
            )
            return cdg

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
            if isinstance(sub_nodes, list) and sub_nodes and original is not None:
                _apply_split_subnodes(cdg, original, sub_nodes)

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
    hunter_concurrency: int = 1,
) -> OrchestratorResult:
    """Run the full Architect -> Hunter feedback loop.

    Args:
        cdg: Initial CDG from the Architect.
        hunter_agent: A RetrievalAgent (has find_match method).
        llm: An LLM client for refinement.
        prover: Target prover.
        max_rounds: Maximum refinement rounds.
        hunter_concurrency: Maximum concurrent Hunter node matches per round.

    Returns:
        OrchestratorResult with final CDG and match results.
    """
    result = OrchestratorResult(cdg=cdg, match_results=[])

    architect_issues = cdg.architect_issues()
    if architect_issues:
        message = "; ".join(architect_issues)
        logger.warning("Architect produced blocked CDG: %s", message)
        update_stage(
            stage="orchestration",
            status="failed",
            message=message,
            completed=0,
            total=max_rounds,
        )
        log_event(
            "orchestrator",
            "round",
            "ARCHITECT_BLOCKED",
            payload={"issues": architect_issues},
        )
        return result

    for round_num in range(max_rounds):
        result.rounds_used = round_num + 1
        logger.info("Orchestration round %d/%d", round_num + 1, max_rounds)
        update_stage(
            stage="orchestration",
            status="running",
            message=f"round {round_num + 1}/{max_rounds}",
            completed=round_num,
            total=max_rounds,
        )
        log_event(
            "orchestrator",
            "round",
            "ROUND_START",
            payload={"round_num": round_num + 1, "max_rounds": max_rounds},
        )

        # Convert current CDG to PDG nodes
        try:
            pdg_nodes = to_pdg_nodes(cdg, prover=prover, strict=False)
        except ValueError as exc:
            logger.warning("CDG conversion failed: %s", exc)
            log_event(
                "orchestrator",
                "round",
                "CDG_CONVERSION_FAILED",
                payload={"error": str(exc)},
            )
            break

        if not pdg_nodes:
            logger.info("No atomic nodes to match")
            log_event("orchestrator", "round", "NO_ATOMIC_NODES")
            break

        # Filter out already-matched nodes
        matched_ids = {
            mr.pdg_node.predicate_id for mr in result.match_results if mr.success
        }
        pending_nodes = [n for n in pdg_nodes if n.predicate_id not in matched_ids]

        if not pending_nodes:
            logger.info("All nodes matched")
            log_event("orchestrator", "round", "ALL_MATCHED_EARLY")
            break

        # Run Hunter on pending nodes
        round_failures: list[MatchFailureReport] = []
        round_results: list[MatchResult] = []
        hunter_stage = f"hunter_round_{round_num + 1}"
        with telemetry_stage(
            hunter_stage,
            message=f"pending={len(pending_nodes)}",
            total=len(pending_nodes),
        ):
            with telemetry_scope(stage=hunter_stage):
                if hunter_concurrency <= 1 or len(pending_nodes) <= 1:
                    for i, pdg_node in enumerate(pending_nodes, start=1):
                        round_results.append(await hunter_agent.find_match(pdg_node))
                        update_stage(
                            stage=hunter_stage,
                            completed=i,
                            total=len(pending_nodes),
                        )
                else:
                    semaphore = asyncio.Semaphore(max(1, hunter_concurrency))

                    async def _run_hunter(pdg_node: Any) -> MatchResult:
                        async with semaphore:
                            return await hunter_agent.find_match(pdg_node)

                    round_results = list(
                        await asyncio.gather(
                            *[_run_hunter(node) for node in pending_nodes]
                        )
                    )
                    update_stage(
                        stage=hunter_stage,
                        completed=len(round_results),
                        total=len(pending_nodes),
                    )

        for match_result in round_results:
            pdg_node = match_result.pdg_node
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

        log_event(
            "orchestrator",
            "hunter",
            "HUNTER_ROUND_DONE",
            payload={
                "round_num": round_num + 1,
                "pending_nodes": len(pending_nodes),
                "matches_succeeded": sum(1 for mr in round_results if mr.success),
                "matches_failed": len(round_failures),
            },
        )

        if not round_failures:
            logger.info("All nodes matched in round %d", round_num + 1)
            log_event(
                "orchestrator",
                "round",
                "ROUND_ALL_MATCHED",
                payload={"round_num": round_num + 1},
            )
            break

        # If this is the last round, don't refine
        if round_num >= max_rounds - 1:
            for f in round_failures:
                result.ungroundable.append(f.pdg_node.predicate_id)
            log_event(
                "orchestrator",
                "refine",
                "MAX_ROUNDS_REACHED",
                payload={"ungroundable": list(result.ungroundable)},
            )
            break

        # Refine CDG based on failures
        refine_stage = f"refine_round_{round_num + 1}"
        with telemetry_stage(
            refine_stage,
            message=f"failures={len(round_failures)}",
            total=len(round_failures),
        ):
            with telemetry_scope(stage=refine_stage):
                for i, failure in enumerate(round_failures, start=1):
                    cdg = await refine_on_failure(failure, cdg, llm)
                    update_stage(
                        stage=refine_stage,
                        completed=i,
                        total=len(round_failures),
                    )

        result.cdg = cdg

    update_stage(
        stage="orchestration",
        status="completed",
        completed=result.rounds_used,
        total=max_rounds,
    )
    log_event(
        "orchestrator",
        "run",
        "ORCHESTRATION_DONE",
        payload={
            "rounds_used": result.rounds_used,
            "matches_total": len(result.match_results),
            "matches_success": sum(1 for mr in result.match_results if mr.success),
            "ungroundable": list(result.ungroundable),
        },
    )
    return result
