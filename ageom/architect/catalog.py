"""Primitive catalog — the searchable 'alphabet' of known algorithmic operations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
)

if TYPE_CHECKING:
    from ageom.architect.embedder import SkillIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# De-duplication data models
# ---------------------------------------------------------------------------


@dataclass
class DedupResult:
    """Outcome of comparing a candidate against the catalog."""

    is_duplicate: bool
    incumbent_name: str | None = None
    similarity: float = 0.0
    structural_match: bool = False


@dataclass
class CatalogReport:
    """Summary of catalog population with de-duplication metrics."""

    total_candidates: int = 0
    added: int = 0
    merged: int = 0
    structural_skips: int = 0
    source_live_registry_candidates: int = 0
    source_ast_candidates: int = 0
    source_cdg_metadata_matches: int = 0
    source_witness_doc_fallbacks: int = 0
    source_witness_signature_fallbacks: int = 0
    merge_details: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogConfidence:
    """Heuristic confidence that a task text maps onto known catalog primitives."""

    score: float
    exact_matches: tuple[str, ...] = ()
    strong_matches: tuple[str, ...] = ()
    max_overlap: float = 0.0


class PrimitiveCatalog:
    """In-memory catalog of algorithmic primitives with category indexing.

    Acts as the stop-condition oracle: if a CDG node matches a primitive,
    the Decomposer marks it as atomic.
    """

    def __init__(self) -> None:
        self._primitives: dict[str, AlgorithmicPrimitive] = {}
        self._by_category: dict[ConceptType, list[AlgorithmicPrimitive]] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _normalize_key(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    @property
    def size(self) -> int:
        return len(self._primitives)

    def add(self, primitive: AlgorithmicPrimitive) -> None:
        """Add a primitive to the catalog."""
        existing = self._primitives.get(primitive.name)
        if existing is not None:
            bucket = self._by_category.get(existing.category, [])
            self._by_category[existing.category] = [
                item for item in bucket if item.name != primitive.name
            ]
        self._primitives[primitive.name] = primitive
        bucket = self._by_category.setdefault(primitive.category, [])
        if all(item.name != primitive.name for item in bucket):
            bucket.append(primitive)
        self._aliases.setdefault(
            self._normalize_key(primitive.name),
            primitive.name,
        )

    def add_alias(self, alias: str, primitive_name: str) -> None:
        """Register an alternate lookup key for an existing primitive."""
        if primitive_name not in self._primitives:
            raise KeyError(f"Unknown primitive '{primitive_name}'")
        self._aliases[self._normalize_key(alias)] = primitive_name

    def get(self, name: str) -> AlgorithmicPrimitive | None:
        """Look up a primitive by name."""
        direct = self._primitives.get(name)
        if direct is not None:
            return direct
        alias = self._aliases.get(self._normalize_key(name))
        if alias is None:
            return None
        return self._primitives.get(alias)

    def search_by_category(
        self, concept_type: ConceptType
    ) -> list[AlgorithmicPrimitive]:
        """Return all primitives in a given category."""
        return list(self._by_category.get(concept_type, []))

    def all_primitives(self) -> list[AlgorithmicPrimitive]:
        """Return all primitives in the catalog."""
        return list(self._primitives.values())

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) >= 3
        }

    def is_atomic(self, node: AlgorithmicNode) -> bool:
        """Check if a node matches any known primitive by name."""
        if node.matched_primitive and self.get(node.matched_primitive) is not None:
            return True
        return self.get(node.name) is not None

    def estimate_confidence(self, text: str) -> CatalogConfidence:
        """Estimate whether retrieval is likely to pay off for the given task text."""
        normalized_text = self._normalize_key(text)
        text_tokens = self._tokenize_text(text)
        if not normalized_text or not text_tokens or not self._primitives:
            return CatalogConfidence(score=0.0)

        exact_matches: list[str] = []
        strong_matches: list[str] = []
        max_overlap = 0.0

        seen_exact: set[str] = set()
        for alias, primitive_name in self._aliases.items():
            alias_tokens = tuple(token for token in alias.split("_") if token)
            if not alias_tokens:
                continue
            if len(alias_tokens) == 1 and len(alias_tokens[0]) < 5:
                continue
            alias_phrase = "_".join(alias_tokens)
            if alias_phrase in normalized_text and primitive_name not in seen_exact:
                exact_matches.append(primitive_name)
                seen_exact.add(primitive_name)

        for primitive in self._primitives.values():
            prim_tokens = set(self._normalize_key(primitive.name).split("_"))
            prim_tokens |= self._tokenize_text(primitive.description)
            prim_tokens = {token for token in prim_tokens if token}
            if not prim_tokens:
                continue
            overlap = len(text_tokens & prim_tokens) / max(1, len(prim_tokens))
            max_overlap = max(max_overlap, overlap)
            if overlap >= 0.2:
                strong_matches.append(primitive.name)

        exact_score = 0.55 if exact_matches else 0.0
        strong_score = min(0.25, len(strong_matches) * 0.05)
        overlap_score = min(0.20, max_overlap * 0.50)
        score = min(1.0, exact_score + strong_score + overlap_score)

        return CatalogConfidence(
            score=score,
            exact_matches=tuple(exact_matches[:5]),
            strong_matches=tuple(strong_matches[:5]),
            max_overlap=max_overlap,
        )

    # ------------------------------------------------------------------
    # De-duplication
    # ------------------------------------------------------------------

    @staticmethod
    def _structural_match(
        a: AlgorithmicPrimitive, b: AlgorithmicPrimitive
    ) -> bool:
        """Check category equality and IO arity compatibility."""
        if a.category != b.category:
            return False
        a_req = len([p for p in a.inputs if p.required])
        b_req = len([p for p in b.inputs if p.required])
        if abs(a_req - b_req) > 1:
            return False
        if abs(len(a.outputs) - len(b.outputs)) > 1:
            return False
        return True

    @staticmethod
    def _richer(
        a: AlgorithmicPrimitive, b: AlgorithmicPrimitive
    ) -> AlgorithmicPrimitive:
        """Return the primitive with richer metadata."""

        def _score(p: AlgorithmicPrimitive) -> int:
            s = len(p.description)
            s += len(p.inputs) * 10
            s += len(p.outputs) * 10
            s += 50 if p.type_signature else 0
            return s

        return a if _score(a) >= _score(b) else b

    def check_duplicate(
        self,
        candidate: AlgorithmicPrimitive,
        skill_index: SkillIndex | None = None,
        threshold: float = 0.85,
    ) -> DedupResult:
        """Check if *candidate* is a semantic duplicate of an existing primitive.

        1. Exact name match -> always duplicate.
        2. If *skill_index* is provided, query top-1 by embedding similarity.
           If similarity >= *threshold* AND structural match -> duplicate.
        3. Otherwise -> not duplicate.
        """
        existing = self.get(candidate.name)
        if existing is not None:
            return DedupResult(
                is_duplicate=True,
                incumbent_name=existing.name,
                similarity=1.0,
                structural_match=True,
            )

        if skill_index is None:
            return DedupResult(is_duplicate=False)

        hits = skill_index.search_by_embedding(
            skill_index._primitive_to_text(candidate), k=1
        )
        if not hits:
            return DedupResult(is_duplicate=False)

        decl, score = hits[0]
        incumbent = self.get(decl.name)
        if incumbent is None:
            return DedupResult(is_duplicate=False, similarity=score)

        struct = self._structural_match(candidate, incumbent)
        if score >= threshold and struct:
            return DedupResult(
                is_duplicate=True,
                incumbent_name=incumbent.name,
                similarity=score,
                structural_match=True,
            )
        return DedupResult(
            is_duplicate=False,
            incumbent_name=incumbent.name,
            similarity=score,
            structural_match=struct,
        )

    def add_with_dedup(
        self,
        candidate: AlgorithmicPrimitive,
        skill_index: SkillIndex | None = None,
        threshold: float = 0.85,
        report: CatalogReport | None = None,
    ) -> DedupResult:
        """Add a primitive, merging with the incumbent if duplicate.

        When merging, the primitive with richer metadata wins and the
        other's name is registered as an alias.

        Returns the :class:`DedupResult`.
        """
        if report is not None:
            report.total_candidates += 1

        result = self.check_duplicate(candidate, skill_index, threshold)

        if not result.is_duplicate:
            self.add(candidate)
            if report is not None:
                report.added += 1
            return result

        # Merge: keep the richer primitive, alias the loser.
        assert result.incumbent_name is not None
        incumbent = self.get(result.incumbent_name)
        assert incumbent is not None

        winner = self._richer(incumbent, candidate)
        loser_name = (
            candidate.name if winner.name == incumbent.name else incumbent.name
        )

        # Remove the loser from _primitives so it doesn't shadow the alias.
        if loser_name != winner.name and loser_name in self._primitives:
            loser = self._primitives.pop(loser_name)
            bucket = self._by_category.get(loser.category, [])
            self._by_category[loser.category] = [
                p for p in bucket if p.name != loser_name
            ]

        self.add(winner)
        if loser_name != winner.name:
            self._aliases[self._normalize_key(loser_name)] = winner.name

        if report is not None:
            report.merged += 1
            report.merge_details.append(
                (candidate.name, result.incumbent_name, result.similarity)
            )

        logger.debug(
            "Merged primitive '%s' into '%s' (similarity=%.3f)",
            candidate.name,
            winner.name,
            result.similarity,
        )
        return result

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def find_gaps(
        self,
        fallback_nodes: list[AlgorithmicNode],
        skill_index: SkillIndex | None = None,
        similarity_ceiling: float = 0.6,
    ) -> list[list[AlgorithmicNode]]:
        """Cluster fallback nodes that don't match any primitive well.

        Returns groups of similar unmatched nodes (each group is a gap).
        """
        unmatched: list[AlgorithmicNode] = []
        for node in fallback_nodes:
            if self.get(node.name) is not None:
                continue
            if node.matched_primitive and self.get(node.matched_primitive) is not None:
                continue
            if skill_index is not None:
                hits = skill_index.search_by_embedding(node.description, k=1)
                if hits and hits[0][1] >= similarity_ceiling:
                    continue
            unmatched.append(node)

        if not unmatched:
            return []

        # Greedy clustering by pairwise keyword overlap (no torch needed).
        clusters: list[list[AlgorithmicNode]] = []
        used: set[int] = set()
        for i, node_a in enumerate(unmatched):
            if i in used:
                continue
            cluster = [node_a]
            used.add(i)
            words_a = set(node_a.description.lower().split())
            for j in range(i + 1, len(unmatched)):
                if j in used:
                    continue
                words_b = set(unmatched[j].description.lower().split())
                union = words_a | words_b
                if union and len(words_a & words_b) / len(union) > 0.35:
                    cluster.append(unmatched[j])
                    used.add(j)
            clusters.append(cluster)

        # Only return clusters with 2+ members (single nodes aren't a pattern).
        return [c for c in clusters if len(c) >= 2]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_matching_primitives(
        self, node: AlgorithmicNode, k: int = 5
    ) -> list[AlgorithmicPrimitive]:
        """Find primitives matching a node by category, then by keyword overlap.

        For full semantic search, use SkillIndex.search() instead.
        """
        # First: exact category matches
        category_matches = self.search_by_category(node.concept_type)

        # Score by keyword overlap between node description and primitive description
        node_words = set(node.description.lower().split())
        scored: list[tuple[float, AlgorithmicPrimitive]] = []
        for prim in category_matches:
            prim_words = set(prim.description.lower().split())
            overlap = len(node_words & prim_words)
            scored.append((overlap + self._arity_match_bonus(node, prim), prim))

        # If not enough from same category, add cross-category matches
        if len(scored) < k:
            for prim in self._primitives.values():
                if prim.category != node.concept_type:
                    prim_words = set(prim.description.lower().split())
                    overlap = len(node_words & prim_words)
                    scored.append((overlap + self._arity_match_bonus(node, prim), prim))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [prim for _, prim in scored[:k]]

    @staticmethod
    def _arity_match_bonus(node: AlgorithmicNode, prim: AlgorithmicPrimitive) -> float:
        node_in = len(node.inputs)
        node_out = len(node.outputs)
        prim_required_in = len([port for port in prim.inputs if port.required])
        prim_total_in = len(prim.inputs)
        prim_out = len(prim.outputs)

        score = 0.0
        if prim_required_in <= node_in <= prim_total_in:
            score += 1.5
        else:
            score -= abs(node_in - prim_required_in) * 0.25

        if node_out == prim_out:
            score += 0.75
        elif prim_out:
            score -= abs(node_out - prim_out) * 0.1
        return score

    def save(self, path: str | Path) -> None:
        """Persist the catalog to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [prim.model_dump() for prim in self._primitives.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PrimitiveCatalog:
        """Load a catalog from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        catalog = cls()
        for item in data:
            catalog.add(AlgorithmicPrimitive.model_validate(item))
        return catalog


# ---------------------------------------------------------------------------
# Built-in Bayesian / probabilistic primitives
# ---------------------------------------------------------------------------

_BAYESIAN_PRIMITIVES: list[AlgorithmicPrimitive] = [
    AlgorithmicPrimitive(
        name="leapfrog_integrator_step",
        source="bayesian-inference",
        category=ConceptType.MCMC_KERNEL,
        description=(
            "Single leapfrog integrator step for Hamiltonian Monte Carlo: "
            "half-step momentum, full-step position, half-step momentum. "
            "Requires stateless log-density gradient oracle (Oracle Isolation)."
        ),
        inputs=[
            IOSpec(name="q", type_desc="ndarray", constraints="current position"),
            IOSpec(name="p", type_desc="ndarray", constraints="current momentum"),
            IOSpec(name="epsilon", type_desc="float", constraints="step size > 0"),
            IOSpec(name="grad_log_density", type_desc="Callable[[ndarray], ndarray]"),
            IOSpec(
                name="mass_matrix", type_desc="ndarray", constraints="positive definite"
            ),
        ],
        outputs=[
            IOSpec(name="q_new", type_desc="ndarray"),
            IOSpec(name="p_new", type_desc="ndarray"),
        ],
        type_signature="ndarray -> ndarray -> float -> (ndarray -> ndarray) -> ndarray -> ndarray × ndarray",
    ),
    AlgorithmicPrimitive(
        name="nuts_u_turn_check",
        source="bayesian-inference",
        category=ConceptType.MCMC_KERNEL,
        description=(
            "No-U-Turn Sampler criterion: check whether the leapfrog trajectory "
            "has made a U-turn by testing (theta_plus - theta_minus) · r_minus >= 0 "
            "and (theta_plus - theta_minus) · r_plus >= 0."
        ),
        inputs=[
            IOSpec(
                name="theta_minus", type_desc="ndarray", constraints="leftmost position"
            ),
            IOSpec(
                name="theta_plus", type_desc="ndarray", constraints="rightmost position"
            ),
            IOSpec(
                name="r_minus", type_desc="ndarray", constraints="leftmost momentum"
            ),
            IOSpec(
                name="r_plus", type_desc="ndarray", constraints="rightmost momentum"
            ),
        ],
        outputs=[
            IOSpec(name="u_turn", type_desc="bool"),
        ],
        type_signature="ndarray -> ndarray -> ndarray -> ndarray -> bool",
    ),
    AlgorithmicPrimitive(
        name="kalman_gain_update",
        source="bayesian-inference",
        category=ConceptType.CONJUGATE_UPDATE,
        description=(
            "Compute the Kalman gain matrix K = P_pred @ H^T @ (H @ P_pred @ H^T + R)^{-1} "
            "and apply the conjugate Gaussian update to obtain the posterior state and covariance. "
            "State Decoupling: covariance P flows explicitly as an edge, never as hidden state."
        ),
        inputs=[
            IOSpec(
                name="x_pred", type_desc="ndarray", constraints="predicted state mean"
            ),
            IOSpec(
                name="P_pred",
                type_desc="ndarray",
                constraints="predicted covariance, symmetric positive semi-definite",
            ),
            IOSpec(
                name="H", type_desc="ndarray", constraints="observation model matrix"
            ),
            IOSpec(
                name="R",
                type_desc="ndarray",
                constraints="observation noise covariance, symmetric positive definite",
            ),
            IOSpec(name="z", type_desc="ndarray", constraints="observation vector"),
        ],
        outputs=[
            IOSpec(name="x_updated", type_desc="ndarray"),
            IOSpec(name="P_updated", type_desc="ndarray"),
            IOSpec(name="K", type_desc="ndarray"),
        ],
        type_signature="ndarray -> ndarray -> ndarray -> ndarray -> ndarray -> ndarray × ndarray × ndarray",
    ),
    AlgorithmicPrimitive(
        name="sum_product_marginalization",
        source="bayesian-inference",
        category=ConceptType.MESSAGE_PASSING,
        description=(
            "Sum-product marginalization on a factor graph: for a given factor node, "
            "marginalize the joint factor potential over all variables except the target, "
            "weighted by incoming messages. Used in belief propagation / junction tree."
        ),
        inputs=[
            IOSpec(
                name="factor_potential",
                type_desc="ndarray",
                constraints="non-negative tensor",
            ),
            IOSpec(
                name="incoming_messages",
                type_desc="dict[str, ndarray]",
                constraints="messages from all neighboring variables except target",
            ),
            IOSpec(name="target_variable", type_desc="str"),
        ],
        outputs=[
            IOSpec(
                name="message",
                type_desc="ndarray",
                constraints="marginalized message to target variable",
            ),
        ],
        type_signature="ndarray -> dict[str, ndarray] -> str -> ndarray",
    ),
]


_SIGNAL_FILTER_PRIMITIVES: list[tuple[AlgorithmicPrimitive, list[str]]] = [
    (
        AlgorithmicPrimitive(
            name="parse_filter_spec",
            source="ageom-builtins",
            category=ConceptType.DATA_ASSEMBLY,
            description=(
                "Parse and canonicalize a filter specification into typed design "
                "targets such as sample rate, passbands, ripple, and "
                "attenuation constraints."
            ),
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[
                IOSpec(
                    name="design_targets",
                    type_desc="filter design targets",
                )
            ],
            type_signature="filter specification -> filter design targets",
        ),
        [
            "parse filter requirements",
            "normalize specification",
            "interpret filter specification",
            "normalize design targets",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="choose_filter_topology",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description=(
                "Choose the filter family, order, and realization strategy from "
                "typed design targets."
            ),
            inputs=[IOSpec(name="design_targets", type_desc="filter design targets")],
            outputs=[IOSpec(name="design_strategy", type_desc="filter design strategy")],
            type_signature="filter design targets -> filter design strategy",
        ),
        [
            "select filter family",
            "select filter topology",
            "choose filter strategy",
            "select filter architecture",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="design_filter_coefficients",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Generate candidate filter coefficients from the chosen strategy.",
            inputs=[IOSpec(name="design_strategy", type_desc="filter design strategy")],
            outputs=[
                IOSpec(name="candidate_coefficients", type_desc="filter coefficients")
            ],
            type_signature="filter design strategy -> filter coefficients",
        ),
        [
            "synthesize coefficients",
            "synthesize candidate coefficients",
            "select final coefficients",
            "design core",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="validate_filter_response",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description=(
                "Check frequency-response compliance and finalize a coefficient vector "
                "that satisfies the design targets."
            ),
            inputs=[
                IOSpec(name="candidate_coefficients", type_desc="filter coefficients"),
                IOSpec(name="design_targets", type_desc="filter design targets"),
            ],
            outputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            type_signature="filter coefficients -> filter design targets -> filter coefficients",
        ),
        [
            "evaluate compliance",
            "refine design",
            "compliance gate and deterministic finalization",
            "validate and finalize coefficients",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="canonicalize_filter_coefficients",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Normalize coefficient ordering and representation for downstream analysis.",
            inputs=[IOSpec(name="coefficients", type_desc="filter coefficients")],
            outputs=[
                IOSpec(name="normalized_coefficients", type_desc="filter coefficients")
            ],
            type_signature="filter coefficients -> filter coefficients",
        ),
        [
            "normalize coefficient representation",
            "canonicalize coefficient representation",
            "canonicalize filter coefficients",
            "normalize coefficient form",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="compute_pole_locations",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Compute pole locations from a filter characteristic polynomial.",
            inputs=[
                IOSpec(
                    name="characteristic_polynomial",
                    type_desc="np.polynomial.Polynomial",
                )
            ],
            outputs=[IOSpec(name="poles", type_desc="np.ndarray")],
            type_signature="np.polynomial.Polynomial -> np.ndarray",
        ),
        ["solve pole locations", "compute pole locations", "solve for pole locations"],
    ),
    (
        AlgorithmicPrimitive(
            name="construct_characteristic_polynomial",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Construct the characteristic polynomial implied by normalized filter coefficients.",
            inputs=[
                IOSpec(
                    name="normalized_coefficients",
                    type_desc="filter coefficients",
                )
            ],
            outputs=[
                IOSpec(
                    name="characteristic_polynomial",
                    type_desc="np.polynomial.Polynomial",
                )
            ],
            type_signature="filter coefficients -> np.polynomial.Polynomial",
        ),
        [
            "construct filter characteristic polynomial",
            "build characteristic polynomial",
            "construct coefficient polynomial",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="assess_discrete_time_stability",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Assess stability margins from filter poles in the discrete-time plane.",
            inputs=[IOSpec(name="poles", type_desc="np.ndarray")],
            outputs=[IOSpec(name="stability_report", type_desc="stability report")],
            type_signature="np.ndarray -> stability report",
        ),
        [
            "evaluate stability margin",
            "evaluate discrete-time stability",
            "evaluate unit-circle stability margin",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="finalize_stable_coefficients",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Emit validated coefficients after a discrete-time stability report passes acceptance criteria.",
            inputs=[
                IOSpec(name="normalized_coefficients", type_desc="filter coefficients"),
                IOSpec(name="stability_report", type_desc="stability report"),
            ],
            outputs=[IOSpec(name="valid_coefficients", type_desc="filter coefficients")],
            type_signature="filter coefficients -> stability report -> filter coefficients",
        ),
        [
            "emit stable coefficients",
            "pass stable coefficients",
            "gate coefficients by stability",
            "finalize stable coefficients",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="apply_iir_filter",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Apply stable filter coefficients to a numeric signal array.",
            inputs=[
                IOSpec(name="valid_coefficients", type_desc="filter coefficients"),
                IOSpec(name="signal", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="filtered_signal", type_desc="np.ndarray")],
            type_signature="filter coefficients -> np.ndarray -> np.ndarray",
        ),
        [
            "apply coefficients to signal",
            "apply coefficients across ecg trace",
            "execute coefficient-based filtering",
            "apply filter",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="mitigate_filter_transients",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Reduce startup and edge transients in the filtered output trace.",
            inputs=[IOSpec(name="filtered_signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="stabilized_signal", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray",
        ),
        [
            "mitigate edge and startup transients",
            "mitigate transient artifacts",
            "finalize filtered trace",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="compute_frequency_response",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Compute the complex frequency response over a sampling grid.",
            inputs=[
                IOSpec(name="normalized_coefficients", type_desc="filter coefficients"),
                IOSpec(name="frequency_grid", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="complex_response", type_desc="np.ndarray")],
            type_signature="filter coefficients -> np.ndarray -> np.ndarray",
        ),
        [
            "compute complex frequency response",
            "evaluate complex transfer response",
            "frequency response",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="summarize_frequency_response",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_FILTER,
            description="Assemble frequency and magnitude arrays into a typed response summary.",
            inputs=[IOSpec(name="complex_response", type_desc="np.ndarray")],
            outputs=[
                IOSpec(
                    name="response",
                    type_desc="tuple[np.ndarray, np.ndarray]",
                )
            ],
            type_signature="np.ndarray -> tuple[np.ndarray, np.ndarray]",
        ),
        [
            "extract response characteristics",
            "assemble frequency response output",
            "assemble frequency response tuple",
            "derive inspection views",
        ],
    ),
]

_LINEAR_ALGEBRA_PRIMITIVES: list[tuple[AlgorithmicPrimitive, list[str]]] = [
    (
        AlgorithmicPrimitive(
            name="compute_matrix_decomposition",
            source="ageom-builtins",
            category=ConceptType.ALGEBRA,
            description="Decompose a matrix into factors (LU, Cholesky, QR, or SVD).",
            inputs=[
                IOSpec(name="matrix", type_desc="np.ndarray"),
                IOSpec(
                    name="method",
                    type_desc="str",
                    required=False,
                    default_value_repr="'svd'",
                ),
            ],
            outputs=[IOSpec(name="factors", type_desc="tuple[np.ndarray, ...]")],
            type_signature="np.ndarray -> tuple[np.ndarray, ...]",
        ),
        [
            "lu decomposition",
            "cholesky decomposition",
            "qr decomposition",
            "singular value decomposition",
            "svd",
            "matrix factorization",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="solve_linear_system",
            source="ageom-builtins",
            category=ConceptType.ALGEBRA,
            description="Solve a linear system of equations Ax = b.",
            inputs=[
                IOSpec(name="A", type_desc="np.ndarray"),
                IOSpec(name="b", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="x", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray -> np.ndarray",
        ),
        ["solve linear system", "solve ax=b", "linear solver", "matrix inverse solve"],
    ),
    (
        AlgorithmicPrimitive(
            name="compute_eigen_decomposition",
            source="ageom-builtins",
            category=ConceptType.ALGEBRA,
            description="Compute eigenvalues and eigenvectors of a square matrix.",
            inputs=[IOSpec(name="matrix", type_desc="np.ndarray")],
            outputs=[
                IOSpec(name="eigenvalues", type_desc="np.ndarray"),
                IOSpec(name="eigenvectors", type_desc="np.ndarray"),
            ],
            type_signature="np.ndarray -> np.ndarray × np.ndarray",
        ),
        ["eigenvalues", "eigenvectors", "eigendecomposition", "spectral decomposition"],
    ),
]

_OPTIMIZATION_PRIMITIVES: list[tuple[AlgorithmicPrimitive, list[str]]] = [
    (
        AlgorithmicPrimitive(
            name="gradient_descent_step",
            source="ageom-builtins",
            category=ConceptType.ANALYSIS,
            description="Perform a single step of gradient descent optimization.",
            inputs=[
                IOSpec(name="x", type_desc="np.ndarray"),
                IOSpec(name="gradient", type_desc="np.ndarray"),
                IOSpec(name="learning_rate", type_desc="float"),
            ],
            outputs=[IOSpec(name="x_new", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray -> float -> np.ndarray",
        ),
        ["gradient step", "update parameters", "descent step"],
    ),
    (
        AlgorithmicPrimitive(
            name="compute_objective_gradient",
            source="ageom-builtins",
            category=ConceptType.ANALYSIS,
            description="Evaluate the gradient of an objective function at a point.",
            inputs=[
                IOSpec(name="objective_fn", type_desc="Callable[[np.ndarray], float]"),
                IOSpec(name="x", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="gradient", type_desc="np.ndarray")],
            type_signature="(np.ndarray -> float) -> np.ndarray -> np.ndarray",
        ),
        ["evaluate gradient", "objective gradient", "compute jacobian"],
    ),
    (
        AlgorithmicPrimitive(
            name="line_search_optimization",
            source="ageom-builtins",
            category=ConceptType.ANALYSIS,
            description="Determine the optimal step size along a descent direction.",
            inputs=[
                IOSpec(name="objective_fn", type_desc="Callable[[np.ndarray], float]"),
                IOSpec(name="x", type_desc="np.ndarray"),
                IOSpec(name="direction", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="alpha", type_desc="float")],
            type_signature="(np.ndarray -> float) -> np.ndarray -> np.ndarray -> float",
        ),
        ["line search", "backtracking line search", "step size optimization"],
    ),
]

_GRAPH_ALGEBRA_PRIMITIVES: list[tuple[AlgorithmicPrimitive, list[str]]] = [
    (
        AlgorithmicPrimitive(
            name="compute_graph_laplacian",
            source="ageom-builtins",
            category=ConceptType.GRAPH_SIGNAL_PROCESSING,
            description="Compute the combinatorial or normalized Laplacian matrix of a graph.",
            inputs=[
                IOSpec(name="adjacency", type_desc="np.ndarray"),
                IOSpec(
                    name="normalized",
                    type_desc="bool",
                    required=False,
                    default_value_repr="True",
                ),
            ],
            outputs=[IOSpec(name="laplacian", type_desc="np.ndarray")],
            type_signature="np.ndarray -> bool -> np.ndarray",
        ),
        ["graph laplacian", "discrete laplacian", "compute laplacian"],
    ),
    (
        AlgorithmicPrimitive(
            name="dijkstra_shortest_path",
            source="ageom-builtins",
            category=ConceptType.GRAPH_OPTIMIZATION,
            description="Compute shortest paths from a source node in a weighted graph.",
            inputs=[
                IOSpec(name="adjacency", type_desc="np.ndarray"),
                IOSpec(name="source_node", type_desc="int"),
            ],
            outputs=[
                IOSpec(name="distances", type_desc="np.ndarray"),
                IOSpec(name="predecessors", type_desc="np.ndarray"),
            ],
            type_signature="np.ndarray -> int -> np.ndarray × np.ndarray",
        ),
        ["dijkstra", "shortest path", "single source shortest path"],
    ),
]

_SIGNAL_TRANSFORM_PRIMITIVES: list[tuple[AlgorithmicPrimitive, list[str]]] = [
    (
        AlgorithmicPrimitive(
            name="apply_window_function",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_TRANSFORM,
            description="Apply a deterministic window function to a numeric signal segment.",
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="windowed", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray",
        ),
        [
            "window",
            "apply window",
            "apply window function",
            "window signal",
            "windowing",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="compute_forward_transform",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_TRANSFORM,
            description="Apply a forward spectral transform such as FFT or DCT.",
            inputs=[IOSpec(name="windowed", type_desc="np.ndarray")],
            outputs=[IOSpec(name="spectrum", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray",
        ),
        [
            "forward transform",
            "compute forward transform",
            "fft",
            "forward fft",
            "dct",
            "stft analysis",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="process_spectrum",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_TRANSFORM,
            description="Modify, filter, or weight spectral coefficients in the transform domain.",
            inputs=[IOSpec(name="spectrum", type_desc="np.ndarray")],
            outputs=[IOSpec(name="modified_spectrum", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray",
        ),
        [
            "spectral processing",
            "process spectrum",
            "modify spectral coefficients",
            "spectral filtering",
            "spectral shaping",
        ],
    ),
    (
        AlgorithmicPrimitive(
            name="compute_inverse_transform",
            source="ageom-builtins",
            category=ConceptType.SIGNAL_TRANSFORM,
            description="Apply an inverse spectral transform to recover a time-domain signal.",
            inputs=[IOSpec(name="modified_spectrum", type_desc="np.ndarray")],
            outputs=[IOSpec(name="result", type_desc="np.ndarray")],
            type_signature="np.ndarray -> np.ndarray",
        ),
        [
            "inverse transform",
            "compute inverse transform",
            "inverse fft",
            "ifft",
            "reconstruct signal",
        ],
    ),
]


def seed_bayesian_primitives(catalog: PrimitiveCatalog) -> None:
    """Add built-in Bayesian primitives to an existing catalog."""
    for prim in _BAYESIAN_PRIMITIVES:
        if catalog.get(prim.name) is None:
            catalog.add(prim)


def seed_builtin_primitives(catalog: PrimitiveCatalog) -> None:
    """Add built-in primitives used by deterministic architect flows."""
    seed_bayesian_primitives(catalog)
    for primitives in [
        _SIGNAL_FILTER_PRIMITIVES,
        _SIGNAL_TRANSFORM_PRIMITIVES,
        _LINEAR_ALGEBRA_PRIMITIVES,
        _OPTIMIZATION_PRIMITIVES,
        _GRAPH_ALGEBRA_PRIMITIVES,
    ]:
        for prim, aliases in primitives:
            if catalog.get(prim.name) is None:
                catalog.add(prim)
            for alias in aliases:
                catalog.add_alias(alias, prim.name)
