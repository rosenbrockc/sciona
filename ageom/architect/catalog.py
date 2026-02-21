"""Primitive catalog — the searchable 'alphabet' of known algorithmic operations."""

from __future__ import annotations

import json
from pathlib import Path

from ageom.architect.models import AlgorithmicNode, AlgorithmicPrimitive, ConceptType, IOSpec


class PrimitiveCatalog:
    """In-memory catalog of algorithmic primitives with category indexing.

    Acts as the stop-condition oracle: if a CDG node matches a primitive,
    the Decomposer marks it as atomic.
    """

    def __init__(self) -> None:
        self._primitives: dict[str, AlgorithmicPrimitive] = {}
        self._by_category: dict[ConceptType, list[AlgorithmicPrimitive]] = {}

    @property
    def size(self) -> int:
        return len(self._primitives)

    def add(self, primitive: AlgorithmicPrimitive) -> None:
        """Add a primitive to the catalog."""
        self._primitives[primitive.name] = primitive
        self._by_category.setdefault(primitive.category, []).append(primitive)

    def get(self, name: str) -> AlgorithmicPrimitive | None:
        """Look up a primitive by name."""
        return self._primitives.get(name)

    def search_by_category(
        self, concept_type: ConceptType
    ) -> list[AlgorithmicPrimitive]:
        """Return all primitives in a given category."""
        return list(self._by_category.get(concept_type, []))

    def all_primitives(self) -> list[AlgorithmicPrimitive]:
        """Return all primitives in the catalog."""
        return list(self._primitives.values())

    def is_atomic(self, node: AlgorithmicNode) -> bool:
        """Check if a node matches any known primitive by name."""
        if node.matched_primitive and node.matched_primitive in self._primitives:
            return True
        # Also try matching by node name (case-insensitive)
        name_lower = node.name.lower().replace(" ", "_")
        for prim_name in self._primitives:
            if prim_name.lower().replace(" ", "_") == name_lower:
                return True
        return False

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
            scored.append((overlap, prim))

        # If not enough from same category, add cross-category matches
        if len(scored) < k:
            for prim in self._primitives.values():
                if prim.category != node.concept_type:
                    prim_words = set(prim.description.lower().split())
                    overlap = len(node_words & prim_words)
                    scored.append((overlap, prim))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [prim for _, prim in scored[:k]]

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
            IOSpec(name="mass_matrix", type_desc="ndarray", constraints="positive definite"),
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
            IOSpec(name="theta_minus", type_desc="ndarray", constraints="leftmost position"),
            IOSpec(name="theta_plus", type_desc="ndarray", constraints="rightmost position"),
            IOSpec(name="r_minus", type_desc="ndarray", constraints="leftmost momentum"),
            IOSpec(name="r_plus", type_desc="ndarray", constraints="rightmost momentum"),
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
            IOSpec(name="x_pred", type_desc="ndarray", constraints="predicted state mean"),
            IOSpec(name="P_pred", type_desc="ndarray", constraints="predicted covariance, symmetric positive semi-definite"),
            IOSpec(name="H", type_desc="ndarray", constraints="observation model matrix"),
            IOSpec(name="R", type_desc="ndarray", constraints="observation noise covariance, symmetric positive definite"),
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
            IOSpec(name="factor_potential", type_desc="ndarray", constraints="non-negative tensor"),
            IOSpec(name="incoming_messages", type_desc="dict[str, ndarray]",
                   constraints="messages from all neighboring variables except target"),
            IOSpec(name="target_variable", type_desc="str"),
        ],
        outputs=[
            IOSpec(name="message", type_desc="ndarray", constraints="marginalized message to target variable"),
        ],
        type_signature="ndarray -> dict[str, ndarray] -> str -> ndarray",
    ),
]


def seed_bayesian_primitives(catalog: PrimitiveCatalog) -> None:
    """Add built-in Bayesian primitives to an existing catalog."""
    for prim in _BAYESIAN_PRIMITIVES:
        if catalog.get(prim.name) is None:
            catalog.add(prim)
