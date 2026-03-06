"""Primitive catalog — the searchable 'alphabet' of known algorithmic operations."""

from __future__ import annotations

import json
from pathlib import Path

from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
)


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

    def is_atomic(self, node: AlgorithmicNode) -> bool:
        """Check if a node matches any known primitive by name."""
        if node.matched_primitive and self.get(node.matched_primitive) is not None:
            return True
        return self.get(node.name) is not None

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
                "requirements such as sample rate, passbands, ripple, and "
                "attenuation constraints."
            ),
            inputs=[IOSpec(name="spec", type_desc="filter specification")],
            outputs=[
                IOSpec(
                    name="design_requirements",
                    type_desc="filter design requirements",
                )
            ],
            type_signature="filter specification -> filter design requirements",
        ),
        ["parse filter requirements", "normalize specification", "interpret filter specification"],
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


def seed_bayesian_primitives(catalog: PrimitiveCatalog) -> None:
    """Add built-in Bayesian primitives to an existing catalog."""
    for prim in _BAYESIAN_PRIMITIVES:
        if catalog.get(prim.name) is None:
            catalog.add(prim)


def seed_builtin_primitives(catalog: PrimitiveCatalog) -> None:
    """Add built-in primitives used by deterministic architect flows."""
    seed_bayesian_primitives(catalog)
    for prim, aliases in _SIGNAL_FILTER_PRIMITIVES:
        if catalog.get(prim.name) is None:
            catalog.add(prim)
        for alias in aliases:
            catalog.add_alias(alias, prim.name)
