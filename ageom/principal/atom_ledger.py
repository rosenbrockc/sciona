"""Atom Performance Ledger — UCB1 bandit ranking for CDG primitive selection."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ageom.architect.models import AlgorithmicNode


@dataclass(frozen=True)
class SlotSignature:
    """Structural identity of a CDG slot, independent of which atom fills it."""

    parent_name: str
    concept_type: str
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]

    @property
    def key(self) -> str:
        """16-char SHA256 hex digest for dict keying."""
        raw = f"{self.parent_name}|{self.concept_type}|{self.input_types}|{self.output_types}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class AtomObservation:
    """Single trial observation for an atom in a slot."""

    gradient_score: float
    trial: int


class AtomLedger:
    """In-memory ledger tracking atom performance per structural slot.

    Records (slot, atom) -> gradient observations across Principal trials
    and ranks candidate atoms using the UCB1 bandit algorithm.
    """

    def __init__(self) -> None:
        self._observations: dict[str, dict[str, list[AtomObservation]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(
        self,
        slot: SlotSignature,
        atom_name: str,
        gradient_score: float,
        trial: int,
    ) -> None:
        """Record a gradient observation for an atom in a slot."""
        self._observations[slot.key][atom_name].append(
            AtomObservation(gradient_score=gradient_score, trial=trial)
        )

    def rank_candidates(
        self,
        slot: SlotSignature,
        candidate_names: list[str],
        exploration_weight: float = 1.414,
        benchmark_priors: dict[str, float] | None = None,
        atom_statuses: dict[str, str] | None = None,
        prior_strength: int = 2,
    ) -> list[tuple[str, float]]:
        """Return candidates sorted by UCB1 score (descending). Untried atoms get inf.

        Parameters
        ----------
        benchmark_priors
            Optional mapping of atom name to [0, 1] benchmark prior reward.
        atom_statuses
            Optional mapping of atom name to status (e.g., ``"superseded"``).
        prior_strength
            Number of virtual observations for benchmark prior (default 2).
        """
        if not candidate_names:
            return []

        priors = benchmark_priors or {}
        statuses = atom_statuses or {}
        slot_data = self._observations.get(slot.key, {})
        total_plays = sum(len(obs) for obs in slot_data.values())

        scored: list[tuple[str, float]] = []
        for name in candidate_names:
            obs_list = slot_data.get(name)
            prior = priors.get(name)

            if not obs_list:
                # Untried atom: use benchmark prior for ordering if available
                if prior is not None:
                    # Large finite score so priors create a total order
                    score = 1e6 + prior
                else:
                    score = float("inf")
                # Apply supersession penalty
                if statuses.get(name) == "superseded":
                    if math.isinf(score):
                        score = 1e6 - 1.0
                    else:
                        score = score * 0.5
                scored.append((name, score))
                continue

            n_plays = len(obs_list)
            mean_reward = sum(
                1.0 - min(1.0, max(0.0, obs.gradient_score / 100.0))
                for obs in obs_list
            ) / n_plays

            # Mix in benchmark prior if available
            if prior is not None:
                mean_reward = (
                    prior_strength * prior + n_plays * mean_reward
                ) / (prior_strength + n_plays)

            ucb = self._ucb1(mean_reward, n_plays, total_plays, exploration_weight)

            # Apply supersession penalty
            if statuses.get(name) == "superseded":
                ucb *= 0.5

            scored.append((name, ucb))

        scored.sort(key=lambda x: -x[1])
        return scored

    def total_observations_for_slot(self, slot: SlotSignature) -> int:
        """Total observation count across all atoms in a slot."""
        slot_data = self._observations.get(slot.key, {})
        return sum(len(obs) for obs in slot_data.values())

    def observation_count(self, slot: SlotSignature, atom_name: str) -> int:
        """Observation count for a specific atom in a slot."""
        return len(self._observations.get(slot.key, {}).get(atom_name, []))

    @staticmethod
    def _ucb1(mean_reward: float, n_plays: int, total_plays: int, c: float) -> float:
        if n_plays == 0 or total_plays == 0:
            return float("inf")
        return mean_reward + c * math.sqrt(math.log(total_plays) / n_plays)


def compute_slot_signature(
    node: AlgorithmicNode,
    parent: AlgorithmicNode | None,
) -> SlotSignature:
    """Compute the structural slot signature for a CDG node."""
    parent_name = ""
    if parent is not None:
        parent_name = parent.name.strip().lower().replace(" ", "_")

    return SlotSignature(
        parent_name=parent_name,
        concept_type=node.concept_type.value if hasattr(node.concept_type, "value") else str(node.concept_type),
        input_types=tuple(sorted(io.type_desc for io in node.inputs)),
        output_types=tuple(sorted(io.type_desc for io in node.outputs)),
    )
