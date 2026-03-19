"""Benchmark prior computation for AtomLedger UCB1 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ageom.ecosystem.benchmarks import compute_atom_prior, load_benchmarks_sqlite
from ageom.ecosystem.models import BenchmarkRecord


def load_benchmark_priors(
    manifest_path: Path,
    *,
    direction: str = "minimize",
) -> dict[str, float]:
    """Load benchmark priors from a manifest.sqlite file.

    Returns a mapping of atom FQDN to [0, 1] prior reward.
    """
    benchmarks = load_benchmarks_sqlite(manifest_path)
    return {
        fqdn: compute_atom_prior(records, direction=direction)
        for fqdn, records in benchmarks.items()
    }


def apply_benchmark_prior(
    empirical_mean: float,
    n_plays: int,
    prior_reward: float,
    prior_strength: int = 2,
) -> float:
    """Mix a benchmark prior with empirical observations (Bayesian prior).

    Treats the benchmark as *prior_strength* virtual observations with
    the given prior reward.  The prior washes out after enough real data.

    Parameters
    ----------
    empirical_mean
        Mean reward from real observations.
    n_plays
        Number of real observations.
    prior_reward
        Prior reward from benchmark data (0-1).
    prior_strength
        Number of virtual observations for the prior (default 2).

    Returns
    -------
    float
        Effective mean reward.
    """
    return (prior_strength * prior_reward + n_plays * empirical_mean) / (
        prior_strength + n_plays
    )


def score_untried_with_prior(prior_reward: float) -> float:
    """Compute the UCB1 score for an untried atom with a benchmark prior.

    Untried atoms with priors get a large finite score based on
    ``1e6 + prior_reward``, creating a total ordering among untried
    atoms by benchmark strength while keeping them above all tried atoms
    (whose UCB scores are bounded by ~2-3).
    """
    return 1e6 + prior_reward
