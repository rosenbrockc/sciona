"""Runtime atoms for MCMC/HMC expansion rules.

Provides deterministic, pure functions for HMC sampler diagnostics
and adaptive corrections:

  - Divergent transition detection (energy conservation test)
  - Step size adaptation (Nesterov dual averaging)
  - Mass matrix estimation (diagonal or dense covariance)
  - Convergence diagnostics (split R-hat and bulk ESS)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Divergent transition detection
# ---------------------------------------------------------------------------


def detect_divergent_transitions(
    energies_initial: np.ndarray,
    energies_proposed: np.ndarray,
    threshold: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect divergent transitions in HMC via energy conservation violation.

    A divergent transition occurs when |H_proposed - H_initial| exceeds
    *threshold*, indicating the leapfrog integrator has failed (typically
    because the step size is too large for the local curvature).

    Returns:
        (energy_errors, divergence_mask) where energy_errors[k] = |delta_H|
        and divergence_mask[k] is True if the transition is divergent.
    """
    energies_initial = np.asarray(energies_initial, dtype=np.float64)
    energies_proposed = np.asarray(energies_proposed, dtype=np.float64)

    energy_errors = np.abs(energies_proposed - energies_initial)
    divergence_mask = energy_errors > threshold
    return energy_errors, divergence_mask


# ---------------------------------------------------------------------------
# Step size adaptation (dual averaging)
# ---------------------------------------------------------------------------


def compute_dual_averaging_step_size(
    accept_probs: np.ndarray,
    target_accept: float = 0.65,
    epsilon_0: float = 1.0,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> float:
    """Adapt HMC step size via Nesterov dual averaging (Hoffman & Gelman 2014).

    Solves for epsilon such that the mean acceptance probability converges
    to *target_accept*.  The update rule is:

        H_bar_{m+1} = (1 - 1/(m + t0)) * H_bar_m + (target - alpha_m) / (m + t0)
        log(epsilon_{m+1}) = mu - sqrt(m) / gamma * H_bar_{m+1}
        log(epsilon_bar_{m+1}) = m^{-kappa} * log(epsilon_m) + (1 - m^{-kappa}) * log(epsilon_bar_m)

    Returns:
        adapted_epsilon — the dual-averaged step size.
    """
    accept_probs = np.asarray(accept_probs, dtype=np.float64)
    n = len(accept_probs)
    if n == 0:
        return epsilon_0

    mu = np.log(10.0 * epsilon_0)
    log_epsilon_bar = 0.0
    h_bar = 0.0
    log_epsilon = np.log(epsilon_0)

    for m in range(1, n + 1):
        alpha = accept_probs[m - 1]
        w = 1.0 / (m + t0)
        h_bar = (1.0 - w) * h_bar + w * (target_accept - alpha)
        log_epsilon = mu - (np.sqrt(m) / gamma) * h_bar
        mk = m ** (-kappa)
        log_epsilon_bar = mk * log_epsilon + (1.0 - mk) * log_epsilon_bar

    return float(np.exp(log_epsilon_bar))


# ---------------------------------------------------------------------------
# Mass matrix estimation
# ---------------------------------------------------------------------------


def estimate_mass_matrix(
    samples: np.ndarray,
    diagonal_only: bool = True,
) -> np.ndarray:
    """Estimate mass matrix M from warmup samples.

    When *diagonal_only* is True, returns diag(var(samples)) — one variance
    per parameter.  When False, returns the full sample covariance matrix.

    Regularization: adds 1e-3 * I to prevent singular matrices when
    variance is near zero.

    Returns:
        M_estimated — shape (d,) if diagonal_only else (d, d).
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape
    if n < 2:
        if diagonal_only:
            return np.ones(d)
        return np.eye(d)

    if diagonal_only:
        variances = np.var(samples, axis=0, ddof=1)
        # Regularize near-zero variances
        variances = np.maximum(variances, 1e-3)
        return variances
    else:
        cov = np.cov(samples, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        # Regularize
        cov += 1e-3 * np.eye(d)
        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)
        return cov


# ---------------------------------------------------------------------------
# Convergence diagnostics (R-hat and ESS)
# ---------------------------------------------------------------------------


def compute_convergence_diagnostics(
    chains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute split R-hat and bulk ESS across chains.

    Uses rank-normalized split-R-hat (Vehtari et al. 2021) and bulk ESS
    via autocorrelation with Geyer's initial monotone sequence estimator.

    Args:
        chains: shape (n_chains, n_samples, n_params)

    Returns:
        (rhat_per_param, ess_per_param) — both shape (n_params,)
    """
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim == 2:
        # Single chain: split it
        chains = chains[np.newaxis, :, :]
    if chains.ndim == 1:
        chains = chains.reshape(1, -1, 1)

    n_chains, n_samples, n_params = chains.shape

    # Split chains in half for split-R-hat
    half = n_samples // 2
    if half < 2:
        return np.ones(n_params), np.full(n_params, float(n_chains * n_samples))

    split_chains = np.concatenate(
        [chains[:, :half, :], chains[:, half : 2 * half, :]],
        axis=0,
    )  # (2*n_chains, half, n_params)
    m = split_chains.shape[0]  # number of split chains
    n = half  # samples per split chain

    # Rank-normalize across all split chains
    ranked = np.empty_like(split_chains)
    for p in range(n_params):
        all_draws = split_chains[:, :, p].ravel()
        order = np.argsort(np.argsort(all_draws))
        # Transform ranks to normal scores
        ranks = (order + 0.5) / len(order)
        # Inverse normal CDF approximation (rational approximation)
        z = _inv_normal_cdf(ranks)
        ranked[:, :, p] = z.reshape(m, n)

    rhat = np.empty(n_params)
    ess = np.empty(n_params)

    for p in range(n_params):
        chain_means = ranked[:, :, p].mean(axis=1)  # (m,)
        chain_vars = ranked[:, :, p].var(axis=1, ddof=1)  # (m,)

        grand_mean = chain_means.mean()
        B = n * np.var(chain_means, ddof=1)  # between-chain variance
        W = np.mean(chain_vars)  # within-chain variance

        if W < 1e-15:
            rhat[p] = 1.0
            ess[p] = float(m * n)
            continue

        var_hat = ((n - 1) / n) * W + B / n
        rhat[p] = np.sqrt(var_hat / W)

        # Bulk ESS via autocorrelation
        ess[p] = _compute_ess(ranked[:, :, p])

    return rhat, ess


def _inv_normal_cdf(p: np.ndarray) -> np.ndarray:
    """Approximate inverse normal CDF (probit) using rational approximation.

    Abramowitz & Stegun 26.2.23 — accurate to ~4.5e-4.
    """
    p = np.clip(p, 1e-10, 1.0 - 1e-10)

    # Constants for the rational approximation
    a0, a1, a2 = 2.515517, 0.802853, 0.010328
    b1, b2, b3 = 1.432788, 0.189269, 0.001308

    mask = p < 0.5
    pp = np.where(mask, p, 1.0 - p)
    t = np.sqrt(-2.0 * np.log(pp))
    z = t - (a0 + a1 * t + a2 * t**2) / (1.0 + b1 * t + b2 * t**2 + b3 * t**3)
    return np.where(mask, -z, z)


def _compute_ess(ranked_chains: np.ndarray) -> float:
    """Compute bulk ESS using Geyer's initial monotone sequence estimator.

    Args:
        ranked_chains: shape (m, n) — rank-normalized draws per split chain
    """
    m, n = ranked_chains.shape
    if n < 4:
        return float(m * n)

    # Compute per-chain autocorrelation and average
    max_lag = n - 1
    rho_hat = np.zeros(max_lag)

    for lag in range(max_lag):
        acf_sum = 0.0
        for chain in ranked_chains:
            centered = chain - chain.mean()
            var = np.var(centered)
            if var < 1e-15:
                continue
            if lag == 0:
                acf_sum += 1.0
            else:
                acf_sum += float(np.mean(centered[:-lag] * centered[lag:])) / var
        rho_hat[lag] = acf_sum / m

    # Geyer's initial monotone sequence: sum consecutive pairs,
    # stop when pair sum becomes negative
    tau = -1.0 + 2.0 * rho_hat[0]
    lag = 1
    while lag < max_lag - 1:
        pair_sum = rho_hat[lag] + rho_hat[lag + 1]
        if pair_sum < 0:
            break
        tau += 2.0 * pair_sum
        lag += 2

    tau = max(tau, 1.0 / (m * n))  # floor to prevent negative ESS
    return float(m * n / tau)
