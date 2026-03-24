"""Runtime atoms for Dimensionality Reduction expansion rules.

Provides deterministic, pure functions for dimensionality reduction
pipeline diagnostics:

  - Explained variance analysis (cumulative variance ratio)
  - Crowding detection (neighbor preservation / trustworthiness)
  - Reconstruction error checking (relative reconstruction error)
  - Orthogonality validation (component orthogonality check)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Explained variance analysis
# ---------------------------------------------------------------------------


def analyze_explained_variance(
    eigenvalues: np.ndarray,
) -> tuple[float, bool]:
    """Analyze cumulative explained variance ratio.

    Args:
        eigenvalues: 1D array of eigenvalues (descending order).

    Returns:
        (cumulative_ratio, is_sufficient) where is_sufficient is True
        if cumulative_ratio >= 0.95.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if eigs.size == 0:
        return 0.0, False

    total = float(np.sum(np.abs(eigs)))
    if total == 0.0:
        return 0.0, False

    cumulative_ratio = float(np.sum(np.abs(eigs)) / total)
    return cumulative_ratio, cumulative_ratio >= 0.95


# ---------------------------------------------------------------------------
# Crowding detection
# ---------------------------------------------------------------------------


def detect_crowding(
    neighbor_ranks_original: np.ndarray,
    neighbor_ranks_embedded: np.ndarray,
) -> tuple[float, bool]:
    """Detect crowding by measuring neighbor preservation quality.

    Computes a trustworthiness-like metric based on rank correlation
    between original and embedded space neighbors.

    Args:
        neighbor_ranks_original: 1D array of neighbor ranks in original space.
        neighbor_ranks_embedded: 1D array of neighbor ranks in embedded space.

    Returns:
        (trustworthiness, is_trustworthy) where is_trustworthy is True
        if trustworthiness >= 0.9.
    """
    orig = np.asarray(neighbor_ranks_original, dtype=np.float64).ravel()
    emb = np.asarray(neighbor_ranks_embedded, dtype=np.float64).ravel()

    if orig.size == 0 or emb.size == 0:
        return 1.0, True

    n = min(len(orig), len(emb))
    if n == 0:
        return 1.0, True

    # Spearman-like rank correlation as trustworthiness proxy
    d = orig[:n] - emb[:n]
    max_d_sq = float(n * (n * n - 1) / 3.0) if n > 1 else 1.0
    sum_d_sq = float(np.sum(d ** 2))

    if max_d_sq == 0.0:
        return 1.0, True

    trustworthiness = max(0.0, 1.0 - sum_d_sq / max_d_sq)
    return trustworthiness, trustworthiness >= 0.9


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


def check_reconstruction_error(
    X: np.ndarray,
    X_reconstructed: np.ndarray,
) -> tuple[float, bool]:
    """Check reconstruction quality after dimensionality reduction.

    Computes ||X - X_rec|| / ||X|| and checks it is below threshold.

    Args:
        X: Original data matrix.
        X_reconstructed: Reconstructed data matrix.

    Returns:
        (relative_error, is_acceptable) where is_acceptable is True
        if relative_error <= 0.1.
    """
    X = np.asarray(X, dtype=np.float64)
    X_rec = np.asarray(X_reconstructed, dtype=np.float64)

    if X.size == 0 or X_rec.size == 0:
        return 0.0, True

    if X.shape != X_rec.shape:
        return float("inf"), False

    norm_X = float(np.linalg.norm(X))
    if norm_X == 0.0:
        norm_diff = float(np.linalg.norm(X_rec))
        return norm_diff, norm_diff <= 0.1

    relative_error = float(np.linalg.norm(X - X_rec)) / norm_X
    return relative_error, relative_error <= 0.1


# ---------------------------------------------------------------------------
# Orthogonality validation
# ---------------------------------------------------------------------------


def validate_orthogonality(
    components: np.ndarray,
) -> tuple[float, bool]:
    """Validate orthogonality of projection components.

    Computes C^T @ C and checks that the maximum off-diagonal element
    is below threshold.

    Args:
        components: 2D array (n_components, n_features) of projection components.

    Returns:
        (max_off_diagonal, is_orthogonal) where is_orthogonal is True
        if max_off_diagonal <= 1e-6.
    """
    C = np.asarray(components, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] == 0:
        return 0.0, True

    gram = C @ C.T
    # Zero out diagonal
    np.fill_diagonal(gram, 0.0)
    max_off = float(np.max(np.abs(gram)))
    return max_off, max_off <= 1e-6
