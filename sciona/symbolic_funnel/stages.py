"""Pure-function implementations of the five heuristic funnel stages.

Each stage takes a dataset, a list of candidates, and a config, and returns
the filtered candidates with stage verdicts attached.  No side effects,
no LLM calls, no database writes.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sciona.symbolic_funnel.contracts import (
    FunnelCandidate,
    FunnelConfig,
    FunnelResult,
    StageVerdict,
)
from sciona.symbolic_funnel.dataset import EmpiricalDataset
from sciona.symbolic_funnel.index import FunnelAtomEntry, FunnelIndex


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_funnel(
    dataset: EmpiricalDataset,
    index: FunnelIndex,
    config: FunnelConfig | None = None,
) -> FunnelResult:
    """Run the complete heuristic funnel cascade on *dataset*.

    Returns a :class:`FunnelResult` with candidates ranked by aggregate score.
    """
    cfg = config or FunnelConfig()
    timing: dict[str, float] = {}
    stages_executed: list[str] = []

    # Start with equivalence class representatives only.
    candidates = [FunnelCandidate(entry=e) for e in index.representatives]
    total_considered = len(candidates)

    # Stage 1: Boundary Triage
    t0 = time.perf_counter()
    candidates = stage_boundary_triage(dataset, candidates, cfg)
    timing["boundary_triage"] = time.perf_counter() - t0
    stages_executed.append("boundary_triage")

    # Stage 2: Exponent Extraction (adds new candidates from index lookup)
    t0 = time.perf_counter()
    candidates = stage_exponent_extraction(dataset, candidates, index, cfg)
    timing["exponent_extraction"] = time.perf_counter() - t0
    stages_executed.append("exponent_extraction")

    # Stage 3: Invariant Variance
    t0 = time.perf_counter()
    candidates = stage_invariant_variance(dataset, candidates, cfg)
    timing["invariant_variance"] = time.perf_counter() - t0
    stages_executed.append("invariant_variance")

    # Stage 5: RANSAC for remaining non-linearizable candidates
    t0 = time.perf_counter()
    candidates = stage_ransac(dataset, candidates, cfg)
    timing["ransac"] = time.perf_counter() - t0
    stages_executed.append("ransac")

    # Rank by aggregate score descending.
    candidates.sort(key=lambda c: c.aggregate_score, reverse=True)

    return FunnelResult(
        ranked_candidates=candidates,
        stages_executed=stages_executed,
        timing=timing,
        equivalence_classes_tested=len(index.representatives),
        total_candidates_considered=total_considered,
    )


# ---------------------------------------------------------------------------
# Stage 1: Boundary Triage
# ---------------------------------------------------------------------------


def stage_boundary_triage(
    dataset: EmpiricalDataset,
    candidates: list[FunnelCandidate],
    config: FunnelConfig,
) -> list[FunnelCandidate]:
    """O(1) per candidate: reject based on variable count, bounds, dimensions."""
    dataset_names = set(dataset.column_names)
    passed: list[FunnelCandidate] = []

    for candidate in candidates:
        entry = candidate.entry
        # Check that all input + output variables have matching dataset columns.
        required = {
            name
            for name, role in entry.variables.items()
            if role in ("input", "output")
        }
        missing = required - dataset_names
        if missing:
            candidate.add_verdict(
                StageVerdict(
                    stage_name="boundary_triage",
                    passed=False,
                    evidence={"missing_columns": sorted(missing)},
                )
            )
            continue

        # Check validity bounds against dataset column ranges.
        bounds_ok = True
        for symbol, (lower, upper) in entry.validity_bounds.items():
            if symbol not in dataset_names:
                continue
            col_min = dataset.column_min(symbol)
            col_max = dataset.column_max(symbol)
            if lower is not None and col_max < lower:
                bounds_ok = False
                break
            if upper is not None and col_min > upper:
                bounds_ok = False
                break
        if not bounds_ok:
            candidate.add_verdict(
                StageVerdict(
                    stage_name="boundary_triage",
                    passed=False,
                    evidence={"reason": "validity_bounds_violated"},
                )
            )
            continue

        # Check dimensional compatibility if both sides have signatures.
        dim_ok = _check_dimensional_compatibility(dataset, entry)
        if not dim_ok:
            candidate.add_verdict(
                StageVerdict(
                    stage_name="boundary_triage",
                    passed=False,
                    evidence={"reason": "dimensional_mismatch"},
                )
            )
            continue

        candidate.add_verdict(
            StageVerdict(stage_name="boundary_triage", passed=True, score=1.0)
        )
        passed.append(candidate)

    return passed


def _check_dimensional_compatibility(
    dataset: EmpiricalDataset,
    entry: FunnelAtomEntry,
) -> bool:
    """Check if dataset column dimensions are compatible with expression."""
    from sciona.ghost.dimensions import DimensionalSignature

    for col in dataset.columns:
        if col.dim_signature is None:
            continue
        sym_dim_str = entry.dim_signature.get(col.name)
        if sym_dim_str is None:
            continue
        try:
            expr_dim = DimensionalSignature.from_compact(sym_dim_str)
        except Exception:  # noqa: BLE001
            continue
        if not col.dim_signature.is_compatible(expr_dim):
            return False
    return True


# ---------------------------------------------------------------------------
# Stage 2: Exponent Extraction (Log-Space SVD)
# ---------------------------------------------------------------------------


def stage_exponent_extraction(
    dataset: EmpiricalDataset,
    candidates: list[FunnelCandidate],
    index: FunnelIndex,
    config: FunnelConfig,
) -> list[FunnelCandidate]:
    """O(N) SVD on log-transformed data, then O(1) index lookup.

    Extracts the power-law exponent fingerprint from the data and matches
    it against the index.  Candidates that match get a score boost;
    non-power-law candidates pass through unfiltered.
    """
    log_dataset, log_names = dataset.log_transform()
    if log_dataset.n_cols < 2:
        # Not enough positive columns for SVD.
        return candidates

    exponents = _extract_exponents_svd(log_dataset, config)
    if exponents is None:
        return candidates

    # Build a signature dict mapping column name -> exponent string.
    extracted_sig = {}
    for name, exp in zip(log_names, exponents):
        frac = Fraction(exp).limit_denominator(config.max_exponent_denominator)
        if abs(float(frac) - exp) <= config.exponent_snap_tolerance:
            extracted_sig[name] = str(frac)

    if not extracted_sig:
        return candidates

    # Look up matching entries in the index by exponent signature.
    # We do a structural match: for each indexed signature, check if the
    # extracted exponents are compatible (same relative ratios).
    matched_entries: set[str] = set()  # expression_ids
    for candidate in candidates:
        entry = candidate.entry
        if entry.exponent_signature is None:
            continue
        if _exponent_signatures_compatible(extracted_sig, entry.exponent_signature):
            matched_entries.add(entry.expression_id)
            candidate.add_verdict(
                StageVerdict(
                    stage_name="exponent_extraction",
                    passed=True,
                    score=config.exponent_match_confidence,
                    evidence={
                        "extracted_exponents": extracted_sig,
                        "matched_signature": entry.exponent_signature,
                    },
                )
            )

    # Candidates that weren't matched by exponents pass through.
    result: list[FunnelCandidate] = []
    for candidate in candidates:
        if candidate.entry.expression_id in matched_entries:
            result.append(candidate)
        elif candidate.entry.exponent_signature is None:
            # Non-power-law: pass through to later stages.
            result.append(candidate)
        # Power-law entries that didn't match are filtered out.

    return result


def _extract_exponents_svd(
    log_dataset: EmpiricalDataset,
    config: FunnelConfig,
) -> NDArray[np.floating[Any]] | None:
    """Extract power-law exponents via SVD of the log-transformed data.

    The null-space of the log-covariance matrix reveals linear relationships
    in log-space, which correspond to power-law exponents.
    """
    data = log_dataset.data
    if data.shape[0] < data.shape[1] + 1:
        return None

    # Center the data.
    centered = data - data.mean(axis=0)
    # SVD of the centered data matrix.
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=True)
    except np.linalg.LinAlgError:
        return None

    # The null-space vector(s) correspond to near-zero singular values.
    # For a perfect power law with D variables, there should be D-1
    # significant singular values and 1 near-zero one.
    if len(s) < 2:
        return None

    ratio = s[-1] / s[0] if s[0] > 0 else 1.0
    if ratio > 0.1:
        # No clear null-space — data doesn't follow a single power law.
        return None

    # The last row of V^T is the null-space vector.
    null_vec = vt[-1]

    # Normalize so the largest absolute exponent is the reference.
    max_abs = np.max(np.abs(null_vec))
    if max_abs < 1e-12:
        return None
    normalized = null_vec / max_abs

    return normalized


def _exponent_signatures_compatible(
    extracted: dict[str, str],
    indexed: dict[str, str],
) -> bool:
    """Check if extracted exponents match the indexed signature.

    The match is based on relative ratios between shared variables.
    """
    shared = set(extracted) & set(indexed)
    if len(shared) < 2:
        return False

    # Check that the ratios between exponents are consistent.
    shared_list = sorted(shared)
    ref_name = shared_list[0]
    ext_ref = Fraction(extracted[ref_name])
    idx_ref = Fraction(indexed[ref_name])
    if ext_ref == 0 or idx_ref == 0:
        return ext_ref == idx_ref

    for name in shared_list[1:]:
        ext_val = Fraction(extracted[name])
        idx_val = Fraction(indexed[name])
        # Check: ext_val / ext_ref == idx_val / idx_ref
        if ext_val * idx_ref != idx_val * ext_ref:
            return False
    return True


# ---------------------------------------------------------------------------
# Stage 3: Invariant Variance
# ---------------------------------------------------------------------------


def stage_invariant_variance(
    dataset: EmpiricalDataset,
    candidates: list[FunnelCandidate],
    config: FunnelConfig,
) -> list[FunnelCandidate]:
    """O(N) per candidate: evaluate invariant expression, check CV.

    For each candidate with pre-computed invariant forms, compile the
    invariant expression to NumPy and evaluate vectorized over the dataset.
    If the Coefficient of Variation (CV = std/|mean|) is below the threshold,
    the law holds and the mean is the fitted constant.
    """
    if dataset.n_rows < config.min_rows_for_cv:
        return candidates

    result: list[FunnelCandidate] = []
    for candidate in candidates:
        entry = candidate.entry
        if not entry.invariant_forms:
            # No invariant forms — pass through to RANSAC.
            result.append(candidate)
            continue

        best_cv = float("inf")
        best_form: dict[str, Any] | None = None
        best_values: NDArray[np.floating[Any]] | None = None

        for form in entry.invariant_forms:
            try:
                values = _evaluate_invariant(dataset, form, entry)
                if values is None:
                    continue
                finite = values[np.isfinite(values)]
                if len(finite) < config.min_rows_for_cv:
                    continue
                mean = float(np.mean(finite))
                if abs(mean) < 1e-30:
                    continue
                cv = float(np.std(finite) / abs(mean))
                if cv < best_cv:
                    best_cv = cv
                    best_form = form
                    best_values = finite
            except Exception:  # noqa: BLE001
                continue

        if best_form is not None and best_cv < config.cv_threshold:
            mean_val = float(np.mean(best_values))  # type: ignore[arg-type]
            score = max(0.0, 1.0 - best_cv / config.cv_threshold)
            score = score * config.cv_high_confidence
            candidate.add_verdict(
                StageVerdict(
                    stage_name="invariant_variance",
                    passed=True,
                    score=score,
                    evidence={
                        "cv": best_cv,
                        "fitted_constant_name": best_form["isolated_symbol"],
                        "fitted_constant_value": mean_val,
                        "known_value": best_form.get("known_value"),
                        "n_finite": len(best_values),  # type: ignore[arg-type]
                    },
                )
            )
            candidate.fitted_constants[best_form["isolated_symbol"]] = mean_val
            result.append(candidate)
        elif best_form is not None:
            # CV too high — filter out.
            candidate.add_verdict(
                StageVerdict(
                    stage_name="invariant_variance",
                    passed=False,
                    evidence={"cv": best_cv, "threshold": config.cv_threshold},
                )
            )
        else:
            # No invariant form could be evaluated — pass to RANSAC.
            result.append(candidate)

    return result


def _evaluate_invariant(
    dataset: EmpiricalDataset,
    form: dict[str, Any],
    entry: FunnelAtomEntry,
) -> NDArray[np.floating[Any]] | None:
    """Compile and evaluate an invariant expression over the dataset."""
    import sympy as sp

    from sciona.ghost.symbolic import deserialize_expr

    inv_expr = deserialize_expr(form["invariant_expr_srepr"])
    free_symbols = sorted(inv_expr.free_symbols, key=lambda s: str(s))
    symbol_names = [str(s) for s in free_symbols]

    # Check that all required symbols are available as dataset columns.
    available = set(dataset.column_names)
    if not all(name in available for name in symbol_names):
        return None

    fn = sp.lambdify(free_symbols, inv_expr, "numpy")
    args = [dataset.column_by_name(name) for name in symbol_names]
    return fn(*args)


# ---------------------------------------------------------------------------
# Stage 5: Multi-Fidelity RANSAC
# ---------------------------------------------------------------------------


def stage_ransac(
    dataset: EmpiricalDataset,
    candidates: list[FunnelCandidate],
    config: FunnelConfig,
) -> list[FunnelCandidate]:
    """RANSAC-style minimal point solver for non-linearizable expressions.

    For candidates that didn't go through invariant variance (no invariant
    forms), sample K+1 points, solve for K unknowns, and evaluate on holdout.
    """
    result: list[FunnelCandidate] = []
    for candidate in candidates:
        # Skip candidates that already have a high-confidence verdict.
        if candidate.aggregate_score >= config.cv_high_confidence * 0.8:
            result.append(candidate)
            continue

        entry = candidate.entry
        # Try RANSAC if the expression has constants to fit.
        if not entry.constants and not entry.constant_variables:
            # Fully data-backed: evaluate residual directly.
            residual = _evaluate_direct_residual(dataset, entry)
            if residual is not None and residual < config.ransac_residual_threshold:
                candidate.add_verdict(
                    StageVerdict(
                        stage_name="ransac",
                        passed=True,
                        score=config.ransac_confidence * (1.0 - residual),
                        evidence={"residual": residual, "method": "direct"},
                    )
                )
                result.append(candidate)
            continue

        # RANSAC with minimal point solvers.
        ransac_result = _ransac_fit(dataset, entry, config)
        if ransac_result is not None:
            residual, fitted = ransac_result
            if residual < config.ransac_residual_threshold:
                candidate.add_verdict(
                    StageVerdict(
                        stage_name="ransac",
                        passed=True,
                        score=config.ransac_confidence * (1.0 - residual),
                        evidence={
                            "residual": residual,
                            "fitted_constants": fitted,
                            "method": "ransac",
                        },
                    )
                )
                candidate.fitted_constants.update(fitted)
                result.append(candidate)

    return result


def _evaluate_direct_residual(
    dataset: EmpiricalDataset,
    entry: FunnelAtomEntry,
) -> float | None:
    """Evaluate residual for a fully data-backed expression (no constants)."""
    import sympy as sp

    from sciona.ghost.symbolic import deserialize_expr

    try:
        expr = deserialize_expr(entry.srepr_str)
        if not isinstance(expr, sp.Equality):
            return None

        lhs_symbols = sorted(expr.lhs.free_symbols, key=lambda s: str(s))
        rhs_symbols = sorted(expr.rhs.free_symbols, key=lambda s: str(s))
        all_symbols = sorted(expr.free_symbols, key=lambda s: str(s))
        symbol_names = [str(s) for s in all_symbols]

        if not all(name in dataset.column_names for name in symbol_names):
            return None

        lhs_fn = sp.lambdify(all_symbols, expr.lhs, "numpy")
        rhs_fn = sp.lambdify(all_symbols, expr.rhs, "numpy")
        args = [dataset.column_by_name(str(s)) for s in all_symbols]

        lhs_vals = lhs_fn(*args)
        rhs_vals = rhs_fn(*args)

        finite_mask = np.isfinite(lhs_vals) & np.isfinite(rhs_vals)
        if np.sum(finite_mask) < 10:
            return None

        residuals = np.abs(lhs_vals[finite_mask] - rhs_vals[finite_mask])
        scale = np.maximum(
            np.abs(lhs_vals[finite_mask]), np.abs(rhs_vals[finite_mask])
        )
        scale = np.where(scale < 1e-30, 1.0, scale)
        return float(np.median(residuals / scale))
    except Exception:  # noqa: BLE001
        return None


def _ransac_fit(
    dataset: EmpiricalDataset,
    entry: FunnelAtomEntry,
    config: FunnelConfig,
) -> tuple[float, dict[str, float]] | None:
    """RANSAC-style fit: sample K+1 points, solve, evaluate holdout."""
    import sympy as sp

    from sciona.ghost.symbolic import deserialize_expr

    try:
        expr = deserialize_expr(entry.srepr_str)
        if not isinstance(expr, sp.Equality):
            return None

        const_syms = [
            s for s in expr.free_symbols if entry.variables.get(str(s)) == "constant"
        ]
        if not const_syms:
            return None

        all_symbols = sorted(expr.free_symbols, key=lambda s: str(s))
        data_syms = [s for s in all_symbols if str(s) not in {str(c) for c in const_syms}]

        if not all(str(s) in dataset.column_names for s in data_syms):
            return None

        K = len(const_syms)
        n_sample = K + 1
        n_rows = dataset.n_rows

        if n_rows < n_sample + config.ransac_holdout_size:
            return None

        rng = np.random.default_rng(42)
        best_residual = float("inf")
        best_fitted: dict[str, float] = {}

        residual_expr = expr.lhs - expr.rhs

        for _ in range(config.ransac_iterations):
            # Sample K+1 random rows.
            sample_idx = rng.choice(n_rows, size=n_sample, replace=False)
            holdout_idx = rng.choice(
                np.setdiff1d(np.arange(n_rows), sample_idx),
                size=min(config.ransac_holdout_size, n_rows - n_sample),
                replace=False,
            )

            # Build K+1 equations by substituting data values.
            equations = []
            for idx in sample_idx:
                subs = {
                    s: float(dataset.column_by_name(str(s))[idx]) for s in data_syms
                }
                equations.append(residual_expr.subs(subs))

            # Solve for the constant symbols.
            try:
                solutions = sp.solve(equations[:K], const_syms, dict=True)
            except Exception:  # noqa: BLE001
                continue
            if not solutions:
                continue

            sol = solutions[0]
            fitted = {}
            valid = True
            for cs in const_syms:
                val = sol.get(cs)
                if val is None or not val.is_number:
                    valid = False
                    break
                fitted[str(cs)] = float(val)
            if not valid:
                continue

            # Evaluate residual on holdout.
            full_subs = {s: fitted[str(s)] for s in const_syms}
            eval_expr = residual_expr.subs(full_subs)
            eval_fn = sp.lambdify(
                sorted(eval_expr.free_symbols, key=lambda s: str(s)),
                eval_expr,
                "numpy",
            )
            eval_symbols = sorted(eval_expr.free_symbols, key=lambda s: str(s))
            holdout_args = [
                dataset.column_by_name(str(s))[holdout_idx] for s in eval_symbols
            ]

            try:
                residuals = eval_fn(*holdout_args)
            except Exception:  # noqa: BLE001
                continue

            finite = residuals[np.isfinite(residuals)]
            if len(finite) < 5:
                continue

            # Normalized median absolute residual.
            median_res = float(np.median(np.abs(finite)))
            # Normalize by scale of the output variable.
            output_names = [
                n for n, r in entry.variables.items() if r == "output"
            ]
            if output_names and output_names[0] in dataset.column_names:
                scale = float(
                    np.median(
                        np.abs(dataset.column_by_name(output_names[0])[holdout_idx])
                    )
                )
                if scale > 1e-30:
                    median_res /= scale

            if median_res < best_residual:
                best_residual = median_res
                best_fitted = fitted

        if best_residual < float("inf"):
            return best_residual, best_fitted
        return None
    except Exception:  # noqa: BLE001
        return None
