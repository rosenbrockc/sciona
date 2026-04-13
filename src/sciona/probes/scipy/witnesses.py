from __future__ import annotations

from typing import Any, Callable, Sequence

from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal


def _as_array_or_scalar(
    shape: tuple[int, ...],
    *,
    dtype: str = "float64",
    min_val: float | None = None,
    max_val: float | None = None,
) -> AbstractArray | AbstractScalar:
    if shape == ():
        return AbstractScalar(dtype=dtype, min_val=min_val, max_val=max_val)
    return AbstractArray(shape=shape, dtype=dtype, min_val=min_val, max_val=max_val)


def _shape_without_axis(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    if not shape:
        return ()
    ndim = len(shape)
    ax = axis if axis >= 0 else ndim + axis
    if ax < 0 or ax >= ndim:
        raise ValueError(f"axis {axis} out of bounds for shape {shape}")
    return shape[:ax] + shape[ax + 1 :]


def _leading_len(x: AbstractArray) -> int:
    return x.shape[0] if x.shape else 1


def witness_scipy_linalg_solve(
    a: AbstractArray,
    b: AbstractArray,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str | None = None,
    transposed: bool = False,
) -> AbstractArray:
    _ = (lower, overwrite_a, overwrite_b, check_finite, assume_a, transposed)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    if not b.shape or b.shape[0] != a.shape[0]:
        raise ValueError(f"Incompatible shapes for solve: a={a.shape}, b={b.shape}")
    return AbstractArray(shape=b.shape, dtype=a.dtype)


def witness_scipy_linalg_inv(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> AbstractArray:
    _ = (overwrite_a, check_finite)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    return AbstractArray(shape=a.shape, dtype=a.dtype)


def witness_scipy_linalg_det(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> AbstractArray | AbstractScalar:
    _ = (overwrite_a, check_finite)
    if len(a.shape) < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"a must be at least 2D with square trailing dims, got {a.shape}")
    return _as_array_or_scalar(a.shape[:-2], dtype="float64")


def witness_scipy_lu_factor(
    a: AbstractArray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    _ = (overwrite_a, check_finite)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square 2D, got {a.shape}")
    n = a.shape[0]
    return (
        AbstractArray(shape=a.shape, dtype=a.dtype),
        AbstractArray(shape=(n,), dtype="int64", min_val=0.0, max_val=float(max(n - 1, 0))),
    )


def witness_scipy_lu_solve(
    lu_and_piv: tuple[AbstractArray, AbstractArray],
    b: AbstractArray,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> AbstractArray:
    _ = (trans, overwrite_b, check_finite)
    lu, piv = lu_and_piv
    if len(lu.shape) != 2 or lu.shape[0] != lu.shape[1]:
        raise ValueError(f"lu must be square 2D, got {lu.shape}")
    if piv.shape != (lu.shape[0],):
        raise ValueError(f"piv shape must be {(lu.shape[0],)}, got {piv.shape}")
    if not b.shape or b.shape[0] != lu.shape[0]:
        raise ValueError(f"Incompatible shapes for lu_solve: lu={lu.shape}, b={b.shape}")
    return AbstractArray(shape=b.shape, dtype=lu.dtype)


def witness_scipy_minimize(
    fun: Any,
    x0: AbstractArray,
    args: tuple = (),
    method: str | None = None,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Sequence | None = None,
    constraints: Any = (),
    tol: float | None = None,
    callback: Any = None,
    options: dict | None = None,
) -> AbstractArray:
    _ = (fun, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    return AbstractArray(shape=x0.shape, dtype="float64")


def witness_scipy_root(
    fun: Any,
    x0: AbstractArray,
    args: tuple = (),
    method: str = "hybr",
    jac: Any = None,
    tol: float | None = None,
    callback: Any = None,
    options: dict | None = None,
) -> AbstractArray:
    _ = (fun, args, method, jac, tol, callback, options)
    return AbstractArray(shape=x0.shape, dtype="float64")


def witness_scipy_linprog(
    c: AbstractArray,
    A_ub: AbstractArray | None = None,
    b_ub: AbstractArray | None = None,
    A_eq: AbstractArray | None = None,
    b_eq: AbstractArray | None = None,
    bounds: Sequence | None = None,
    method: str = "highs",
    callback: Any = None,
    options: dict | None = None,
    x0: AbstractArray | None = None,
) -> AbstractArray:
    _ = (A_ub, b_ub, A_eq, b_eq, bounds, method, callback, options, x0)
    n_vars = _leading_len(c)
    return AbstractArray(shape=(n_vars,), dtype="float64")


def witness_scipy_curve_fit(
    f: Any,
    xdata: AbstractArray,
    ydata: AbstractArray,
    p0: AbstractArray | None = None,
    sigma: AbstractArray | None = None,
    absolute_sigma: bool = False,
    check_finite: bool | None = None,
    bounds: Sequence | None = (-float("inf"), float("inf")),
    method: str | None = None,
    jac: Any = None,
    **kwargs: Any,
) -> tuple[AbstractArray, AbstractArray]:
    _ = (f, sigma, absolute_sigma, check_finite, bounds, method, jac, kwargs)
    if _leading_len(xdata) != _leading_len(ydata):
        raise ValueError(f"xdata and ydata must have same length, got {xdata.shape} and {ydata.shape}")
    n_params = 1 if p0 is None else _leading_len(p0)
    return (
        AbstractArray(shape=(n_params,), dtype="float64"),
        AbstractArray(shape=(n_params, n_params), dtype="float64"),
    )


def witness_scipy_shgo(
    func: Any,
    bounds: Sequence[tuple[float, float]],
    args: tuple = (),
    constraints: Any = (),
    n: int = 100,
    iters: int = 1,
    callback: Any = None,
    minimizer_kwargs: dict | None = None,
    options: dict | None = None,
    sampling_method: str | Callable[..., Any] = "simplicial",
) -> AbstractArray:
    _ = (func, args, constraints, n, iters, callback, minimizer_kwargs, options, sampling_method)
    return AbstractArray(shape=(len(bounds),), dtype="float64")


def witness_scipy_differential_evolution(
    func: Any,
    bounds: Sequence[tuple[float, float]],
    args: tuple = (),
    strategy: str = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    rng: int | Any | None = None,
    callback: Any = None,
    disp: bool = False,
    polish: bool = True,
    init: str | Any = "latinhypercube",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int | Callable[..., Any] = 1,
    constraints: Any = (),
    x0: AbstractArray | None = None,
    integrality: AbstractArray | None = None,
    vectorized: bool = False,
) -> AbstractArray:
    _ = (
        func,
        args,
        strategy,
        maxiter,
        popsize,
        tol,
        mutation,
        recombination,
        rng,
        callback,
        disp,
        polish,
        init,
        atol,
        updating,
        workers,
        constraints,
        integrality,
        vectorized,
    )
    if x0 is not None and x0.shape:
        return AbstractArray(shape=x0.shape, dtype="float64")
    return AbstractArray(shape=(len(bounds),), dtype="float64")
