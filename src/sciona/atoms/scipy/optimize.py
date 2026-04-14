from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple, Union

import icontract
import numpy as np
import scipy.optimize

from sciona.ghost.registry import register_atom

from sciona.probes.scipy.witnesses import (
    witness_scipy_curve_fit,
    witness_scipy_differential_evolution,
    witness_scipy_linprog,
    witness_scipy_minimize,
    witness_scipy_root,
    witness_scipy_shgo,
)

ArrayLike = Union[np.ndarray, list, tuple]
CurveFitKwarg = float | int | bool | str | np.ndarray | Sequence[float] | tuple[float, float] | None


@register_atom(witness_scipy_minimize)
@icontract.require(lambda x0: np.asarray(x0).ndim >= 1, "Initial guess x0 must be at least 1D")
@icontract.require(lambda fun, x0: fun is not None and x0 is not None, "Objective function and initial guess must not be None")
@icontract.ensure(lambda result: result is not None, "Optimization result must not be None")
def minimize(
    fun: Callable,
    x0: ArrayLike,
    args: tuple = (),
    method: str | None = None,
    jac: Callable | str | bool | None = None,
    hess: Callable | str | None = None,
    hessp: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Dict | Sequence[Dict] = (),
    tol: float | None = None,
    callback: Callable | None = None,
    options: Dict | None = None,
) -> scipy.optimize.OptimizeResult:
    return scipy.optimize.minimize(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        callback=callback,
        options=options,
    )


@register_atom(witness_scipy_root)
@icontract.require(lambda fun, x0: fun is not None and x0 is not None, "Function and initial guess must not be None")
@icontract.ensure(lambda result: result is not None, "Root finding result must not be None")
def root(
    fun: Callable,
    x0: ArrayLike,
    args: tuple = (),
    method: str = "hybr",
    jac: Callable | bool | None = None,
    tol: float | None = None,
    callback: Callable | None = None,
    options: Dict | None = None,
) -> scipy.optimize.OptimizeResult:
    return scipy.optimize.root(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        tol=tol,
        callback=callback,
        options=options,
    )


@register_atom(witness_scipy_linprog)
@icontract.require(lambda c: np.asarray(c).ndim >= 1, "Objective coefficients c must be at least 1D")
@icontract.require(lambda c: c is not None, "Coefficients of the linear objective function must not be None")
@icontract.ensure(lambda result: result is not None, "Linear programming result must not be None")
def linprog(
    c: ArrayLike,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Sequence | None = None,
    method: str = "highs",
    callback: Callable | None = None,
    options: Dict | None = None,
    x0: ArrayLike | None = None,
) -> scipy.optimize.OptimizeResult:
    return scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        callback=callback,
        options=options,
        x0=x0,
    )


@register_atom(witness_scipy_curve_fit)
@icontract.require(lambda f, xdata, ydata: len(xdata) == len(ydata), "xdata and ydata must have the same length")
@icontract.ensure(lambda result: len(result) == 2, "Result must be a tuple of (popt, pcov)")
def curve_fit(
    f: Callable[..., np.ndarray | float],
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    check_finite: bool | None = None,
    bounds: Sequence | None = (-np.inf, np.inf),
    method: str | None = None,
    jac: Callable[..., np.ndarray] | str | None = None,
    **kwargs: CurveFitKwarg,
) -> Tuple[np.ndarray, np.ndarray]:
    return scipy.optimize.curve_fit(
        f,
        xdata,
        ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=bounds,
        method=method,
        jac=jac,
        **kwargs,
    )


@register_atom(witness_scipy_shgo)
@icontract.require(lambda func: func is not None, "func cannot be None")
@icontract.require(lambda bounds: bounds is not None, "bounds cannot be None")
@icontract.require(lambda args: args is not None, "args cannot be None")
@icontract.require(lambda constraints: constraints is not None, "constraints cannot be None")
@icontract.require(lambda n: n is not None, "n cannot be None")
@icontract.require(lambda iters: iters is not None, "iters cannot be None")
@icontract.require(lambda minimizer_kwargs: minimizer_kwargs is not None, "minimizer_kwargs cannot be None")
@icontract.require(lambda options: options is not None, "options cannot be None")
@icontract.require(lambda sampling_method: sampling_method is not None, "sampling_method cannot be None")
@icontract.ensure(lambda result: result is not None, "Shgo output must not be None")
def shgo(
    func: Callable[..., float],
    bounds: list[tuple[float, float]],
    args: tuple = (),
    constraints: list[dict] | dict = (),
    n: int = 100,
    iters: int = 1,
    callback: Callable | None = None,
    minimizer_kwargs: dict | None = None,
    options: dict | None = None,
    sampling_method: str | Callable = "simplicial",
) -> scipy.optimize.OptimizeResult:
    return scipy.optimize.shgo(
        func,
        bounds,
        args=args,
        constraints=constraints,
        n=n,
        iters=iters,
        callback=callback,
        minimizer_kwargs=minimizer_kwargs,
        options=options,
        sampling_method=sampling_method,
    )


@register_atom(witness_scipy_differential_evolution)
@icontract.require(lambda mutation: isinstance(mutation, (float, int, np.number, tuple)), "mutation must be numeric or tuple")
@icontract.require(lambda recombination: isinstance(recombination, (float, int, np.number)), "recombination must be numeric")
@icontract.require(lambda atol: isinstance(atol, (float, int, np.number)), "atol must be numeric")
@icontract.ensure(lambda result: result is not None, "Differential evolution output must not be None")
def differential_evolution(
    func: Callable[..., float],
    bounds: list[tuple[float, float]],
    args: tuple = (),
    strategy: str = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    rng: int | np.random.RandomState | np.random.Generator | None = None,
    callback: Callable | None = None,
    disp: bool = False,
    polish: bool = True,
    init: str | np.ndarray = "latinhypercube",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int | Callable = 1,
    constraints: list[dict] | dict | tuple | None = (),
    x0: np.ndarray | None = None,
    integrality: np.ndarray | None = None,
    vectorized: bool = False,
) -> scipy.optimize.OptimizeResult:
    return scipy.optimize.differential_evolution(
        func,
        bounds,
        args=args,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        rng=rng,
        callback=callback,
        disp=disp,
        polish=polish,
        init=init,
        atol=atol,
        updating=updating,
        workers=workers,
        constraints=constraints,
        x0=x0,
        integrality=integrality,
        vectorized=vectorized,
    )
