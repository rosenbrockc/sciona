from __future__ import annotations

from typing import Tuple, Union

import icontract
import numpy as np
import scipy.linalg

from sciona.ghost.registry import register_atom

from sciona.probes.scipy.witnesses import (
    witness_scipy_linalg_det,
    witness_scipy_linalg_inv,
    witness_scipy_linalg_solve,
    witness_scipy_lu_factor,
    witness_scipy_lu_solve,
)

ArrayLike = Union[np.ndarray, list, tuple]


def _is_square_2d(a: ArrayLike) -> bool:
    a_arr = np.asarray(a)
    return a_arr.ndim == 2 and a_arr.shape[0] == a_arr.shape[1]


@register_atom(witness_scipy_linalg_solve)
@icontract.require(lambda a, b: np.asarray(a).ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a, b: _is_square_2d(a), "a must be square")
@icontract.require(
    lambda a, b: np.asarray(a).shape[0] == np.asarray(b).shape[0],
    "Dimensions of a and b must match",
)
@icontract.ensure(lambda result, a, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def solve(
    a: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str | None = None,
    transposed: bool = False,
) -> np.ndarray:
    return scipy.linalg.solve(
        a,
        b,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a=assume_a,
        transposed=transposed,
    )


@register_atom(witness_scipy_linalg_inv)
@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result.shape == np.asarray(a).shape, "Inverse has same shape as input")
def inv(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    return scipy.linalg.inv(a, overwrite_a=overwrite_a, check_finite=check_finite)


@register_atom(witness_scipy_linalg_det)
@icontract.require(lambda a: np.asarray(a).ndim >= 2, "a must have at least 2 dimensions")
@icontract.require(
    lambda a: np.asarray(a).shape[-1] == np.asarray(a).shape[-2],
    "Last two dimensions of a must be square",
)
@icontract.ensure(lambda result: result is not None, "Determinant must not be None")
def det(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> float:
    return float(scipy.linalg.det(a, overwrite_a=overwrite_a, check_finite=check_finite))


@register_atom(witness_scipy_lu_factor)
@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result[0].shape == np.asarray(a).shape, "LU factor has same shape as input")
@icontract.ensure(lambda result, a: result[1].shape == (np.asarray(a).shape[0],), "Pivot array has length n")
def lu_factor(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return scipy.linalg.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)


@register_atom(witness_scipy_lu_solve)
@icontract.require(lambda lu_and_piv, b: len(lu_and_piv) == 2, "lu_and_piv must be a tuple of (lu, piv)")
@icontract.require(
    lambda lu_and_piv, b: lu_and_piv[0].shape[0] == np.asarray(b).shape[0],
    "Dimensions of LU and b must match",
)
@icontract.ensure(lambda result, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def lu_solve(
    lu_and_piv: Tuple[np.ndarray, np.ndarray],
    b: ArrayLike,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    return scipy.linalg.lu_solve(
        lu_and_piv,
        b,
        trans=trans,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )
