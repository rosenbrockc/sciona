"""Independent mass proxy from the Go Polar Bears published equations.

Reference: aguschin/flavours-of-physics, author solution PDF, section 2.1.
This is a component of the corrected method, not a competition classifier.
Caller supplies consistently scaled physical quantities; no unit conversion or
claim of a relativistically exact invariant mass is made here.
"""
import numpy as np


def mass_proxy(momentum, transverse_momentum, parent_transverse_momentum,
               flight_distance, lifetime):
    """Return the source's positive-longitudinal-branch momentum/speed proxy.

    Daughter arrays have shape (n, 3); parent arrays have shape (n,).
    Invalid physical inputs fail rather than being silently clipped. The
    positive square-root convention is part of the published reconstruction.
    """
    p, pt, parent_pt, distance, time = (
        np.asarray(value, dtype=np.float64)
        for value in (momentum, transverse_momentum,
                      parent_transverse_momentum, flight_distance, lifetime)
    )
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] == 0:
        raise ValueError('Daughter momentum must have nonempty shape (n, 3)')
    if pt.shape != p.shape or any(
        value.shape != (len(p),) for value in (parent_pt, distance, time)
    ):
        raise ValueError('Physical quantity shapes must align exactly')
    if any(not np.isfinite(value).all() for value in (p, pt, parent_pt, distance, time)):
        raise ValueError('Physical quantities must be finite')
    if (np.any(p < 0) or np.any(pt < 0) or np.any(pt > p)
            or np.any(parent_pt < 0) or np.any(distance <= 0) or np.any(time <= 0)):
        raise ValueError('Invalid physical magnitude or nonpositive flight distance/lifetime')
    # Factored difference avoids overflow from squaring the momenta.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        longitudinal = (np.sqrt(p - pt) * np.sqrt(p) * np.sqrt(1 + pt / np.where(p > 0, p, 1))).sum(axis=1)
        result = np.hypot(parent_pt, longitudinal) * (time / distance)
    if not np.isfinite(result).all():
        raise ValueError('Mass proxy is outside the finite numeric range')
    return result
