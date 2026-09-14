"""Independent numerical feature assembly for the corrected source method.

The 46 base measurements and excluded-column position are supplied explicitly
by the caller. This module contains no dataset-specific schema or adapter.
"""
import numpy as np
from sciona.flavours_kinematics import mass_proxy


def prepare_features(base, momentum, transverse_momentum, parent_transverse_momentum,
                     flight_distance, lifetime, excluded_column):
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 46 or not np.isfinite(base).all():
        raise ValueError('Finite base features require exactly 46 ordered columns')
    if (isinstance(excluded_column, (bool, np.bool_))
            or not isinstance(excluded_column, (int, np.integer))
            or not 0 <= excluded_column < 46):
        raise ValueError('Excluded column must be an integer position in the base view')
    proxy = mass_proxy(momentum, transverse_momentum, parent_transverse_momentum,
                       flight_distance, lifetime)
    if len(proxy) != len(base):
        raise ValueError('Physical inputs and base rows must align')
    p, pt = np.asarray(momentum, dtype=float), np.asarray(transverse_momentum, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        longitudinal = (np.sqrt(p - pt) * np.sqrt(p) *
                        np.sqrt(1 + pt / np.where(p > 0, p, 1))).sum(axis=1)
        total = np.hypot(parent_transverse_momentum, longitudinal)
        speed = np.asarray(flight_distance, dtype=float) / np.asarray(lifetime, dtype=float)
        regression = np.column_stack([base, proxy, total, speed, longitudinal])
    if not np.isfinite(regression).all():
        raise ValueError('Derived physical features must be finite')
    return dict(base=base.copy(), restricted=np.delete(base, excluded_column, axis=1),
                proxy=np.column_stack([base, proxy]), regression=regression,
                mass_proxy=proxy)


def classifier_views(prepared, corrected_mass):
    corrected = np.asarray(corrected_mass, dtype=np.float64)
    proxy = prepared['mass_proxy']
    if corrected.shape != proxy.shape or not np.isfinite(corrected).all():
        raise ValueError('Corrected mass must be finite and row-aligned')
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = (proxy + 1e-10) / (corrected + 1e-10)
        enriched = np.column_stack([prepared['base'], proxy, corrected, proxy-corrected, ratio])
    if not np.isfinite(enriched).all():
        raise ValueError('Corrected mass supplements must be finite')
    return {key: prepared[key] for key in ('base', 'restricted', 'proxy')} | {'corrected': enriched}
