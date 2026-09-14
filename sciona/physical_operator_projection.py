"""Nonnegative fixed-mass projection and conservative periodic smoothing."""
import numpy as np


def project_and_smooth(predictions,initial_states,*,smoothing=.1):
    """Euclidean projection onto each initial state's nonnegative mass simplex.

Uniform periodic grid: preserving the sum preserves the physical integral for
unchanged grid spacing. Subsequent nearest-neighbor convex smoothing preserves
nonnegativity and mass; it is postprocessing, not an exact PDE time integrator.
"""
    y=np.asarray(predictions,dtype=np.float64);initial=np.asarray(initial_states,dtype=np.float64)
    if y.ndim!=2 or y.shape!=initial.shape or not y.shape[0] or y.shape[1]<2 or not np.isfinite(y).all() or not np.isfinite(initial).all() or (initial<0).any():
        raise ValueError('Expected aligned finite batch/grid arrays and nonnegative initial states')
    if type(smoothing) not in (int,float) or not 0<=smoothing<=.5:
        raise ValueError('Smoothing must lie in [0,0.5]')
    result=np.zeros_like(y)
    for i,row in enumerate(y):
        mass=initial[i].sum()
        if not np.isfinite(mass):raise ValueError('Nonfinite total mass')
        if mass==0:continue
        # Translation does not change the projection and reduces cancellation.
        shifted=row-row.max()
        ordered=np.sort(shifted)[::-1]
        threshold=(np.cumsum(ordered)-mass)/np.arange(1,len(row)+1)
        active=np.flatnonzero(ordered>threshold)
        if not len(active):raise ValueError('Unrepresentable projection')
        projected=np.maximum(shifted-threshold[active[-1]],0.)
        if not np.isfinite(projected).all() or not np.isclose(projected.sum(),mass,rtol=1e-10,atol=1e-12):
            raise ValueError('Numerically unstable mass projection')
        result[i]=projected
    result=(1-2*smoothing)*result+smoothing*(np.roll(result,1,axis=1)+np.roll(result,-1,axis=1))
    if not np.isfinite(result).all():raise ValueError('Nonfinite smoothed result')
    return result
