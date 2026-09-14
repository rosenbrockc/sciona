"""TrackML three-point circle fit; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution at the pinned source commit.
See docs/licenses/TrackML-BSD-2-Clause.txt for copyright and license.
"""
import sys
import numpy as np


def _source_circle(x1, y1, x2, y2, x3, y3, large_radius=np.float64(1000000.0)):
    """Calculate the circle going through three points in the x,y-plane.
    Args:
        x1, y1, x2, y2, x3, y3 (array (N,)): x,y-coordinates of the three points.
        large_radius (np.float64): large radius to use instead of the theoretically
            "infinite radius" if the points are exactly collinear.
    Returns:
        xm, ym (array (N,)): x,y-coordinates of the circle centers
        r (array (N,)): circle radii
    """
    x21 = x2 - x1
    y21 = y2 - y1
    x31 = x3 - x1
    y31 = y3 - y1
    x32 = x3 - x2
    y32 = y3 - y2
    same21 = ~np.logical_or(x21, y21)
    same31 = ~np.logical_or(x31, y31)
    same32 = ~np.logical_or(x32, y32)
    allsame = same21 & same31
    xd = np.where(allsame, x1, np.where(same21, x31, x21))
    yd = np.where(allsame, y1, np.where(same21, y31, y21))
    if np.any(same21):
        print('PROBLEM12', file=sys.stderr)
    if np.any(same31):
        print('PROBLEM13', file=sys.stderr)
    if np.any(same32):
        print('PROBLEM23', file=sys.stderr)
    if np.any(allsame):
        print('PROBLEMALL', file=sys.stderr)
    rsqr1 = np.square(x1) + np.square(y1)
    rsqr2 = np.square(x2) + np.square(y2)
    rsqr3 = np.square(x3) + np.square(y3)
    denom = 2 * (y1 * x32 - x1 * y32 + x2 * y3 - x3 * y2)
    r_too_large = denom == 0
    denom[r_too_large] = 1.0
    xm = -(rsqr1 * y32 - rsqr2 * y31 + rsqr3 * y21) / denom
    ym = (rsqr1 * x32 - rsqr2 * x31 + rsqr3 * x21) / denom
    r = np.sqrt(np.square(x1 - xm) + np.square(y1 - ym))
    r_too_large |= r > large_radius
    r[r_too_large] = large_radius
    r_over_d = large_radius / np.sqrt(np.square(xd[r_too_large]) + np.square(yd[r_too_large]))
    xm[r_too_large] = x1[r_too_large] - r_over_d * yd[r_too_large]
    ym[r_too_large] = y1[r_too_large] + r_over_d * xd[r_too_large]
    return (xm, ym, r)


def circle_from_three_points(x1,y1,x2,y2,x3,y3,*,large_radius=1e6):
    """Fit aligned finite float64 point triples with source large-circle fallback.

    Coincident points at the origin have no source-defined finite tangent;
    singular/nonfinite source results reject rather than inventing geometry.
    """
    arrays=[np.asarray(value) for value in [x1,y1,x2,y2,x3,y3]]
    if any(a.ndim!=1 or a.dtype!=np.float64 or a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Aligned finite float64 coordinate vectors required')
    if isinstance(large_radius,bool) or not np.isscalar(large_radius) or not np.isfinite(large_radius) or large_radius<=0:
        raise ValueError('Positive finite large radius required')
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        result=_source_circle(*arrays,large_radius=np.float64(large_radius))
    if any(not np.isfinite(a).all() for a in result):
        raise ValueError('Source circle has no finite result for these points')
    return result
