"""TrackML pitch and helix movement; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution at the pinned source commit.
See docs/licenses/TrackML-BSD-2-Clause.txt for copyright and license.
"""
import numpy as np


def _vectors(values):
    arrays=[np.asarray(value) for value in values]
    if any(a.ndim!=1 or a.dtype!=np.float64 or a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Aligned finite float64 vectors required')
    return arrays


def _evaluate(function,arrays,**options):
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        result=function(*arrays,**options)
    if any(not np.isfinite(a).all() for a in result):
        raise ValueError('Source helix calculation has no finite result')
    return result


def _replacement(zero_pitch):
    if isinstance(zero_pitch,bool) or not isinstance(zero_pitch,(int,float,np.floating)) or not np.isfinite(zero_pitch):
        raise ValueError('Finite real zero-pitch replacement required')
    return zero_pitch


def pitch_from_two_points(x1,y1,z1,x2,y2,z2,hel_xm,hel_ym,*,zero_pitch=.001):
    return _evaluate(_pitch_two,_vectors([x1,y1,z1,x2,y2,z2,hel_xm,hel_ym]),zero_pitch=_replacement(zero_pitch))


def pitch_least_squares(x1,y1,z1,x2,y2,z2,x3,y3,z3,hel_xm,hel_ym,*,zero_pitch=.001):
    return _evaluate(_pitch_three,_vectors([x1,y1,z1,x2,y2,z2,x3,y3,z3,hel_xm,hel_ym]),zero_pitch=_replacement(zero_pitch))


def move_along_helix(x0,y0,z0,uz0,hel_xm,hel_ym,hel_r,hel_pitch,hel_s):
    return _evaluate(_move,_vectors([x0,y0,z0,uz0,hel_xm,hel_ym,hel_r,hel_pitch,hel_s]))


def _pitch_two(x1, y1, z1, x2, y2, z2, hel_xm, hel_ym, zero_pitch=0.001):
    """Calculate the pitch of a helix from two points on the helix,
    when the helix center is known. The points are assumed to be
    separated by less than half a turn of the helix.
    Args:
        x1, y1, z1, x2, y2, z2 (array (N,)): coordinates of the two points
            Note: The order of the points on the helix does not matter for
            the pitch calculation, it only affects the signs of the returned
            phid and dz values.
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        zero_pitch (float): pitch value to replace zero pitch with to avoid numerical problems
    Returns:
        hel_pitch (array (N,)): helix pitch
        phid (array (N,)): helix polar angle difference between the points
        dz (array (N,)): z-difference of the points. Normally this is (z2 - z1),
            but if the helix pitch would be zero, this is replaced with a
            matching value.
    """
    phi1 = np.arctan2(y1 - hel_ym, x1 - hel_xm)
    phi2 = np.arctan2(y2 - hel_ym, x2 - hel_xm)
    phid = phi2 - phi1
    phid[phid > np.pi] -= 2 * np.pi
    phid[phid <= -np.pi] += 2 * np.pi
    dz = z2 - z1
    phid[phid == 0] = 0.001
    hel_pitch = dz / phid * 2 * np.pi
    has_zero_pitch = hel_pitch == 0.0
    hel_pitch[has_zero_pitch] = zero_pitch
    dz[has_zero_pitch] = zero_pitch * phid[has_zero_pitch] / (2 * np.pi)
    return (hel_pitch, phid, dz)

def _pitch_three(x1, y1, z1, x2, y2, z2, x3, y3, z3, hel_xm, hel_ym, zero_pitch=0.001):
    """Calculate the pitch of a helix from three points on the helix,
    when the helix center is known. The points are assumed to be
    in helix path order and separated by less than half a turn of the
    helix between consecutive points. The result is the
    helix pitch that minimizes the squared error in the z-coordinate.
    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 (array (N,)): coordinates of the points
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        zero_pitch (float): pitch value to replace zero pitch with to avoid numerical problems
    Returns:
        hel_pitch (array (N,)): helix pitch
        phid (array (N,)): helix polar angle difference from point 2 to point 3
        dz (array (N,)): z-difference of the points. Normally this is (z3 - z2),
            but if the helix pitch would be zero, this is replaced with a
            matching value.
        loss (array (N,)): sum of squared residuals (assuming the optimum
            intercept is chosen).
    """
    phi1 = np.arctan2(y1 - hel_ym, x1 - hel_xm)
    phi2 = np.arctan2(y2 - hel_ym, x2 - hel_xm)
    phi3 = np.arctan2(y3 - hel_ym, x3 - hel_xm)
    phid21 = phi2 - phi1
    phid31 = phi3 - phi1
    phid21[phid21 >= np.pi] -= 2 * np.pi
    phid21[phid21 < -np.pi] += 2 * np.pi
    phid21neg = phid21 < 0
    phid31[~phid21neg & (phid31 < phid21)] += 2 * np.pi
    phid31[phid21neg & (phid31 > phid21)] -= 2 * np.pi
    sum_phi = phid21 + phid31
    sum_phisqr = np.square(phid21) + np.square(phid31)
    sum_z = z1 + z2 + z3
    sum_zphi = z2 * phid21 + z3 * phid31
    n = 3.0
    hel_p = (n * sum_zphi - sum_z * sum_phi) / (n * sum_phisqr - np.square(sum_phi))
    hel_pitch = 2 * np.pi * hel_p
    dz = z3 - z2
    has_zero_pitch = hel_pitch == 0.0
    phid = phid31 - phid21
    hel_pitch[has_zero_pitch] = zero_pitch
    dz[has_zero_pitch] = zero_pitch * phid[has_zero_pitch] / (2 * np.pi)
    zs = np.stack([z1, z2, z3], axis=1)
    phis = np.stack([np.zeros_like(phid21), phid21, phid31], axis=1)
    zs -= (sum_z / n)[:, np.newaxis]
    phis -= (sum_phi / n)[:, np.newaxis]
    loss = np.sum(np.square(zs - hel_p[:, np.newaxis] * phis), axis=1)
    return (hel_pitch, phid, dz, loss)

def _move(x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch, hel_s):
    """Move the given point along the helix by the given arc length.
    Args:
        x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
        uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
            Note: Only the sign of uz0 is used in the calculation.
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        hel_r (array (n_points,)): radii of the helices in the x,y-plane
        hel_pitch (array (n_points,)): pitches of the helices along the z-axis
        hel_s (array (n_points,)): arc length to move in direction given by uz0
            (negative values move in negated uz0 direction)
    Returns:
        xf (array (n_points,)): x-coordinate of end point
        yf (array (n_points,)): y-coordinate of end point
        zf (array (n_points,)): z-coordinate of end point
        dphi (array (n_points,)): helix phase difference moved
    """
    sign_dphi = np.sign(uz0 * hel_pitch)
    hel_p = hel_pitch / (2 * np.pi)
    dphi = sign_dphi * hel_s / np.sqrt(np.square(hel_r) + np.square(hel_p))
    dx = x0 - hel_xm
    dy = y0 - hel_ym
    cos_dphi = np.cos(dphi)
    sin_dphi = np.sin(dphi)
    xf = hel_xm + cos_dphi * dx - sin_dphi * dy
    yf = hel_ym + sin_dphi * dx + cos_dphi * dy
    zf = z0 + dphi * hel_p
    return (xf, yf, zf, dphi)
