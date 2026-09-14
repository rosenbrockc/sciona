"""TrackML nearest-point and tangent geometry; BSD-2-Clause adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
The nearest-point routine preserves the source fixed-iteration approximation.
"""
import numpy as np
from sciona.trackml_helix import _vectors,_evaluate


def _parameter(value,shape):
    array=np.asarray(value)
    if array.ndim==0 and array.dtype.kind in 'iuf':array=array.astype(np.float64)
    if array.dtype!=np.float64 or array.shape not in [(),shape] or not np.isfinite(array).all():
        raise ValueError('Finite scalar or aligned float64 helix parameter required')
    return array


def direction_from_two_points(hel_xm,hel_ym,hel_pitch,x1,y1,z1,x2,y2,z2):
    return _evaluate(_direction,_vectors([hel_xm,hel_ym,hel_pitch,x1,y1,z1,x2,y2,z2]))


def helix_from_tangent(x0,y0,z0,ux0,uy0,uz0,hel_pitch,*,min_uz0=.001,max_mg_over_qB=1e6):
    values=_vectors([x0,y0,z0,ux0,uy0,uz0])
    for value in [min_uz0,max_mg_over_qB]:
        if isinstance(value,bool) or not isinstance(value,(int,float,np.floating)) or not np.isfinite(value) or value<=0:
            raise ValueError('Positive finite regulation limits required')
    pitch=_parameter(hel_pitch,values[0].shape)
    return _evaluate(_from_tangent,[*values,pitch],min_uz0=min_uz0,max_mg_over_qB=max_mg_over_qB)


def nearest_point_distance(x0,y0,z0,hel_xm,hel_ym,hel_r,hel_pitch,x,y,z,*,iterations=3,return_hel_s=False):
    """Execute the source Newton/Kepler approximation, not a global minimizer."""
    targets=_vectors([x,y,z])
    parameters=[_parameter(value,targets[0].shape) for value in [x0,y0,z0,hel_xm,hel_ym,hel_r,hel_pitch]]
    if isinstance(iterations,bool) or not isinstance(iterations,int) or iterations<1:
        raise ValueError('Positive integer Newton iteration count required')
    if not isinstance(return_hel_s,bool):raise ValueError('Boolean optional-output flag required')
    return _evaluate(_nearest,[*parameters,*targets],iterations=iterations,return_hel_s=return_hel_s)


def unit_tangent(x0,y0,z0,uz0,hel_xm,hel_ym,hel_r,hel_pitch):
    return _evaluate(_unit_tangent,_vectors([x0,y0,z0,uz0,hel_xm,hel_ym,hel_r,hel_pitch]))


def _direction(hel_xm, hel_ym, hel_pitch, x1, y1, z1, x2, y2, z2):
    """Get a value indicating the direction of helix motion in the z-coordinate,
    in the sense of moving from a given first point towards a given second point.
    The sign of this value also determines the sense of rotation in the x,y-plane,
    so it is needed (and must be non-zero) even if there is no real motion
    along the z-axis.
    Args:
        hel_xm, hel_ym, hel_pitch: helix parameters
        x1, y1, z1: first point on the helix
        x2, y2, z2: second point on the helix
    Returns:
        hel_dz: A value, the sign of which specifies the sign of the motion in
            the z-coordinate. Usually, hel_dz is simply the difference z2 - z1,
            but the function needs to take care of some special cases when the
            two points have the same z-coordinate.
    """
    hel_dz = z2 - z1
    nodz = hel_dz == 0
    hel_dz[nodz] = hel_pitch[nodz] * np.sign((y1[nodz] - y2[nodz]) * (hel_xm[nodz] - x1[nodz]) + (x2[nodz] - x1[nodz]) * (hel_ym[nodz] - y1[nodz]))
    return hel_dz

def _from_tangent(x0, y0, z0, ux0, uy0, uz0, hel_pitch, min_uz0=0.001, max_mg_over_qB=1000000.0):
    """Construct helices through the given initial points and
    with the given tangent vectors and pitches.
    Args:
        x0, y0, z0 (array (n_points,)): coordinates of initial vertices on the helices
        ux0, uy0, uz0 (array (n_points,)): tangent vector components
            The magnitude of the tangent vector is not used.
            uz0 must be non-zero.
        hel_pitch (array (n_points,) or scalar): pitch(es) of the helices along the z-axis
            Note: For a physical relativistic particle in a constant magnetic
                  field (0, 0, Bz), hel_pitch = -2 * np.pi * pz / (q * Bz),
                  where
                      pz is the z-component of the momentum
                      q is the particle charge
                      Bz is the z-component of the magnetic field
        min_uz0 (float): minimum absolute value to use for uz0 in order to regulate
            numerical instabilities.
        max_mg_over_qB (float): maximum absolute value to allow for (m gamma / (q * Bz)).
    Returns:
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        hel_r (array (n_points,)): radii of the helices in the x,y-plane
    """
    uz0_regulated = np.where(np.abs(uz0) >= min_uz0, uz0, min_uz0 * (1 - 2 * (uz0 < 0)))
    mg_over_qB = -hel_pitch / (2 * np.pi * uz0_regulated)
    too_large = np.abs(mg_over_qB) > max_mg_over_qB
    mg_over_qB[too_large] = np.sign(mg_over_qB[too_large]) * max_mg_over_qB
    hel_xm = x0 + uy0 * mg_over_qB
    hel_ym = y0 - ux0 * mg_over_qB
    hel_r = np.sqrt(np.square(x0 - hel_xm) + np.square(y0 - hel_ym))
    return (hel_xm, hel_ym, hel_r)

def _nearest(x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch, x, y, z, iterations=3, return_hel_s=False):
    """On each helix, find the point nearest to a respectively given refernce point
        and calculate the Euclidean distance between those points.
        Args:
            x0, y0, z0 (array (n_points,) [*]): coordinates of initial vertices on the helices
            hel_xm, hel_ym (array (n_points,) [*]): center coordinates of helices in the x,y-plane
            hel_r (array (n_points,) [*]): radii of the helices in the x,y-plane
            hel_pitch (array (n_points,) [*]): pitches of the helices along the z-axis
            x, y, z (array (n_points,)): reference point coordinates
            iterations (positive int): number of iteration of Newton's method to perform
            return_hel_s (bool): If True, return dphi and hel_s
        Notes:
           [*]...The helix parameters x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch may also be scalars.
                 In that case the same helix is used for each reference point.
        Returns:
            x1 (array (n_points,)): x-coordinate of nearest point
            y1 (array (n_points,)): y-coordinate of nearest point
            z1 (array (n_points,)): z-coordinate of nearest point
            dist (array (n_points,)): Euclidean distance from the reference point to
                the nearest point on the corresponding helix
        Returns optionally:
            dphi (aray (n_points,)): helix phase difference from (x0, y0, z0) to the
                nearest point
            hel_s (aray (n_points,)): positive helix arc length from (x0, y0, z0) to the
                nearest point
        """
    dx = x - hel_xm
    dy = y - hel_ym
    dx0 = x0 - hel_xm
    dy0 = y0 - hel_ym
    dr2 = np.sqrt(np.square(dx) + np.square(dy))
    hel_p = hel_pitch / (2 * np.pi)
    dph_z = (z - z0) / hel_p
    dph_xy = np.arctan2(dy, dx) - np.arctan2(dy0, dx0)
    dph_diff = dph_z - dph_xy
    e = hel_r * dr2 / np.square(hel_p)
    E = np.full(x.shape[0], np.pi)
    k = np.ceil(dph_diff / (2 * np.pi) - 0.5)
    M = np.pi + dph_diff - 2 * k * np.pi
    for i in range(iterations):
        f = E - e * np.sin(E) - M
        fprime = 1.0 - e * np.cos(E)
        E = E - f / fprime
    u = E - np.pi + 2 * k * np.pi
    w = u - dph_diff
    z1 = z + hel_p * w
    dphi = (z - z0) / hel_p + w
    dx0 = x0 - hel_xm
    dy0 = y0 - hel_ym
    cos1 = np.cos(dphi)
    sin1 = np.sin(dphi)
    x1 = hel_xm + cos1 * dx0 - sin1 * dy0
    y1 = hel_ym + sin1 * dx0 + cos1 * dy0
    dist = np.sqrt(np.square(x - x1) + np.square(y - y1) + np.square(z - z1))
    if return_hel_s:
        hel_s = np.sqrt(np.square(hel_r) + np.square(hel_p)) * np.abs(dphi)
        return (x1, y1, z1, dist, dphi, hel_s)
    else:
        return (x1, y1, z1, dist)

def _unit_tangent(x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch):
    """"Return the unit tangent vectors to the given helices at the given points.
    Args:
        x0, y0, z0 (array (n_points,)): coordinates of the points on the helices
        uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
            Note: Only the sign of uz0 is used in the calculation.
        hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
        hel_r (array (n_points,)): radii of the helices in the x,y-plane
        hel_pitch (array (n_points,)): pitches of the helices along the z-axis
    Returns:
        udir (array (n_points,3)): unit tangent vectors
    """
    sign_dphi = np.sign(uz0 * hel_pitch)
    hel_p = hel_pitch / (2 * np.pi)
    signed_norm = sign_dphi * np.sqrt(np.square(hel_r) + np.square(hel_p))
    dx = x0 - hel_xm
    dy = y0 - hel_ym
    udir = np.stack((-dy, dx, hel_p), axis=1) / signed_norm[:, np.newaxis]
    return udir
