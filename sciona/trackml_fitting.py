"""TrackML fit/follow support; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
"""
import numpy as np
from sciona.trackml_extension import TrackExtension
from sciona.trackml_geometry import _source_circle as circleFromThreePoints
from sciona.trackml_helix import _pitch_two as helixPitchFromTwoPoints
from sciona.trackml_helix import _pitch_three as helixPitchLeastSquares

class FittedTrackExtension(TrackExtension):

    def fitTracks(self, run, crossing=-1, mask=None, origin_coords=(0.0, 0.0, 0.0), pitch_from=(1, 2), step=0):
        """Fit helix parameters for candidate tracks.
        XXX document arguments
        """
        x0, y0, z0 = run.candidates.hitCoordinates(crossing - 2, mask=mask)
        x1, y1, z1 = run.candidates.hitCoordinates(crossing - 1, mask=mask)
        x2, y2, z2 = run.candidates.hitCoordinates(crossing - 0, mask=mask)
        for coord, origin_coord in zip((x0, y0, z0), origin_coords):
            coord[np.isnan(coord)] = origin_coord
        hel_xm, hel_ym, hel_r = circleFromThreePoints(x0, y0, x1, y1, x2, y2)
        assert hel_xm.shape == hel_ym.shape == hel_r.shape == x0.shape
        xs = (x0, x1, x2)
        ys = (y0, y1, y2)
        zs = (z0, z1, z2)
        pitch_coords = (coords[index] for index in pitch_from for coords in (xs, ys, zs))
        hel_pitch, _, hel_dz = helixPitchFromTwoPoints(*pitch_coords, hel_xm, hel_ym)
        assert hel_pitch.shape == hel_dz.shape
        hel_pitch_ls, _, _, loss = helixPitchLeastSquares(x0, y0, z0, x1, y1, z1, x2, y2, z2, hel_xm, hel_ym)
        run.candidates.setFit(crossing, 'hel_xm', hel_xm, mask=mask)
        run.candidates.setFit(crossing, 'hel_ym', hel_ym, mask=mask)
        run.candidates.setFit(crossing, 'hel_r', hel_r, mask=mask)
        run.candidates.setFit(crossing, 'hel_pitch', hel_pitch, mask=mask)
        run.candidates.setFit(crossing, 'hel_pitch_ls', hel_pitch_ls, mask=mask)
        run.candidates.setFit(crossing, 'hel_ploss', loss, mask=mask)
        is_hel_r_too_small = hel_r < run.params['fit__hel_r_min']
        if np.any(is_hel_r_too_small):
            if mask is not None:
                keep_mask = np.ones(run.candidates.n, dtype=np.bool)
                keep_mask[mask] = ~is_hel_r_too_small
            else:
                keep_mask = ~is_hel_r_too_small
            self.log('dropping candidates with too small hel_r: %d' % np.sum(is_hel_r_too_small))
            run.candidates.update(keep_mask=keep_mask)
