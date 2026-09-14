"""TrackML first/second hit seeding; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
The source second-hit API requires a pandas Series of runtime hit IDs.
"""
import numpy as np
import pandas as pd
from sciona.trackml_fitting import FittedTrackExtension
from sciona.trackml_nearest import _from_tangent as helixWithTangentVector

class SeededTrackExtension(FittedTrackExtension):

    def chooseLikelyFirstHits(self, run):
        """Find hits which are likely to be the first hit of a track.
        Args:
            run (Run): the Run object holding the data structures for this round
        Returns:
            nh (pd.DataFrame): dataframe with a 'hit_id' column listing the candidate hits.
                Note: Best not to rely on any other columns being present.
                XXX This should really be changed into an array of hit_ids.
        """
        nh = run.neighbors.findFirstHitNeighborhood()
        self.log('chosen candidates for first hits: ', len(nh))
        return nh

    def chooseLikelySecondHits(self, run, hit_id, origin_coords=(0.0, 0.0, 0.0)):
        """Given an array of potential first hits of tracks, choose a set of likely second
        hits for each first hit to get seeds for starting candidate tracks.
        Args:
            run (Run): the Run object holding the data structures for this round
            hit_id (int32 array or pd.Series): the hit_ids of the first candidate hits
            origin_coords (3-tuple of coordinates): coordinates to assume for the
                 most likely origin of particle tracks.
        Returns:
            nb_df (pd.DataFrame): Dataframe with two columns:
                'hit_id': hit id of the first hit of the candidate
                'nb_hit_id': hit id of the second hit of the candidate
                XXX change to a list of two arrays?
        """
        nb_dfs = []
        x0, y0, z0 = run.event.hitCoordinatesById(hit_id)
        hel_pitch = np.full(x0.shape, 1000000.0)
        considered = []
        zo = origin_coords[2]
        for origin_z in (zo, zo - run.params['nb__origin_dz'], zo + run.params['nb__origin_dz']):
            ux0, uy0, uz0 = (x0 - origin_coords[0], y0 - origin_coords[1], z0 - origin_z)
            hel_xm, hel_ym, hel_r = helixWithTangentVector(x0, y0, z0, ux0, uy0, uz0, hel_pitch)
            last_x, last_y, last_z = (x0, y0, z0)
            last_hel_s = 0.0
            for i_layer in range(int(run.params['nb__nlayers'])):
                xi, yi, zi, cyl_closer, next_id, dphi, hel_s, _ = self.intersector.findNextHelixIntersection(last_x, last_y, last_z, uz0, hel_xm, hel_ym, hel_r, hel_pitch)
                hel_s += last_hel_s
                last_hel_s = hel_s
                last_x, last_y, last_z = (xi, yi, zi)
                r20_sqr = np.square(x0) + np.square(y0)
                r20 = np.sqrt(r20_sqr)
                r0 = np.sqrt(r20_sqr + np.square(z0))
                factor = (hel_s / r0) ** run.params['nb__radius_exp']
                cyl_dz = np.sqrt(run.params['nb__cyl_origin_area'] * run.params['nb__cyl_scale'] / np.pi) * factor
                cos_theta = z0 / r0
                abs_sin_theta = np.sqrt(1 - np.square(cos_theta))
                cap_dr2 = run.params['nb__cap_origin_radius'] * abs_sin_theta * factor
                radius = cap_dr2
                radius[cyl_closer] = cyl_dz[cyl_closer]
                mask_new = np.full(len(xi), True)
                for prev_cyl_closer, prev_next_id in considered:
                    mask_new[(cyl_closer == prev_cyl_closer) & (next_id == prev_next_id)] = False
                self.log('new intersections for origin_z = ', origin_z, ': ', np.sum(mask_new))
                nb_df = run.neighbors.findIntersectionNeighborhood(xi[mask_new], yi[mask_new], zi[mask_new], cyl_closer[mask_new], next_id[mask_new], radius[mask_new])
                if nb_df is not None:
                    vind = nb_df['vind'].values
                    nb_df.drop(columns='vind', inplace=True)
                    nb_df['hit_id'] = hit_id.values[mask_new][vind]
                    nb_df['xi'] = xi[mask_new][vind]
                    nb_df['yi'] = yi[mask_new][vind]
                    nb_df['zi'] = zi[mask_new][vind]
                    nb_dfs.append(nb_df)
                considered.append((cyl_closer, next_id))
        return pd.concat(nb_dfs, ignore_index=True) if nb_dfs else None
