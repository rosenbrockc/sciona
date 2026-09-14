"""TrackML candidate extension; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
Requires explicit runtime candidate, neighbor, event and detector contexts.
"""
from collections import OrderedDict
from string import ascii_lowercase
import numpy as np
import pandas as pd
from sciona.trackml_bayesian import BayesianNeighborScorer
from sciona.trackml_corrections import HelixCorrector
from sciona.trackml_helix import _pitch_two as helixPitchFromTwoPoints
from sciona.trackml_nearest import _direction as helixDirectionFromTwoPoints
from sciona.trackml_nearest import _nearest as helixNearestPointDistance
from sciona.trackml_nearest import _unit_tangent as helixUnitTangentVector

class TrackExtension(BayesianNeighborScorer):

    def __init__(self, spec, intersector):
        self.spec = spec
        self.intersector = intersector

    def log(self, *args):
        pass

    def shortStats(self, data, fmt='%.4f'):
        """Return a string giving some very brief descriptive statistics of the given array."""
        return ('mean ' + fmt + ' std ' + fmt + ' [' + fmt + '; ' + fmt + ']') % (np.mean(data), np.std(data), np.amin(data, axis=0), np.amax(data, axis=0))

    def getHelixParams(self, run, i_fit0, i_fit1, i_fit2, mask=None, pitch_from=(1, 2), origin_coords=(0.0, 0.0, 0.0)):
        """Get helix parameters for each candidate track.
        XXX document arguments
        """
        x0, y0, z0 = run.candidates.hitCoordinates(i_fit0, mask=mask)
        x1, y1, z1 = run.candidates.hitCoordinates(i_fit1, mask=mask)
        x2, y2, z2 = run.candidates.hitCoordinates(i_fit2, mask=mask)
        for coord, origin_coord in zip((x0, y0, z0), origin_coords):
            coord[np.isnan(coord)] = origin_coord
        hel_xm = run.candidates.getFit(i_fit2, 'hel_xm', mask=mask).astype(np.float64)
        hel_ym = run.candidates.getFit(i_fit2, 'hel_ym', mask=mask).astype(np.float64)
        hel_r = run.candidates.getFit(i_fit2, 'hel_r', mask=mask).astype(np.float64)
        if pitch_from != (1, 2):
            xs = (x0, x1, x2)
            ys = (y0, y1, y2)
            zs = (z0, z1, z2)
            pitch_coords = (coords[index] for index in pitch_from for coords in (xs, ys, zs))
            hel_pitch, _, hel_dz = helixPitchFromTwoPoints(*pitch_coords, hel_xm, hel_ym)
            assert hel_pitch.shape == hel_dz.shape
        else:
            hel_pitch = run.candidates.getFit(i_fit2, 'hel_pitch', mask=mask).astype(np.float64)
            hel_dz = helixDirectionFromTwoPoints(hel_xm, hel_ym, hel_pitch, x1, y1, z1, x2, y2, z2)
        return (hel_dz, hel_xm, hel_ym, hel_r, hel_pitch)

    def dropNeighborsInconsistentWithCellFeatures(self, run, nb_df):
        """Discard neighbor hit candidates the cell features of which are inconsistent
        with the predicted helix parameters.
        Precondition:
            run.cell_features is not None: cell features are available
        Args:
            run (Run): the Run object holding the data structures for this round
            nb_df (pd.DataFrame): dataframes listing the neighbors
        Returns:
            nb_df (pd.DataFrame): filtered dataframe listing the neighbors
                which passed the consistency check.
        """
        assert run.cell_features is not None
        hel_params = tuple((nb_df[col].to_numpy() for col in ('hel_dz', 'hel_xm', 'hel_ym', 'hel_r', 'hel_pitch')))
        hit_id = nb_df['extend_hit_id'].values
        x, y, z = run.event.hitCoordinatesById(hit_id)
        in_cyl = run.hit_in_cyl[hit_id]
        layer_id = run.hit_layer_id[hit_id]
        udir = helixUnitTangentVector(x, y, z, *hel_params)
        udir_cells, inner, d = run.cell_features.estimateClosestDirection(hit_id, udir)
        is_inner_cyl = in_cyl & (layer_id < 4)
        keep = ~is_inner_cyl | (d <= run.params['nb__cells_cut'])
        nb_df = nb_df.loc[keep]
        return nb_df

    def findHelixIntersectionNeighbors(self, run, i_fit0, i_fit1, i_fit2, last_x, last_y, last_z, last_dphi=0.0, k=2, nmax_per_crossing=2, pitch_from=(1, 2), revisit=False, mask=None, force_cyl_closer=None, origin_coords=(0.0, 0.0, 0.0), corrector=None, step=0, details=None):
        """Predict helix intersections for candidate tracks and find the hits closest
        to the predicted intersections.
        Args:
            run (Algorithm.Run): data and parameters for this algorithm run
            i_fit0, i_fit1, i_fit2 (int): crossing indices along the candidate track to use
                for fitting the helix parameters. XXX remove since we always use the last three?
            last_x, last_y, last_z (float64 array (N,)): coordinates of the last known
                point on each track (do not have to be coordinates of a hit).
            last_dphi (float64 scalar or array (N,)): helix phase difference from
                the last known hit to the point (last_x, last_y, last_z)
            k (int): number of neighbors to look up per intersection
            nmax_per_crossing (int): maximum number of neighbors to pair together for one
                extension
            pitch_from (tuple of two int): indicates which of the i_fit* crossings to
                use for helix pitch fitting (and in which order).
            revisit (bool): Whether we are revisiting an already found intersection.
                If True, move backwards along each helix
                before predicting the next intersection (the intention is to revisit
                the latest layer crossing). If False, do the default move forwards
                before predicition the next intersection in order to prevent getting
                stuck at an intersection.
            mask (None or bool array (run.candidates.n,)): If given, consider only the
                track candidates for which mask is True.
            force_cyl_closer (None or bool array (N,)): If given, force predicted
                intersection to be on a cylinder (for True) or on a cap (for False).
            origin_coords (3-tuple of coordinates): coordinates to assume as the first
                 point in helix fitting if only two points are known.
            corrector (None or HelixCorrector): If given, use this corrector to
                 predict perturbations of the helices.
            step (integer): identifies the algorithm step for logging, analysis, etc.
                 XXX also use for follow__weird_triples, not so clean.
            details (None or SimpleNamespace, etc.): if given, some detailed info
                for the supervisor will be stored in this object.
        Returns:
            has_intersection (bool array (N,)): True if an intersection with the
                detector geometry has been predicted for the respective candidate track.
            xi, yi, zi, dphi, hel_s (float64 array (N,)): predicted intersection
                coordinates, phase differences, and arc length (the latter two
                relative to (last_x, last_y, last_z).
            nb_df (pd.DataFrame): dataframe with found extension candidates.
                nb_df['extend_index'] index of candidate track (in the full candidates
                    list if mask is None, otherwise into the masked candidates list).
        where:
            N...is run.candidates.n if mask=None, otherwise N == np.sum(mask)
        """
        assert np.isscalar(last_dphi) or last_dphi.shape == last_x.shape
        if corrector is not None:
            assert corrector.isAlignedWith(mask)
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = corrector.helixParams()
        else:
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = self.getHelixParams(run, i_fit0, i_fit1, i_fit2, pitch_from=pitch_from, mask=mask, origin_coords=origin_coords)
        assert hel_dz.shape == hel_xm.shape == hel_ym.shape == hel_r.shape == hel_pitch.shape == last_x.shape == last_y.shape == last_z.shape
        pre_move = 'back' if revisit else None
        xi, yi, zi, cyl_closer, next_id, dphi, hel_s, mult = self.intersector.findNextHelixIntersection(last_x, last_y, last_z, hel_dz, hel_xm, hel_ym, hel_r, hel_pitch, cyl_pre_move=pre_move, cap_pre_move=pre_move, missable=not revisit, force_cyl_closer=force_cyl_closer, corrector=corrector)
        assert xi.shape == yi.shape == zi.shape == cyl_closer.shape == next_id.shape == dphi.shape == hel_s.shape == mult.shape == last_x.shape
        if details:
            details.cyl_closer = cyl_closer
            details.next_id = next_id
            details.mult = mult
        if corrector is not None:
            corrector.updateHelices(xi, yi, zi, mask=np.isfinite(xi))
            hel_dz, hel_xm, hel_ym, hel_r, hel_pitch = corrector.helixParams()
        total_dphi = np.where(np.isnan(dphi), np.inf, last_dphi + dphi)
        reject = np.abs(total_dphi) >= np.pi
        if details:
            details.nno_intersection = np.sum(np.isnan(xi))
            details.nrejected_dphi = np.sum(reject) - details.nno_intersection
            details.reject_total_dphi = reject & ~np.isnan(xi)
        xi[reject] = np.nan
        yi[reject] = np.nan
        zi[reject] = np.nan
        hel_s[reject] = np.inf
        dphi[reject] = np.nan
        mult[reject] = 0
        has_intersection = np.isfinite(hel_s)
        n_intersection = np.sum(has_intersection, axis=0)
        nb_df = run.neighbors.findIntersectionNeighborhoodK(xi, yi, zi, cyl_closer, next_id, k=k)
        if nb_df is None:
            self.log('no intersection neighbors found (nb_df is None)')
            return (has_intersection, xi, yi, zi, dphi, hel_s, nb_df)
        if details:
            self.log('number of seeds, intersections, and neighbors: ' + '%d -> %d (%d, %.2f%%, %d !i/s, %d dphi) -> %d (%.1f each)' % (len(xi), n_intersection, n_intersection - len(xi), (n_intersection - len(xi)) / len(xi) * 100.0, details.nno_intersection, details.nrejected_dphi, len(nb_df), len(nb_df) / n_intersection))
        assert len(nb_df) == k * n_intersection
        fit_df = pd.DataFrame(data=OrderedDict([('last_x', last_x), ('last_y', last_y), ('last_z', last_z), ('hel_xm', hel_xm), ('hel_ym', hel_ym), ('hel_r', hel_r), ('hel_pitch', hel_pitch), ('hel_dz', hel_dz), ('xi', xi), ('yi', yi), ('zi', zi), ('ri', np.sqrt(np.square(xi) + np.square(yi) + np.square(zi))), ('ri2', np.sqrt(np.square(xi) + np.square(yi))), ('cyl_closer', cyl_closer), ('next_id', next_id), ('dphi', dphi), ('hel_s', hel_s), ('mult', mult), ('prev_hit_id', run.candidates.hitIds(i_fit2, mask=mask))]))
        fit_df.name = 'fit_df'
        nb_df = nb_df.join(fit_df, on='vind', how='left', sort=False).rename(columns={'vind': 'extend_index', 'nb_hit_id': 'extend_hit_id'})
        nb_df.name = 'nb_df'
        del fit_df
        x, y, z = run.event.hitCoordinatesById(nb_df['extend_hit_id'])
        hel_params = [nb_df[col].values for col in ('last_x', 'last_y', 'last_z', 'hel_xm', 'hel_ym', 'hel_r', 'hel_pitch')]
        xn, yn, zn, dist = helixNearestPointDistance(*hel_params, x, y, z)
        nb_df['dist'] = dist
        nb_df['xn'] = xn
        nb_df['yn'] = yn
        nb_df['zn'] = zn
        new_pairs = revisit and run.hasLayerFunction('pair_theta')
        if new_pairs:
            d_theta, d_phi = self.projectNeighborDisplacement(x, y, z, xn, yn, zn)
            db_theta, db_phi, e_theta, e_phi = run.neighbors.evaluateLayerFunctions(x, y, z, nb_df['cyl_closer'].values, nb_df['next_id'].values, functions=('d_utheta_abs', 'd_uphi_abs', 'pair_theta', 'pair_phi'))
            de_theta = np.abs(d_theta) / e_theta
            de_phi = np.abs(d_phi) / e_phi
            dbe_theta = db_theta / e_theta
            dbe_phi = db_phi / e_phi
            weight_theta = np.square(dbe_theta)
            weight_phi = np.square(dbe_phi)
            de = (weight_theta * de_theta + weight_phi * de_phi) / (weight_theta + weight_phi)
            dist = de
            nb_df['dist'] = dist
        dist_reshaped = np.reshape(dist, (dist.shape[0] // k, k))
        dist_argsort = np.argsort(dist_reshaped, axis=1)
        dist_argsort = np.reshape(dist_argsort, dist.shape[0])
        dist_argsort = dist_argsort + np.repeat(np.arange(0, dist.shape[0], k), k)
        nb_df = nb_df.iloc[dist_argsort]
        nb_df.reset_index(drop=True, inplace=True)
        if not revisit:
            stuck = np.full(xi.shape, False)
            stuck[nb_df.loc[nb_df['extend_hit_id'] == nb_df['prev_hit_id'], 'extend_index']] = True
            nstuck = np.sum(stuck)
            if nstuck > 0:
                self.log('stuck intersections: ', nstuck)
                nb_df = nb_df.loc[~stuck[nb_df['extend_index']]]
        assert revisit or not np.any(nb_df['extend_hit_id'] == nb_df['prev_hit_id'])
        if run.cell_features is not None:
            nb_df = self.dropNeighborsInconsistentWithCellFeatures(run, nb_df)
        self.log('neighbors before pairing: ', len(nb_df))
        npaired = []
        forbidden_module_ids = [run.event.hitModuleIdById(nb_df['extend_hit_id'].values)]
        for i_neighbor in range(1, nmax_per_crossing):
            good_pair = (nb_df['extend_index'] == np.roll(nb_df['extend_index'], -1)) & (nb_df['extend_index'] != np.roll(nb_df['extend_index'], 1))
            if new_pairs:
                good_pair &= np.roll(nb_df['dist'].values, -1) < run.params['pair__cut']
            else:
                nb_df['dist_diff'] = np.roll(nb_df['dist'].values, -1) - nb_df['dist'].values
                good_pair &= (nb_df['dist'] < run.params['pair__dist_threshold'] * nb_df['ri2']) & (nb_df['dist_diff'] < run.params['pair__diff_threshold'] * nb_df['ri2'])
            module_id = np.roll(forbidden_module_ids[0], -1)
            same_module_id = np.zeros(len(module_id), dtype=np.bool)
            for forbidden_module_id in forbidden_module_ids:
                same_module_id |= module_id == forbidden_module_id
            same_module_id &= good_pair
            keep_nb = ~np.roll(same_module_id, 1)
            nb_df = nb_df.loc[keep_nb]
            good_pair = good_pair[keep_nb] & ~same_module_id[keep_nb]
            module_id = module_id[keep_nb]
            for i in range(len(forbidden_module_ids)):
                forbidden_module_ids[i] = forbidden_module_ids[i][keep_nb]
            to_copy = good_pair.copy()
            for i_pair in range(1, i_neighbor + 1):
                hit_col = 'extend_hit_id_' + ascii_lowercase[i_pair]
                if i_pair == i_neighbor:
                    nb_df[hit_col] = 0
                is_free_slot = nb_df[hit_col] == 0
                use_this_slot = to_copy & is_free_slot & (i_pair < nb_df['mult'])
                nb_df.loc[use_this_slot, hit_col] = np.roll(nb_df['extend_hit_id'].values, -1)[use_this_slot]
                to_copy &= ~use_this_slot
            module_id[~good_pair] = forbidden_module_ids[0][~good_pair]
            forbidden_module_ids.append(module_id)
            nbefore = len(nb_df)
            not_paired_away = ~np.roll(good_pair, 1)
            nb_df = nb_df.loc[not_paired_away]
            for i in range(len(forbidden_module_ids)):
                forbidden_module_ids[i] = forbidden_module_ids[i][not_paired_away]
            npaired.append(nbefore - len(nb_df))
        self.log('paired: ' + ', '.join(map(str, npaired)))
        if step == 0 and run.params['follow__weird_triples']:
            nb_df['dubious'] = True
        else:
            if not revisit:
                if run.hasLayerFunction('d_utheta_abs'):
                    good_neighbor, dubious = self.bayesianNeighborEvaluation(run, nb_df, mask=mask, step=step, details=details)
                    nb_df['dubious'] = dubious
                    nb_df = nb_df.loc[good_neighbor]
                else:
                    dist_threshold = nb_df['hel_s'] * run.params['nb__dist_threshold']
                    good_neighbor = nb_df['dist'] < dist_threshold
                    nb_df = nb_df.loc[good_neighbor]
                    dist_threshold = dist_threshold[good_neighbor]
                    nb_df['dubious'] = nb_df['dist'] > run.params['nb__dist_trust'] * dist_threshold
            if not nb_df.empty and nb_df['extend_index'].iat[0] == nb_df['extend_index'].iat[-1]:
                nb_df = nb_df.head(1)
            else:
                is_first = nb_df['extend_index'] != np.roll(nb_df['extend_index'], 1)
                nb_df = nb_df.loc[is_first]
        return (has_intersection, xi, yi, zi, dphi, hel_s, nb_df)

    def chooseLikelyNextHits(self, run, k=2, nmax_per_crossing=2, ncross_min_keep=2, nhits_min_keep=2, step=0, origin_coords=(0.0, 0.0, 0.0)):
        df = run.candidates.df
        last_x, last_y, last_z, last_dphi, last_hel_s, last_nskipped = [df[col].values for col in ('xf', 'yf', 'zf', 'dphi', 'hel_s', 'nskipped')]
        corrector = HelixCorrector(self, run)
        has_intersection, xi, yi, zi, dphi, hel_s, nb_df = self.findHelixIntersectionNeighbors(run, -3, -2, -1, last_x, last_y, last_z, last_dphi=last_dphi, k=k, nmax_per_crossing=nmax_per_crossing, origin_coords=origin_coords, corrector=corrector, step=step)
        will_be_extended = np.zeros(run.candidates.n, dtype=np.bool)
        dubious_extension = np.zeros(run.candidates.n, dtype=np.bool)
        if nb_df is not None:
            nneighbors = len(nb_df)
            self.log('neighbors after selection: ', nneighbors)
            self.log('skipping stats: ', self.shortStats(last_nskipped))
            skipped_too_many = last_nskipped > run.params['follow__nskip_max']
            nb_df = nb_df.loc[~skipped_too_many[nb_df['extend_index'].values]]
            self.log('dropped because they skipped too many intersections: ', nneighbors - len(nb_df))
            extend_index = nb_df['extend_index'].values
            will_be_extended[extend_index] = True
            dubious_extension[extend_index] = nb_df['dubious']
            extend_hit_cols = ['extend_hit_id'] + ['extend_hit_id_' + ch for ch in ascii_lowercase[1:nmax_per_crossing]]
            extend_hit_ids = [nb_df[col].values for col in extend_hit_cols]
        else:
            self.log('no neighbors')
            extend_index = []
            extend_hit_ids = []
        df.loc[has_intersection, 'xf'] = xi[has_intersection]
        df.loc[has_intersection, 'yf'] = yi[has_intersection]
        df.loc[has_intersection, 'zf'] = zi[has_intersection]
        df.loc[has_intersection, 'dphi'] += dphi[has_intersection]
        df.loc[has_intersection, 'hel_s'] += hel_s[has_intersection]
        assert np.amax(last_nskipped, axis=0) < np.iinfo(last_nskipped.dtype).max
        df.loc[has_intersection, 'nskipped'] += 1
        close_mask = ~has_intersection & (run.candidates.ncross >= 4)
        keep_mask = (~will_be_extended | dubious_extension) & (run.candidates.ncross >= ncross_min_keep) & (run.candidates.nHits() >= nhits_min_keep)
        n_kept = np.sum(keep_mask, axis=0)
        self.log('keeping ', n_kept, ', of which ', np.sum(keep_mask & has_intersection, axis=0), ' had an intersection')
        run.candidates.update(close_mask=close_mask, keep_mask=keep_mask, extend_index=extend_index, extend_hit_ids=extend_hit_ids)
        if nb_df is not None:
            assert run.candidates.n == n_kept + len(nb_df)
            xf, yf, zf = run.candidates.hitCoordinates(-1)
            df = run.candidates.df
            df.loc[n_kept:, 'xf'] = xf[n_kept:]
            df.loc[n_kept:, 'yf'] = yf[n_kept:]
            df.loc[n_kept:, 'zf'] = zf[n_kept:]
            df.loc[n_kept:, 'nskipped'] = 0
            df.loc[n_kept:, 'dphi'] = 0.0
            df.loc[n_kept:, 'hel_s'] = 0.0
            df.loc[n_kept:, 'dist'] = nb_df['dist'].values
