"""TrackML track scoring and redundancy filters; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
from sciona.trackml_seeding import SeededTrackExtension
from sciona.trackml_nearest import _nearest as helixNearestPointDistance
from sciona.trackml_nearest import _unit_tangent as helixUnitTangentVector

class RankedTrackExtension(SeededTrackExtension):

    def evaluateLocalBayesianValue(self, run, x, y, z, cyl_closer, layer_id, xn, yn, zn, hel_s):
        d_theta, d_phi = self.projectNeighborDisplacement(x, y, z, xn, yn, zn)
        ld_utheta_abs, ld_uphi_abs, ldnt_utheta_abs, ldnt_uphi_abs = run.neighbors.evaluateLayerFunctions(x, y, z, cyl_closer, layer_id, functions=('d_utheta_abs', 'd_uphi_abs', 'dnt_utheta_abs', 'dnt_uphi_abs'))
        e_theta, e_phi = self.predictRandomError(run, hel_s=hel_s, e_theta_meas=ldnt_utheta_abs, e_phi_meas=ldnt_uphi_abs)
        de_theta = d_theta / e_theta
        de_phi = d_phi / e_phi
        dbe_theta = ld_utheta_abs / e_theta
        dbe_phi = ld_uphi_abs / e_phi
        s = np.sqrt(np.square(de_theta) + np.square(de_phi))
        b = np.sqrt(np.square(dbe_theta) + np.square(dbe_phi))
        p0 = run.params['value__p0']
        xsb = np.exp(-np.square(s) / 2) * (2 * np.square(b) + np.pi) * (1 - p0) / np.pi + p0
        value = np.log(xsb)
        return value

    def evaluateLocally(self, run, i_fit, i_test, pair_test=0, fit_crossing=0, analysis=True):
        mask = run.candidates.has([(i_test, pair_test), i_fit, fit_crossing], allow_negative=False)
        x0, y0, z0 = run.candidates.hitCoordinates(i_test, pair=pair_test, mask=mask)
        xf, yf, zf = run.candidates.hitCoordinates(i_fit, mask=mask)
        hel_xm = run.candidates.getFit(fit_crossing, 'hel_xm', mask=mask).astype(np.float64)
        hel_ym = run.candidates.getFit(fit_crossing, 'hel_ym', mask=mask).astype(np.float64)
        hel_r = run.candidates.getFit(fit_crossing, 'hel_r', mask=mask).astype(np.float64)
        hel_pitch = run.candidates.getFit(fit_crossing, 'hel_pitch', mask=mask).astype(np.float64)
        hel_params = (xf, yf, zf, hel_xm, hel_ym, hel_r, hel_pitch)
        xn, yn, zn, dist = helixNearestPointDistance(*hel_params, x0, y0, z0)
        if run.hasLayerFunction('d_utheta_abs'):
            hit_ids = run.candidates.hitIds(i_test, pair=pair_test, mask=mask)
            cyl_closer = run.hit_in_cyl[hit_ids]
            layer_id = run.hit_layer_id[hit_ids]
            pseudo_hel_s = np.sqrt(np.square(x0 - xf) + np.square(y0 - yf) + np.square(z0 - zf))
            local_value_masked = run.params['value__bayes_weight'] * self.evaluateLocalBayesianValue(run, x0, y0, z0, cyl_closer, layer_id, xn, yn, zn, pseudo_hel_s)
        else:
            r = np.sqrt(np.square(x0) + np.square(y0) + np.square(z0))
            dist /= r
            local_value_masked = -np.square(dist)
        local_value = np.zeros(run.candidates.n, dtype=np.float64)
        np.place(local_value, mask, local_value_masked)
        if analysis:

            def uncompress(x, mask=mask):
                y = np.full(run.candidates.n, np.nan, dtype=np.float64)
                np.place(y, mask, x)
                return y
            local_info = OrderedDict([(k, uncompress(v)) for k, v in [('dist', dist), ('locval', local_value)]])
        else:
            local_info = None
        assert not np.any(np.isnan(local_value))
        return (local_value, mask, local_info)

    def evaluateCellFeatureConsistency(self, run):
        """Evaluate how consistent the candidate tracks are with features
        extracted from the cells data.
        Args:
            run (Run): the Run object holding the data structures for this round
        Returns:
            value (np.float32 array (run.candidates.n,)): value of candidate track
                (the higher the value, the better the track)
        """
        value = np.zeros(run.candidates.n, dtype=np.float32)
        mask = run.candidates.ncross >= 3
        hel_xm = run.candidates.getFit(2, 'hel_xm').astype(np.float64)
        hel_ym = run.candidates.getFit(2, 'hel_ym').astype(np.float64)
        hel_r = run.candidates.getFit(2, 'hel_r').astype(np.float64)
        hel_pitch = run.candidates.getFit(2, 'hel_pitch').astype(np.float64)
        hel_dz = np.full(len(hel_pitch), 1.0, dtype=np.float64)
        hel_params = (hel_dz, hel_xm, hel_ym, hel_r, hel_pitch)
        for i_crossing in range(4):
            for i_pair in range(1):
                mask_check = mask & run.candidates.has([i_crossing, i_pair])
                hit_id = run.candidates.hitIds(i_crossing, pair=i_pair, mask=mask_check)
                is_inner_cyl = run.hit_in_cyl[hit_id] & (run.hit_layer_id[hit_id] < 4)
                mask_check[mask_check] = is_inner_cyl
                hit_id = hit_id[is_inner_cyl]
                assert np.sum(mask_check) == len(hit_id)
                x, y, z = run.event.hitCoordinatesById(hit_id)
                hel_params_masked = tuple((par[mask_check] for par in hel_params))
                udir = helixUnitTangentVector(x, y, z, *hel_params_masked)
                udir_cells, inner, _ = run.cell_features.estimateClosestDirection(hit_id, udir)
                value[mask_check] += inner - run.params['value__cells_bias']
        return value

    def evaluateTracks(self, run, analysis=True):
        """Calculate a heuristic value of each candidate track.
        Args:
            run (Run): the Run object holding the data structures for this round
            analysis (bool): If True, store data for off-line analysis.
        Returns:
            value (np.float32 array (run.candidates.n,)): value of candidate track
                (the higher the value, the better the track)
            eval_df (pd.DataFrame or None): if analysis==True, this is a dataframe
                aligned with the candidates list. `None` if not `analysis`.
        """
        ncross = run.candidates.ncross
        max_ncross = np.amax(ncross, axis=0)
        value = np.zeros(len(ncross), dtype=np.float32)
        nvalues = np.zeros(len(ncross), dtype=np.int8)
        info = OrderedDict()
        for i in range(max_ncross - 3):
            for i_pair in range(2):
                local_value, has_value, local_info = self.evaluateLocally(run, i + 1, i, pair_test=i_pair, fit_crossing=i + 3, analysis=analysis)
                value += local_value
                nvalues += has_value
                if analysis and i_pair == 0:
                    info.update(OrderedDict([('%d_' % i + key, value) for key, value in local_info.items()]))
        for i_pair in range(2):
            local_value, has_value, local_info = self.evaluateLocally(run, ncross - 4, ncross - 1, pair_test=i_pair, fit_crossing=ncross - 2, analysis=analysis)
            value += local_value
            nvalues += has_value
            if analysis and i_pair == 0:
                info.update(OrderedDict([('f_' + key, value) for key, value in local_info.items()]))
        if run.params['value__ploss_weight'] != 0:
            value_hel_ploss = np.zeros(run.candidates.n, dtype=np.float32)
            for i in range(2, max_ncross):
                hel_ploss = run.candidates.getFit(i, 'hel_ploss')
                hel_ploss_valid = np.isfinite(hel_ploss)
                value_hel_ploss[hel_ploss_valid] += run.params['value__ploss_bias'] - hel_ploss[hel_ploss_valid]
            value += run.params['value__ploss_weight'] * value_hel_ploss
        for i_ncross in range(4, max_ncross + 1):
            mask = ncross == i_ncross
            hel_pitches = [run.candidates.getFit(i, 'hel_pitch', mask=mask) for i in range(2, i_ncross)]
            hel_pitches = np.stack(hel_pitches, axis=1)
            hel_rs = [run.candidates.getFit(i, 'hel_r', mask=mask) for i in range(2, i_ncross)]
            hel_rs = np.stack(hel_rs, axis=1)
            hel_curvatures = hel_rs / (np.square(hel_pitches / (2 * np.pi)) + np.square(hel_rs))
            hel_mean = np.mean(np.abs(hel_curvatures), axis=1)
            hel_std = np.std(np.sign(hel_pitches) * hel_curvatures, axis=1)
            fit_value = -run.params['value__fit_weight_hcs'] * (hel_std / hel_mean) / i_ncross
            value[mask] += fit_value
        mask_doubles = ncross == 2
        assert np.all(value[mask_doubles] == 0)
        mask_triples = ncross == 3
        if run.params['follow__weird_triples']:
            value[mask_triples] = -run.candidates.getFit(2, 'hel_ploss', mask=mask_triples)
        else:
            value[mask_triples] -= run.candidates.df['dist'][mask_triples]
        nvalues[mask_triples] += run.candidates.nHits()[mask_triples]
        if run.cell_features is not None:
            value += run.params['value__cells_weight'] * self.evaluateCellFeatureConsistency(run)
        value = value + run.params['value__hit_bonus'] * nvalues + run.params['value__cross_bonus'] * ncross
        if analysis:
            info['value'] = value
            eval_df = pd.DataFrame(data=info)
        else:
            eval_df = None
        return (value, eval_df)

    def dropRedundantTracks(self, run, eval_df, nlayers=3):
        """Drop candidates which are redundant in that they have the same hits
        recorded for the given number of layer crossings at the start.
        """
        keep_mask = run.candidates.findUnique(crossings=list(range(nlayers)))
        self.log('dropping redundant candidates: ', np.sum(~keep_mask))
        run.candidates.update(keep_mask=keep_mask)
        self.log('remaining candidates: ', run.candidates.n)
        if eval_df is not None:
            eval_df = eval_df.loc[keep_mask].reset_index(drop=True)
        return eval_df

    def filterInvalidTrackCandidates(self, run, mask=None, step=0, prev_candidates=None):
        """Filter out candidates which repeatedly hit exactly the same point within the
        latest three layer crossings.
        Returns:
            keep_mask (bool array): aligned with the candidates list *before* dropping
                the candidates, this array is true for the kept candidates.
        """
        xyz0 = run.candidates.hitCoordinates(-3, mask=mask)
        xyz1 = run.candidates.hitCoordinates(-2, mask=mask)
        xyz2 = run.candidates.hitCoordinates(-1, mask=mask)
        is_same_xyz_01 = tuple((a == b for a, b in zip(xyz0, xyz1)))
        is_same_xyz_02 = tuple((a == b for a, b in zip(xyz0, xyz2)))
        is_same_xyz_12 = tuple((a == b for a, b in zip(xyz1, xyz2)))
        is_same_xy_01 = np.all(is_same_xyz_01[:2], axis=0)
        is_same_xy_02 = np.all(is_same_xyz_02[:2], axis=0)
        is_same_all_01 = is_same_xy_01 & is_same_xyz_01[2]
        is_same_all_02 = is_same_xy_02 & is_same_xyz_02[2]
        is_same_all_12 = np.all(is_same_xyz_12, axis=0)
        is_invalid = is_same_all_01 | is_same_all_02 | is_same_all_12 | is_same_xy_01 & is_same_xy_02
        if np.any(is_invalid):
            self.log('dropping candidates with repeated hit coordinates: %d' % np.sum(is_invalid))
        if mask is None:
            keep_mask = ~is_invalid
        else:
            keep_mask = np.ones(run.candidates.n, dtype=np.bool)
            keep_mask[mask] = ~is_invalid
        run.candidates.update(keep_mask=keep_mask)
        return keep_mask
