"""TrackML cell features; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
Module geometry and cell values are supplied at runtime; legacy column extraction
uses modern pandas selection. Numerical source behavior is retained.
"""
import numpy as np

class CellFeatures:
    """This class calculates and provides features derived from the
    cell data of the event.
    """

    def __init__(self, algo, run, details=None):
        """Create cell features data for the given algorithm run.
        Args:
            algo (Algorithm): main algorithm object
            run (Algorithm.Run): data and parameters for this algorithm run
            details (None or SimpleNamespace): If given, this is filled with some
                detailed data useful in off-line analyis.
        """
        self._algo = algo
        self._run = run
        with self._algo.timed('calculating cell features'):
            self._cell_features = self._calculateCellFeatures(details=details)

    def cellFeaturesByHitId(self, hit_id):
        """Return cell features for the given hit_ids.
        Args:
            hit_id (np.int32 array(N,)): hit_ids for which to get cell features
        Returns:
            a tuple containing the cell feature arrays:
            Note: All of these arrays are aligned with the given hit_ids.
            [0], udir_xyz_up (array (len(hit_id), 3)):
                    unit vector in (x,y,z) coordinates in the tangent direction
                    implied by the cell features (variant with positive w-component).
            [1], udir_xyz_down (array (len(hit_id), 3)):
                    unit vector in (x,y,z) coordinates in the tangent direction
                    implied by the cell features (variant with negative w-component).
            [2], ncells (int array (len(hit_id),)):
                    number of activated cells in the detection of the respective hit.
        """
        return tuple((feature[hit_id] for feature in self._cell_features))

    def estimateClosestDirection(self, hit_id, udir):
        """Find directions implied by the cells data such that the directions are
        as close as possible to the given ones and give an estimate for the deviation.
        Args:
            hit_id (np.int32 array(N,)): hit_ids for which to look up directions
            udir (np.float64 array(N,3)): unit vectors defining the expected direction
                for each hit_id.
                Note: The selection of the closest direction also works if udir has
                      a non-zero length different from 1. In such cases, consider
                      that the returned `inner` scales linearly with `udir` and
                      that the returned `d` only makes sense if ||udir|| == 1.
        Returns:
            udir_cell (np.float64 array(N,3)): unit vector in the direction implied
                by the cells data
            inner (np.float64 array(N,)): inner product of the given `udir` direction
                and the direction implied by the cell data.
            d (np.float64 array(N,)): a measure for the difference between the two
                directions. The larger `d`, the larger and more significant the
                difference between the directions.
            XXX should probably also return angle of incidence in order to get a better
                idea about the accuracy of the cells features
        """
        udir_cell_up = self._cell_features[0][hit_id]
        udir_cell_down = self._cell_features[1][hit_id]
        ncells = self._cell_features[2][hit_id]
        inner_up = np.einsum('ij,ij->i', udir, udir_cell_up)
        inner_down = np.einsum('ij,ij->i', udir, udir_cell_down)
        is_up_closer = np.abs(inner_up) > np.abs(inner_down)
        udir_cell = np.where(is_up_closer[:, np.newaxis], udir_cell_up, udir_cell_down)
        inner = np.where(is_up_closer, inner_up, inner_down)
        is_backwards = inner < 0
        udir_cell[is_backwards, :] *= -1
        inner[is_backwards] *= -1
        d = np.sqrt(ncells) * (1 - inner)
        return (udir_cell, inner, d)

    def _calculateCellFeatures(self, details=None):
        """Calculate features from cell data.
        Args:
            details (None or SimpleNamespace): If given, this is filled with some
                detailed data useful in off-line analyis.
        """
        event = self._run.event
        assert event.has_cells
        cells_df = event.cells_df
        detectors_df = self._algo.spec.geospec.detectors_df
        n_module_id = detectors_df['module_id'].max().astype(np.int32) + 1
        n_layer_id = detectors_df['layer_id'].max().astype(np.int32) + 1
        umod_id_by_module = detectors_df['module_id'].values.astype(np.int32) + n_module_id * (detectors_df['layer_id'].values.astype(np.int32) + n_layer_id * detectors_df['volume_id'].values.astype(np.int32))
        max_umod_id = umod_id_by_module.max()
        umod_id_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.int32)
        umod_id_by_hit[event.hits_df['hit_id'].values] = event.hits_df['module_id'].values.astype(np.int32) + n_module_id * (event.hits_df['layer_id'].values.astype(np.int32) + n_layer_id * event.hits_df['volume_id'].values.astype(np.int32))
        assert umod_id_by_hit.max() <= max_umod_id
        module_index_by_umod_id = np.zeros(1 + max_umod_id, dtype=np.int32)
        module_index_by_umod_id[umod_id_by_module] = np.arange(len(umod_id_by_module))
        module_index_by_hit = module_index_by_umod_id[umod_id_by_hit]
        colnames = tuple(('rot_%s%s' % (dst, src) for dst in ('x', 'y', 'z') for src in ('u', 'v', 'w')))
        rot_matrix_by_module = detectors_df.loc[:, list(colnames)].to_numpy()
        rot_matrix_by_hit = rot_matrix_by_module[module_index_by_hit]
        rot_matrix_by_hit = rot_matrix_by_hit.reshape((-1, 3, 3))
        pitch_u_by_hit = detectors_df['pitch_u'].values[module_index_by_hit]
        pitch_v_by_hit = detectors_df['pitch_v'].values[module_index_by_hit]
        module_t_by_hit = detectors_df['module_t'].values[module_index_by_hit]
        hit_id = cells_df['hit_id'].values
        ch0 = cells_df['ch0'].values
        ch1 = cells_df['ch1'].values
        ch0_min_by_hit = np.full(1 + event.max_hit_id, np.inf, dtype=np.float32)
        ch0_max_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float32)
        ch1_min_by_hit = np.full(1 + event.max_hit_id, np.inf, dtype=np.float32)
        ch1_max_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float32)
        np.minimum.at(ch0_min_by_hit, hit_id, ch0)
        np.maximum.at(ch0_max_by_hit, hit_id, ch0)
        np.minimum.at(ch1_min_by_hit, hit_id, ch1)
        np.maximum.at(ch1_max_by_hit, hit_id, ch1)
        dch0_by_hit = ch0_max_by_hit - ch0_min_by_hit
        dch1_by_hit = ch1_max_by_hit - ch1_min_by_hit
        multicell_by_hit = (dch0_by_hit > 0) | (dch1_by_hit > 0)
        multicell_by_hit[0] = False
        ncells_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.int16)
        np.add.at(ncells_by_hit, hit_id, np.ones(len(hit_id), dtype=ncells_by_hit.dtype))
        is_strange = multicell_by_hit != (ncells_by_hit > 1)
        self._algo.log('strange hits with unexpected number of pixels: ', np.where(is_strange)[0])
        cu = ch0.astype(np.float64) * pitch_u_by_hit[hit_id]
        cv = ch1.astype(np.float64) * pitch_v_by_hit[hit_id]
        value = cells_df['value'].values
        value_sum_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        value_sum_by_hit[0] = 1.0
        np.add.at(value_sum_by_hit, hit_id, value)
        cu_weighted = value * cu
        cv_weighted = value * cv
        cu_wmean_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        cv_wmean_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        np.add.at(cu_wmean_by_hit, hit_id, cu_weighted)
        np.add.at(cv_wmean_by_hit, hit_id, cv_weighted)
        cu_wmean_by_hit /= value_sum_by_hit
        cv_wmean_by_hit /= value_sum_by_hit
        cu_red = cu - cu_wmean_by_hit[hit_id]
        cv_red = cv - cv_wmean_by_hit[hit_id]
        wcucv = value * cu_red * cv_red
        wcucu = value * np.square(cu_red)
        wcvcv = value * np.square(cv_red)
        sum_wcucv_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        sum_wcucu_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        sum_wcvcv_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        np.add.at(sum_wcucv_by_hit, hit_id, wcucv)
        np.add.at(sum_wcucu_by_hit, hit_id, wcucu)
        np.add.at(sum_wcvcv_by_hit, hit_id, wcvcv)
        indep_u = multicell_by_hit & (sum_wcucu_by_hit > sum_wcvcv_by_hit)
        indep_v = multicell_by_hit & ~indep_u
        dv_over_du_by_hit = np.full(1 + event.max_hit_id, np.inf, dtype=np.float64)
        dv_over_du_by_hit[indep_u] = sum_wcucv_by_hit[indep_u] / sum_wcucu_by_hit[indep_u]
        du_over_dv_by_hit = np.zeros(1 + event.max_hit_id, dtype=np.float64)
        du_over_dv_by_hit[indep_v] = sum_wcucv_by_hit[indep_v] / sum_wcvcv_by_hit[indep_v]
        du_over_dv_zero = du_over_dv_by_hit == 0
        use_du_over_dv = indep_v & ~du_over_dv_zero
        dv_over_du_by_hit[use_du_over_dv] = 1 / du_over_dv_by_hit[use_du_over_dv]
        dv_over_du_by_hit[~multicell_by_hit] = 0
        finite_dv_over_du = np.isfinite(dv_over_du_by_hit)
        uv_by_hit = np.ones(1 + event.max_hit_id, dtype=np.float64)
        uu_by_hit = 1 / np.sqrt(1 + np.square(dv_over_du_by_hit))
        uv_by_hit[finite_dv_over_du] = dv_over_du_by_hit[finite_dv_over_du] * uu_by_hit[finite_dv_over_du]
        assert np.all(uu_by_hit[~finite_dv_over_du] == 0)
        assert np.all(uv_by_hit[~finite_dv_over_du] == 1)
        hepx_by_hit = 0.5 * (np.abs(uu_by_hit) * pitch_u_by_hit + np.abs(uv_by_hit) * pitch_v_by_hit)
        d_udir2 = cu_red * uu_by_hit[hit_id] + cv_red * uv_by_hit[hit_id]
        max_value_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float64)
        np.maximum.at(max_value_by_hit, hit_id, value)
        max_value = max_value_by_hit[hit_id]
        hepx = hepx_by_hit[hit_id]
        partial_pixel_factor = 1.0
        partial_pixel_intercept = -0.5
        d_udir2_forward = d_udir2 + (partial_pixel_factor * (value / max_value) + partial_pixel_intercept) * hepx
        d_udir2_backward = d_udir2 - (partial_pixel_factor * (value / max_value) + partial_pixel_intercept) * hepx
        d_udir2_min_by_hit = np.full(1 + event.max_hit_id, np.inf, dtype=np.float64)
        d_udir2_max_by_hit = np.full(1 + event.max_hit_id, -np.inf, dtype=np.float64)
        np.minimum.at(d_udir2_min_by_hit, hit_id, d_udir2_backward)
        np.maximum.at(d_udir2_max_by_hit, hit_id, d_udir2_forward)
        hl2_by_hit = (d_udir2_max_by_hit - d_udir2_min_by_hit) / 2
        hl2_by_hit[0] = 0
        hl2_by_hit[~multicell_by_hit] = 0
        hl_by_hit = np.sqrt(np.square(hl2_by_hit) + np.square(module_t_by_hit))
        dir_u_by_hit = hl2_by_hit * uu_by_hit
        dir_v_by_hit = hl2_by_hit * uv_by_hit
        dir_w_by_hit = module_t_by_hit
        udir_uvw_up_by_hit = np.stack([dir_u_by_hit, dir_v_by_hit, dir_w_by_hit], axis=1) / hl_by_hit[:, np.newaxis]
        udir_uvw_down_by_hit = udir_uvw_up_by_hit.copy()
        udir_uvw_down_by_hit[:, 2] *= -1
        udir_xyz_up_by_hit = np.einsum('...ij,...j->...i', rot_matrix_by_hit, udir_uvw_up_by_hit)
        udir_xyz_down_by_hit = np.einsum('...ij,...j->...i', rot_matrix_by_hit, udir_uvw_down_by_hit)
        if details is not None:
            details.hepx_by_hit = hepx_by_hit
            details.hl2_by_hit = hl2_by_hit
            details.hl_by_hit = hl_by_hit
            details.d_udir2_min_by_hit = d_udir2_min_by_hit
            details.d_udir2_max_by_hit = d_udir2_max_by_hit
            details.pitch_u_by_hit = pitch_u_by_hit
            details.pitch_v_by_hit = pitch_v_by_hit
            details.module_t_by_hit = module_t_by_hit
            details.rot_matrix_by_hit = rot_matrix_by_hit
            details.cu_red = cu_red
            details.cv_red = cv_red
        return (udir_xyz_up_by_hit, udir_xyz_down_by_hit, ncells_by_hit)
