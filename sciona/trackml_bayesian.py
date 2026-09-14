"""TrackML Bayesian neighbor scoring; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
Runtime layer functions supply background distances and measurement errors.
Original strict thresholds and nonfinite arithmetic behavior are retained.
"""
import numpy as np

class BayesianNeighborScorer:

    def projectNeighborDisplacement(self, x, y, z, xn, yn, zn):
        """Project the displacement between a hit position and the nearest helix point
        onto two orthogonal directions utheta, uphi.
        Directions:
            utheta is represented by a unit vector in the direction of increasing polar
                angle theta (spherical coordinates).
            uphi is represented by a unit vector in the direction of increasing azimuthal
                angle phi (cylindrical coordinates).
        Args:
            x, y, z (float64 array(N,)): coordinates of the actual hit
            xn, yn, zn (float64 array(N,)): coordinates of the nearest helix point
        Returns:
            d_utheta, d_uphi (float64 array(N,)): (signed) displacements in the
                directions utheta, iphi, respectively.
        """
        r2 = np.sqrt(np.square(x) + np.square(y))
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        utheta_factor = np.abs(z) / (r2 * r)
        utheta_x = utheta_factor * x
        utheta_y = utheta_factor * y
        utheta_z = -np.sign(z) * r2 / r
        uphi_x = -y / r2
        uphi_y = x / r2
        uphi_z = 0
        xd = x - xn
        yd = y - yn
        zd = z - zn
        d_utheta = utheta_x * xd + utheta_y * yd + utheta_z * zd
        d_uphi = uphi_x * xd + uphi_y * yd + uphi_z * zd
        return (d_utheta, d_uphi)

    def predictRandomError(self, run, hel_s, e_theta_meas, e_phi_meas):
        """Predict the random errors of helix intersection prediction.
        """
        e_predict = 0.007884 * hel_s
        e_theta = np.sqrt(np.square(e_theta_meas) + np.square(e_predict * 0.2096))
        e_phi = np.sqrt(np.square(e_phi_meas) + np.square(e_predict * 0.3853))
        return (e_theta, e_phi)

    def bayesianNeighborEvaluation(self, run, nb_df, mask=None, step=0, details=None):
        """Evaluate the neighbor hits found as potential extensions of the candidate hits.
        Args:
            run (Algorithm.Run): data and parameters for this algorithm run
            nb_df (pd.DataFrame): dataframes listing the neighbors found
                used columns are:
                    extend_hit_id: the hit_id of the neighbor suggested
                    cyl_closer, next_id: layer identification of neighbor
                    xn, yn, zn: point on predicted helix nearest to neighbor
                    hel_s: helix arc length from previous layer crossing
                        (currently used for error estimation)
            mask (None or bool array(run.candidates.n,)): if given, True for the
                candidates which were active in the neighbor search. (The
                'extend_index' in nb_df is aligned with the masked candidates.)
            step (integer): identifies the algorithm step for logging, analysis, etc.
            details (None or SimpleNamespace, etc.): if given, some detailed info
                for the supervisor will be stored in this object.
        Returns:
            good_neighbor (bool array(len(nb_df),)): True if the neighbor should
                be considered as an extension of the candidate track
            dubious (bool array(len(nb_df),)): True if the neighbor is dubious
                as the right extension of the candidate track.
        """
        x, y, z = run.event.hitCoordinatesById(nb_df['extend_hit_id'])
        xn = nb_df['xn'].values
        yn = nb_df['yn'].values
        zn = nb_df['zn'].values
        d_theta, d_phi = self.projectNeighborDisplacement(x, y, z, xn, yn, zn)
        ld_utheta_abs, ld_uphi_abs, ldnt_utheta_abs, ldnt_uphi_abs = run.neighbors.evaluateLayerFunctions(x, y, z, nb_df['cyl_closer'].values, nb_df['next_id'].values, functions=('d_utheta_abs', 'd_uphi_abs', 'dnt_utheta_abs', 'dnt_uphi_abs'))
        e_theta, e_phi = self.predictRandomError(run, hel_s=nb_df['hel_s'], e_theta_meas=ldnt_utheta_abs, e_phi_meas=ldnt_uphi_abs)
        if details:
            details.bay_index = nb_df['extend_index'].values
            details.bay_hit_id = nb_df['extend_hit_id'].values
            details.bay_b_theta = ld_utheta_abs
            details.bay_b_phi = ld_uphi_abs
            details.bay_d_theta = d_theta
            details.bay_d_phi = d_phi
            details.bay_e_theta = e_theta
            details.bay_e_phi = e_phi
        de_theta = d_theta / e_theta
        de_phi = d_phi / e_phi
        dbe_theta = ld_utheta_abs / e_theta
        dbe_phi = ld_uphi_abs / e_phi
        de = np.sqrt(np.square(de_theta) + np.square(de_phi))
        cut = run.params['nb__cut_factor'] * np.sqrt(2 * np.log((2 * np.square(dbe_theta) / np.pi + 1) * (2 * np.square(dbe_phi) / np.pi + 1)))
        good_neighbor = de < cut
        dubious = de > run.params['nb__dist_trust'] * cut
        if details:
            details.bay_cut = cut
            details.bay_de = de
        return (good_neighbor, dubious)
