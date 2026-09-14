"""TrackML detector discovery from runtime module geometry; BSD-2-Clause.

Derived from edwinst/trackml_solution. See docs/licenses/TrackML-BSD-2-Clause.txt.
No detector records or file lookup are embedded.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,MeanShift


def discover_detector(modules):
    """Preserve source fixed seven-ring/three-gap detector assumptions."""
    required=['volume_id','layer_id','cx','cy','cz','module_hv']
    if not isinstance(modules,pd.DataFrame) or not set(required)<=set(modules.columns) or not len(modules):
        raise ValueError('Nonempty runtime module geometry table required')
    if modules.columns.duplicated().any():raise ValueError('Unique geometry columns required')
    for name in required:
        if not pd.api.types.is_numeric_dtype(modules[name]) or not np.isfinite(modules[name].to_numpy()).all():
            raise ValueError('Finite numeric geometry fields required')
    if (modules['module_hv']<0).any():raise ValueError('Nonnegative module half-lengths required')
    return DetectorSpec(SimpleNamespace(detectors_df=modules.copy(deep=True)))


def df_vec2_length(df, prefix):
    """Return the length of a vector in the x,y-plane defined by coordinates in
    the given dataframe."""
    return np.sqrt(np.sum([np.square(df[prefix + comp]) for comp in ('x', 'y')], axis=0))

class CylindersSpec:
    """Holds geometrical information about the cylinder layers of the detector.
    """
    cylinder_volume_ids = [8, 13, 17]

    def __init__(self, geospec):
        cyl_modules_df = geospec.detectors_df.loc[geospec.detectors_df['volume_id'].isin(self.cylinder_volume_ids)].copy()
        cyl_modules_df['cr2'] = df_vec2_length(cyl_modules_df, 'c')
        cyl_modules_df['abscz'] = cyl_modules_df['cz'].abs()
        cylinders = cyl_modules_df.groupby(['volume_id', 'layer_id']).agg({'cr2': ['mean', 'max'], 'abscz': 'max', 'module_hv': 'max'})
        cylinders.columns = ['_'.join(col) for col in cylinders.columns.values]
        cylinders = cylinders.sort_values('cr2_mean')
        cylinders['absz_max'] = cylinders['abscz_max'] + cylinders['module_hv_max']
        cylinders['cyl_id'] = range(len(cylinders))
        cylinders.rename(columns={'cr2_mean': 'cyl_r2'}, inplace=True)
        assert np.all(np.diff(cylinders['absz_max']) >= 0)
        self.cyl_absz_max = cylinders['absz_max'].values
        cylinders['absz_cummax'] = cylinders['absz_max'].cummax(axis=0)
        self.cyl_absz_cummax = cylinders['absz_cummax'].values
        self.cyl_absz_cuts = np.unique(self.cyl_absz_cummax)
        absz_max_binned = np.digitize(self.cyl_absz_cummax, self.cyl_absz_cuts, right=True)
        assert np.all(np.diff(absz_max_binned) >= 0)
        binned_unique, binned_indices = np.unique(absz_max_binned, return_index=True)
        assert len(binned_unique) == len(self.cyl_absz_cuts)
        self.cyl_absz_min_cyl_id = np.append(binned_indices, [0], axis=0).astype(np.int8)
        self.cylinders_df = cylinders
        self.cyl_rsqr = np.square(cylinders['cyl_r2'].values)

    def __len__(self):
        """Return the number of cylinder layers."""
        return len(self.cylinders_df)

class CapsSpec:
    """Holds geometrical information about the cap layers of the detector.
    """
    cap_volume_ids = [7, 9, 12, 14, 16, 18]

    def __init__(self, geospec):
        cap_modules_df = geospec.detectors_df.loc[geospec.detectors_df['volume_id'].isin(self.cap_volume_ids)].copy()
        cap_modules_df['cr2'] = df_vec2_length(cap_modules_df, 'c')
        cap_modules_df['r2_max'] = cap_modules_df['cr2'] + cap_modules_df['module_hv']
        cap_modules_df['r2_min'] = cap_modules_df['cr2'] - cap_modules_df['module_hv']
        nrings = 7
        kmeans = KMeans(n_clusters=nrings, n_init=1, random_state=1)
        kmeans.fit(cap_modules_df['r2_max'].values.reshape(-1, 1))
        ring_r2_max = np.sort(kmeans.cluster_centers_.flatten())
        kmeans = KMeans(n_clusters=nrings, n_init=1, random_state=1)
        kmeans.fit(cap_modules_df['r2_min'].values.reshape(-1, 1))
        ring_r2_min = np.sort(kmeans.cluster_centers_.flatten())
        ring_overlap = ring_r2_max[:-1] - ring_r2_min[1:]
        is_ring_gap = ring_overlap < 0
        assert np.sum(is_ring_gap) == 3
        self.ring_gap_r2_min = ring_r2_max[:-1][is_ring_gap]
        self.ring_gap_r2_max = ring_r2_min[1:][is_ring_gap]
        self.ring_gap_r2_min_sqr = np.square(self.ring_gap_r2_min)
        self.ring_gap_r2_max_sqr = np.square(self.ring_gap_r2_max)
        self.ring_overlap_r2_min = ring_r2_min[1:][~is_ring_gap]
        self.ring_overlap_r2_max = ring_r2_max[:-1][~is_ring_gap]
        self.ring_overlap_r2_min_sqr = np.square(self.ring_overlap_r2_min)
        self.ring_overlap_r2_max_sqr = np.square(self.ring_overlap_r2_max)
        caps = cap_modules_df.groupby(['volume_id', 'layer_id']).agg({'cz': 'mean', 'r2_min': 'min', 'r2_max': 'max'})
        caps = caps.sort_values('cz')
        meanshift = MeanShift(bandwidth=2.0)
        meanshift.fit(caps['cz'].values.reshape(-1, 1))
        caps['cap_cz'] = meanshift.cluster_centers_[meanshift.labels_]
        caps['cap_dz'] = caps['cz'] - caps['cap_cz']
        _, indices = np.unique(meanshift.labels_, return_index=True)
        cap_id_codes = np.zeros(len(indices), dtype=np.int8)
        cap_id_codes[np.argsort(indices)] = list(range(len(indices)))
        caps['cap_id'] = cap_id_codes[meanshift.labels_]
        caps_df = caps.groupby('cap_id').agg({'r2_max': 'max', 'r2_min': 'min', 'cap_id': 'first', 'cap_cz': 'first'})
        ncaps = len(caps_df)
        icenter = ncaps // 2
        cz = caps_df['cap_cz'].values
        r2_min = caps_df['r2_min'].values
        r2_first = np.zeros_like(r2_min)
        for direction, i_first in ((-1, icenter - 1), (1, icenter)):
            r2_min_before = r2_min[i_first]
            cz_before = cz[i_first]
            for i in range(1, icenter):
                i_this = i_first + direction * i
                r2_first[i_this] = r2_min_before * cz[i_this] / cz_before
                r2_min_before = min(r2_first[i_this], r2_min[i_this])
                cz_before = cz[i_this]
                if r2_first[i_this] < r2_min[i_this]:
                    r2_first[i_this] = 0.0
        caps_df['r2_first'] = r2_first
        self.cap_layers_df = caps
        self.caps_df = caps_df[['cap_id', 'r2_min', 'r2_max', 'r2_first', 'cap_cz']]
        self.caps_r2_max = self.caps_df['r2_max'].values
        self.caps_r2_max_sqr = np.square(self.caps_r2_max)
        self.caps_r2_min = self.caps_df['r2_min'].values
        self.caps_r2_min_sqr = np.square(self.caps_r2_min)
        self.cap_z = np.unique(caps['cap_cz'].values)

    def __len__(self):
        """Return the number of cap layers."""
        return len(self.caps_df)

class DetectorSpec:
    """Holds geometrical information about cylinder and cap layers of the detector
    and about module positions and orientations.
    """

    def __init__(self, geospec):
        self.geospec = geospec
        self.cylinders = CylindersSpec(geospec)
        self.caps = CapsSpec(geospec)
