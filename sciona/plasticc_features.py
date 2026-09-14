"""Complete Avocado custom numerical featurizer, not a full pipeline.

Source kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Public source algorithm constants retained; no observations or dataset files.
Compatibility change: removed numpy.warnings alias uses Python warnings.
NaNs and source feature choices are preserved without imputation.
"""
import warnings
import numpy as np
import pandas as pd

class Featurizer:
    """Class used to extract features from objects."""

    def extract_raw_features(self, astronomical_object, return_model=False):
        """Extract raw features from an object

        Featurizing is slow, so the idea here is to extract a lot of different
        things, and then in `select_features` these features are postprocessed
        to select the ones that are actually fed into the classifier. This
        allows for rapid iteration of training on different feature sets. Note
        that the features produced by this method are often unsuitable for
        classification, and may include data leaks. Make sure that you
        understand what features can be used for real classification before
        making any changes.

        For now, there is no generic featurizer, so this must be implemented in
        survey-specific subclasses.

        Parameters
        ----------
        astronomical_object : :class:`AstronomicalObject`
            The astronomical object to featurize.
        return_model : bool
            If true, the light curve model is also returned. Defaults to False.

        Returns
        -------
        raw_features : dict
            The raw extracted features for this object.
        model : dict (optional)
            A dictionary with the light curve model in each band. This is only
            returned if return_model is set to True.
        """
        return NotImplementedError

    def select_features(self, raw_features):
        """Select features to use for classification

        This method should take a DataFrame or dictionary of raw features,
        produced by `featurize`, and output a list of processed features that
        can be fed to a classifier.

        Parameters
        ----------
        raw_features : pandas.DataFrame or dict
            The raw features extracted using `featurize`.

        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.
        """
        return NotImplementedError

    def extract_features(self, astronomical_object):
        """Extract features from an object.

        This method extracts raw features with `extract_raw_features`, and then
        selects the ones that should be used for classification with
        `select_features`. This method is just a wrapper around those two
        methods and is intended to be used as a shortcut for feature extraction
        on individual objects.

        Parameters
        ----------
        astronomical_object : :class:`AstronomicalObject`
            The astronomical object to featurize.

        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.
        """
        raw_features = self.extract_raw_features(astronomical_object)
        features = self.select_features(raw_features)
        return features
pad = 100
plasticc_start_time = 59580
plasticc_end_time = 60675
plasticc_bands = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']

class PlasticcFeaturizer(Featurizer):
    """Class used to extract features for the PLAsTiCC dataset."""

    def extract_raw_features(self, astronomical_object, return_model=False):
        """Extract raw features from an object

        Featurizing is slow, so the idea here is to extract a lot of different
        things, and then in `select_features` these features are postprocessed
        to select the ones that are actually fed into the classifier. This
        allows for rapid iteration of training on different feature sets. Note
        that the features produced by this method are often unsuitable for
        classification, and may include data leaks. Make sure that you
        understand what features can be used for real classification before
        making any changes.

        This class implements a featurizer that is tuned to the PLAsTiCC
        dataset.

        Parameters
        ----------
        astronomical_object : :class:`AstronomicalObject`
            The astronomical object to featurize.
        return_model : bool
            If true, the light curve model is also returned. Defaults to False.

        Returns
        -------
        raw_features : dict
            The raw extracted features for this object.
        model : dict (optional)
            A dictionary with the light curve model in each band. This is only
            returned if return_model is set to True.
        """
        from scipy.signal import find_peaks
        features = dict()
        gp_start_time = plasticc_start_time - pad
        gp_end_time = plasticc_end_time + pad
        gp_times = np.arange(gp_start_time, gp_end_time + 1)
        gp, gp_observations, gp_fit_parameters = astronomical_object.fit_gaussian_process()
        gp_fluxes = astronomical_object.predict_gaussian_process(plasticc_bands, gp_times, uncertainties=False, fitted_gp=gp)
        times = gp_observations['time']
        fluxes = gp_observations['flux']
        flux_errors = gp_observations['flux_error']
        bands = gp_observations['band']
        s2ns = fluxes / flux_errors
        metadata = astronomical_object.metadata
        features['host_specz'] = metadata['host_specz']
        features['host_photoz'] = metadata['host_photoz']
        features['host_photoz_error'] = metadata['host_photoz_error']
        features['ra'] = metadata['ra']
        features['decl'] = metadata['decl']
        features['mwebv'] = metadata['mwebv']
        features['ddf'] = metadata['ddf']
        features['count'] = len(fluxes)
        for i, fit_parameter in enumerate(gp_fit_parameters):
            features['gp_fit_%d' % i] = fit_parameter
        max_times = gp_start_time + np.argmax(gp_fluxes, axis=1)
        med_max_time = np.median(max_times)
        max_dts = max_times - med_max_time
        max_fluxes = np.array([gp_fluxes[band_idx, time - gp_start_time] for band_idx, time in enumerate(max_times)])
        features['max_time'] = med_max_time
        for band, max_flux, max_dt in zip(plasticc_bands, max_fluxes, max_dts):
            features['max_flux_%s' % band] = max_flux
            features['max_dt_%s' % band] = max_dt
        min_fluxes = np.min(gp_fluxes, axis=1)
        for band, min_flux in zip(plasticc_bands, min_fluxes):
            features['min_flux_%s' % band] = min_flux
        positive_widths = np.sum(np.clip(gp_fluxes, 0, None), axis=1) / max_fluxes
        negative_widths = np.sum(np.clip(gp_fluxes, None, 0), axis=1) / min_fluxes
        for band_idx, band_name in enumerate(plasticc_bands):
            features['positive_width_%s' % band_name] = positive_widths[band_idx]
            features['negative_width_%s' % band_name] = negative_widths[band_idx]
        abs_diffs = np.sum(np.abs(gp_fluxes[:, 1:] - gp_fluxes[:, :-1]), axis=1)
        for band_idx, band_name in enumerate(plasticc_bands):
            features['abs_diff_%s' % band_name] = abs_diffs[band_idx]
        fractions = [0.8, 0.5, 0.2]
        for band_idx, band_name in enumerate(plasticc_bands):
            forward_times = find_time_to_fractions(gp_fluxes[band_idx], fractions)
            backward_times = find_time_to_fractions(gp_fluxes[band_idx], fractions, forward=False)
            for fraction, forward_time, backward_time in zip(fractions, forward_times, backward_times):
                features['time_fwd_max_%.1f_%s' % (fraction, band_name)] = forward_time
                features['time_bwd_max_%.1f_%s' % (fraction, band_name)] = backward_time
        thresholds = [-20, -10, -5, -3, 3, 5, 10, 20]
        for threshold in thresholds:
            if threshold < 0:
                count = np.sum(s2ns < threshold)
            else:
                count = np.sum(s2ns > threshold)
            features['count_s2n_%d' % threshold] = count
        features['frac_background'] = np.sum(np.abs(s2ns) < 3) / len(s2ns)
        for band_idx, band_name in enumerate(plasticc_bands):
            mask = bands == band_name
            band_fluxes = fluxes[mask]
            band_flux_errors = flux_errors[mask]
            total_band_s2n = np.sqrt(np.sum((band_fluxes / band_flux_errors) ** 2))
            features['total_s2n_%s' % band_name] = total_band_s2n
            for percentile in (10, 30, 50, 70, 90):
                try:
                    val = np.percentile(band_fluxes, percentile)
                except IndexError:
                    val = np.nan
                features['percentile_%s_%d' % (band_name, percentile)] = val
        thresholds = [5, 10, 20]
        for threshold in thresholds:
            significant_times = times[np.abs(s2ns) > threshold]
            if len(significant_times) < 2:
                dt = -1
            else:
                dt = np.max(significant_times) - np.min(significant_times)
            features['time_width_s2n_%d' % threshold] = dt
        time_bins = [(-5, 5, 'center'), (-20, -5, 'rise_20'), (-50, -20, 'rise_50'), (-100, -50, 'rise_100'), (-200, -100, 'rise_200'), (-300, -200, 'rise_300'), (-400, -300, 'rise_400'), (-500, -400, 'rise_500'), (-600, -500, 'rise_600'), (-700, -600, 'rise_700'), (-800, -700, 'rise_800'), (5, 20, 'fall_20'), (20, 50, 'fall_50'), (50, 100, 'fall_100'), (100, 200, 'fall_200'), (200, 300, 'fall_300'), (300, 400, 'fall_400'), (400, 500, 'fall_500'), (500, 600, 'fall_600'), (600, 700, 'fall_700'), (700, 800, 'fall_800')]
        diff_times = times - med_max_time
        for start, end, label in time_bins:
            mask = (diff_times > start) & (diff_times < end)
            count = np.sum(mask)
            features['count_max_%s' % label] = count
            if count == 0:
                bin_mean_fluxes = np.nan
                bin_std_fluxes = np.nan
            else:
                bin_start = np.clip(int(med_max_time + start - gp_start_time), 0, len(gp_times))
                bin_end = np.clip(int(med_max_time + end - gp_start_time), 0, len(gp_times))
                if bin_start == bin_end:
                    scale_gp_fluxes = np.nan
                    bin_mean_fluxes = np.nan
                    bin_std_fluxes = np.nan
                else:
                    scale_gp_fluxes = gp_fluxes[:, bin_start:bin_end] / max_fluxes[:, None]
                    bin_mean_fluxes = np.mean(scale_gp_fluxes)
                    bin_std_fluxes = np.std(scale_gp_fluxes)
            features['mean_max_%s' % label] = bin_mean_fluxes
            features['std_max_%s' % label] = bin_std_fluxes
        for positive in (True, False):
            for band_idx, band_name in enumerate(plasticc_bands):
                if positive:
                    band_flux = gp_fluxes[band_idx]
                    base_name = 'peaks_pos_%s' % band_name
                else:
                    band_flux = -gp_fluxes[band_idx]
                    base_name = 'peaks_neg_%s' % band_name
                peaks, properties = find_peaks(band_flux, height=np.max(np.abs(band_flux) / 5.0))
                num_peaks = len(peaks)
                features['%s_count' % base_name] = num_peaks
                sort_heights = np.sort(properties['peak_heights'])[::-1]
                for i in range(1, 3):
                    if num_peaks > i:
                        rel_height = sort_heights[i] / sort_heights[0]
                    else:
                        rel_height = np.nan
                    features['%s_frac_%d' % (base_name, i + 1)] = rel_height
        if return_model:
            model = {}
            for idx, band in enumerate(plasticc_bands):
                model['%s' % band] = gp_fluxes[idx]
            model['time'] = gp_times
            model = pd.DataFrame(model).set_index('time')
            return (features, model)
        else:
            return features

    def select_features(self, raw_features):
        """Select features to use for classification

        This method should take a DataFrame or dictionary of raw features,
        produced by `featurize`, and output a list of processed features that
        can be fed to a classifier.

        Parameters
        ----------
        raw_features : pandas.DataFrame or dict
            The raw features extracted using `featurize`.

        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.
        """
        rf = raw_features
        features = type(rf)()
        copy_keys = ['host_photoz', 'host_photoz_error']
        for copy_key in copy_keys:
            features[copy_key] = rf[copy_key]
        features['length_scale'] = rf['gp_fit_1']
        max_flux = rf['max_flux_lssti']
        max_mag = -2.5 * np.log10(np.abs(max_flux))
        features['max_mag'] = max_mag
        features['pos_flux_ratio'] = rf['max_flux_lssti'] / (rf['max_flux_lssti'] - rf['min_flux_lssti'])
        features['max_flux_ratio_red'] = np.abs(rf['max_flux_lssty']) / (np.abs(rf['max_flux_lssty']) + np.abs(rf['max_flux_lssti']))
        features['max_flux_ratio_blue'] = np.abs(rf['max_flux_lsstg']) / (np.abs(rf['max_flux_lssti']) + np.abs(rf['max_flux_lsstg']))
        features['min_flux_ratio_red'] = np.abs(rf['min_flux_lssty']) / (np.abs(rf['min_flux_lssty']) + np.abs(rf['min_flux_lssti']))
        features['min_flux_ratio_blue'] = np.abs(rf['min_flux_lsstg']) / (np.abs(rf['min_flux_lssti']) + np.abs(rf['min_flux_lsstg']))
        features['max_dt'] = rf['max_dt_lssty'] - rf['max_dt_lsstg']
        features['positive_width'] = rf['positive_width_lssti']
        features['negative_width'] = rf['negative_width_lssti']
        features['time_fwd_max_0.5'] = rf['time_fwd_max_0.5_lssti']
        features['time_fwd_max_0.2'] = rf['time_fwd_max_0.2_lssti']
        features['time_fwd_max_0.5_ratio_red'] = rf['time_fwd_max_0.5_lssty'] / (rf['time_fwd_max_0.5_lssty'] + rf['time_fwd_max_0.5_lssti'])
        features['time_fwd_max_0.5_ratio_blue'] = rf['time_fwd_max_0.5_lsstg'] / (rf['time_fwd_max_0.5_lsstg'] + rf['time_fwd_max_0.5_lssti'])
        features['time_fwd_max_0.2_ratio_red'] = rf['time_fwd_max_0.2_lssty'] / (rf['time_fwd_max_0.2_lssty'] + rf['time_fwd_max_0.2_lssti'])
        features['time_fwd_max_0.2_ratio_blue'] = rf['time_fwd_max_0.2_lsstg'] / (rf['time_fwd_max_0.2_lsstg'] + rf['time_fwd_max_0.2_lssti'])
        features['time_bwd_max_0.5'] = rf['time_bwd_max_0.5_lssti']
        features['time_bwd_max_0.2'] = rf['time_bwd_max_0.2_lssti']
        features['time_bwd_max_0.5_ratio_red'] = rf['time_bwd_max_0.5_lssty'] / (rf['time_bwd_max_0.5_lssty'] + rf['time_bwd_max_0.5_lssti'])
        features['time_bwd_max_0.5_ratio_blue'] = rf['time_bwd_max_0.5_lsstg'] / (rf['time_bwd_max_0.5_lsstg'] + rf['time_bwd_max_0.5_lssti'])
        features['time_bwd_max_0.2_ratio_red'] = rf['time_bwd_max_0.2_lssty'] / (rf['time_bwd_max_0.2_lssty'] + rf['time_bwd_max_0.2_lssti'])
        features['time_bwd_max_0.2_ratio_blue'] = rf['time_bwd_max_0.2_lsstg'] / (rf['time_bwd_max_0.2_lsstg'] + rf['time_bwd_max_0.2_lssti'])
        features['frac_s2n_5'] = rf['count_s2n_5'] / rf['count']
        features['frac_s2n_-5'] = rf['count_s2n_-5'] / rf['count']
        features['frac_background'] = rf['frac_background']
        features['time_width_s2n_5'] = rf['time_width_s2n_5']
        features['count_max_center'] = rf['count_max_center']
        features['count_max_rise_20'] = rf['count_max_rise_20'] + features['count_max_center']
        features['count_max_rise_50'] = rf['count_max_rise_50'] + features['count_max_rise_20']
        features['count_max_rise_100'] = rf['count_max_rise_100'] + features['count_max_rise_50']
        features['count_max_fall_20'] = rf['count_max_fall_20'] + features['count_max_center']
        features['count_max_fall_50'] = rf['count_max_fall_50'] + features['count_max_fall_20']
        features['count_max_fall_100'] = rf['count_max_fall_100'] + features['count_max_fall_50']
        all_peak_pos_frac_2 = [rf['peaks_pos_lsstu_frac_2'], rf['peaks_pos_lsstg_frac_2'], rf['peaks_pos_lsstr_frac_2'], rf['peaks_pos_lssti_frac_2'], rf['peaks_pos_lsstz_frac_2'], rf['peaks_pos_lssty_frac_2']]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            features['peak_frac_2'] = np.nanmedian(all_peak_pos_frac_2, axis=0)
        features['total_s2n'] = np.sqrt(rf['total_s2n_lsstu'] ** 2 + rf['total_s2n_lsstg'] ** 2 + rf['total_s2n_lsstr'] ** 2 + rf['total_s2n_lssti'] ** 2 + rf['total_s2n_lsstz'] ** 2 + rf['total_s2n_lssty'] ** 2)
        all_frac_percentiles = []
        for percentile in (10, 30, 50, 70, 90):
            frac_percentiles = []
            for band in plasticc_bands:
                percentile_flux = rf['percentile_%s_%d' % (band, percentile)]
                max_flux = rf['max_flux_%s' % band]
                min_flux = rf['min_flux_%s' % band]
                frac_percentiles.append((percentile_flux - min_flux) / (max_flux - min_flux))
            all_frac_percentiles.append(np.nanmedian(frac_percentiles, axis=0))
        features['percentile_diff_10_50'] = all_frac_percentiles[0] - all_frac_percentiles[2]
        features['percentile_diff_30_50'] = all_frac_percentiles[1] - all_frac_percentiles[2]
        features['percentile_diff_70_50'] = all_frac_percentiles[3] - all_frac_percentiles[2]
        features['percentile_diff_90_50'] = all_frac_percentiles[4] - all_frac_percentiles[2]
        return features

def find_time_to_fractions(fluxes, fractions, forward=True):
    """Find the time for a lightcurve to decline to specific fractions of
    maximum light.

    Parameters
    ----------
    fluxes : numpy.array
        A list of GP-predicted fluxes at 1 day intervals.
    fractions : list
        A decreasing list of the fractions of maximum light that will be found
        (eg: [0.8, 0.5, 0.2]).
    forward : bool
        If True (default), look forward in time. Otherwise, look backward in
        time.

    Returns
    -------
    times : numpy.array
        A list of times for the lightcurve to decline to each of the given
        fractions of maximum light.
    """
    max_time = np.argmax(fluxes)
    max_flux = fluxes[max_time]
    result = np.zeros(len(fractions))
    result[:] = np.nan
    frac_idx = 0
    offset = 0
    while True:
        offset += 1
        if forward:
            new_time = max_time + offset
            if new_time >= fluxes.shape:
                break
        else:
            new_time = max_time - offset
            if new_time < 0:
                break
        test_flux = fluxes[new_time]
        while test_flux < max_flux * fractions[frac_idx]:
            result[frac_idx] = offset
            frac_idx += 1
            if frac_idx == len(fractions):
                break
        if frac_idx == len(fractions):
            break
    return result
