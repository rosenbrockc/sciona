"""Avocado object augmentation with explicit empirical reference and RNG.

Derived from kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Preserves numerical object augmentation, including duplicate-index dropping and
source retries. File loading is replaced with supplied empirical reference;
source global RandomState draws use instance RandomState. Retry count is explicit.
Dataset orchestration is a separate outstanding component, not claimed here.
"""
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pandas as pd
import string
import logging
from scipy.special import erf
from .plasticc_observations import AstronomicalObject, get_band_central_wavelength, AvocadoException
logger = logging.getLogger(__name__)

class Augmentor:
    """Class used to augment a dataset.

    This class takes :class:`AstronomicalObject` instances as input and
    generates new :class:`AstronomicalObject` instances with the following
    transformations applied:

    - Drop random observations.
    - Drop large blocks of observations.
    - For galactic observations, adjust the brightness (= distance).
    - For extragalactic observations, adjust the redshift.
    - Add noise.

    When changing the redshift, we use the host_specz measurement as the
    redshift of the reference object. While in simulations we might know the
    true redshift, that isn't the case for real experiments.

    The augmentor needs to have some reasonable idea of the properties of the
    survey that it is being applied to. If there is a large dataset that the
    classifier will be used on, then that dataset can be used directly to
    estimate the properties of the survey.

    This class needs to be subclassed to implement survey specific methods.
    These methods are:

    - :func:`Augmentor._augment_metadata`
    - Either :func:`Augmentor._choose_sampling_times` or
      :func:`Augmentor._choose_target_observation_count`
    - :func:`Augmentor._simulate_light_curve_uncertainties`
    - :func:`Augmentor._simulate_detection`

    Parameters
    ----------
    cosmology_kwargs : kwargs (optional)
        Optional parameters to modify the cosmology assumed in the augmentation
        procedure. These kwargs will be passed to
        astropy.cosmology.FlatLambdaCDM.
    """

    def __init__(self, **cosmology_kwargs):
        cosmology_parameters = {'H0': 70, 'Om0': 0.3, 'Tcmb0': 2.725}
        cosmology_parameters.update(cosmology_kwargs)
        self.cosmology = FlatLambdaCDM(**cosmology_parameters)

    def _augment_metadata(self, reference_object):
        """Generate new metadata for the augmented object.

        This method needs to be implemented in survey-specific subclasses of
        this class.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """
        return NotImplementedError

    def _choose_target_observation_count(self, augmented_metadata):
        """Choose the target number of observations for a new augmented light
        curve.

        This method needs to be implemented in survey-specific subclasses of
        this class if using the default implementation of
        `_choose_sampling_times`.

        Parameters
        ==========
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        target_observation_count : int
            The target number of observations in the new light curve.
        """
        return NotImplementedError

    def _choose_sampling_times(self, reference_object, augmented_metadata, max_time_shift=50, block_width=250, window_padding=100, drop_fraction=0.1):
        """Choose the times at which to sample for a new augmented object.

        This method should really be survey specific, but a default
        implementation is included here that works reasonably well for generic
        light curves. If you are implementing a survey specific version of this
        method, you only need to have the reference_object and
        augmented_metadata parameters. The other parameters are different knobs
        for this method.

        This implementation of _choose_sampling_times requires that the method
        _choose_target_observation_count() be defined that returns how many
        observations we should attempt to have for the new light curve. If a
        different implementation of _choose_sampling_times is used, that method
        may not be required.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        augmented_metadata : dict
            The augmented metadata
        max_time_shift : float (optional)
            The new sampling times will be shifted by up to this amount
            relative to the original ones.
        block_width : float (optional)
            A block of observations with a width specified by this parameter
            will be dropped.
        window_padding : float (optional)
            Observations outside of a window bounded by the first and last
            observations in the reference objects light curve with a padding
            specified by this parameter will be dropped.
        drop_fraction : float (optional)
            This fraction of observations will always be dropped when creating
            the augmented light curve.

        Returns
        =======
        sampling_times : pandas Dataframe
            A pandas Dataframe that has the following columns:

            - time : the times of the simulated observations.
            - band : the bands of the simulated observations.
            - reference_time : the times in the reference light curve that
              correspond to the times of the simulated observations.
        """
        target_observation_count = self._choose_target_observation_count(augmented_metadata)
        reference_observations = reference_object.observations
        sampling_times = reference_observations[['time', 'band']].copy()
        sampling_times['reference_time'] = sampling_times['time'].copy()
        start_time = np.min(sampling_times['time'])
        end_time = np.max(sampling_times['time'])
        augmented_redshift = augmented_metadata['redshift']
        reference_redshift = reference_object.metadata['host_specz']
        redshift_scale = (1 + augmented_redshift) / (1 + reference_redshift)
        if augmented_redshift != reference_redshift:
            ref_peak_time = reference_observations['time'].iloc[np.argmax(reference_observations['flux'].values)]
            sampling_times['time'] = ref_peak_time + redshift_scale * (sampling_times['time'] - ref_peak_time)
        sampling_times['time'] += self.rng.uniform(-max_time_shift, max_time_shift)
        block_start = self.rng.uniform(start_time - block_width, end_time)
        block_end = block_start + block_width
        block_mask = (sampling_times['time'] < block_start) | (sampling_times['time'] > block_end)
        sampling_times = sampling_times[block_mask].copy()
        sampling_times = sampling_times[(sampling_times['time'] > start_time - window_padding).values & (sampling_times['time'] < end_time + window_padding).values].copy()
        if len(sampling_times) == 0:
            return sampling_times
        num_fill = int(target_observation_count * (redshift_scale - 1))
        if num_fill > 0:
            new_indices = self.rng.choice(sampling_times.index, num_fill, replace=True)
            new_rows = sampling_times.loc[new_indices]
            new_rows['band'] = self.rng.choice(reference_object.bands, num_fill, replace=True)
            sampling_times = pd.concat([sampling_times, new_rows])
        num_drop = int(max(len(sampling_times) - target_observation_count, drop_fraction * target_observation_count))
        drop_indices = self.rng.choice(sampling_times.index, num_drop, replace=False)
        sampling_times = sampling_times.drop(drop_indices).copy()
        sampling_times.reset_index(inplace=True, drop=True)
        return sampling_times

    def _simulate_light_curve_uncertainties(self, observations, augmented_metadata):
        """Simulate the observation-related noise for a light curve.

        This method needs to be implemented in survey-specific subclasses of
        this class. It should simulate the observation uncertainties for the
        light curve.

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        return NotImplementedError

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.

        This method needs to be implemented in survey-specific subclasses of
        this class. It should simulate whether each observation is detected as
        a point-source by the survey and set the "detected" flag in the
        observations DataFrame. It should also return whether or not the light
        curve passes a base set of criterion to be included in the sample that
        this classifier will be applied to.

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        return NotImplementedError

    def _resample_light_curve(self, reference_object, augmented_metadata):
        """Resample a light curve as part of the augmenting procedure

        This uses the Gaussian process fit to a light curve to generate new
        simulated observations of that light curve.

        In some cases, the light curve that is generated will be accidentally
        shifted out of the frame, or otherwise missed. If that is the case, the
        light curve will automatically be regenerated with the same metadata
        until it is either detected or until the number of tries has exceeded
        self.augment_retries.

        Parameters
        ----------
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        augmented_observations : pandas.DataFrame
            The simulated observations for the augmented object. If the chosen
            metadata leads to an object that is too faint or otherwise unable
            to be detected, None will be returned instead.
        """
        gp = reference_object.get_default_gaussian_process()
        for attempt in range(self.augment_retries):
            observations = self._choose_sampling_times(reference_object, augmented_metadata)
            new_redshift = augmented_metadata['redshift']
            reference_redshift = reference_object.metadata['host_specz']
            redshift_scale = (1 + new_redshift) / (1 + reference_redshift)
            new_wavelengths = np.array([get_band_central_wavelength(i) for i in observations['band']])
            eval_wavelengths = new_wavelengths / redshift_scale
            pred_x_data = np.vstack([observations['reference_time'], eval_wavelengths]).T
            new_fluxes, new_fluxvars = gp(pred_x_data, return_var=True)
            observations['flux'] = new_fluxes
            observations['flux_error'] = np.sqrt(new_fluxvars)
            augment_brightness = augmented_metadata.get('augment_brightness', 0)
            adjust_scale = 10 ** (-0.4 * augment_brightness)
            if reference_redshift != 0:
                delta_distmod = (self.cosmology.distmod(reference_redshift) - self.cosmology.distmod(new_redshift)).value
                adjust_scale *= 10 ** (0.4 * delta_distmod)
            observations['flux'] *= adjust_scale
            observations['flux_error'] *= adjust_scale
            observations['model_flux'] = observations['flux']
            observations['model_flux_error'] = observations['flux_error']
            observations = self._simulate_light_curve_uncertainties(observations, augmented_metadata)
            observations, pass_detection = self._simulate_detection(observations, augmented_metadata)
            if pass_detection:
                return observations
        return None

    def augment_object(self, reference_object, force_success=True):
        """Generate an augmented version of an object.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        force_success : bool
            If True, then if we fail to generate an augmented light curve for a
            specific set of augmented parameters, we choose a different set of
            augmented parameters until we eventually get an augmented light
            curve. This is useful for debugging/interactive work, but when
            actually augmenting a dataset there is a massive speed up to
            ignoring bad light curves without a major change in classification
            performance.

        Returns
        =======
        augmented_object : :class:`AstronomicalObject`
            The augmented object. If force_success is False, this can be None.
        """
        ref_object_id = reference_object.metadata['object_id']
        random_str = ''.join(self.rng.choice(list(string.ascii_letters), 10))
        new_object_id = '%s_aug_%s' % (ref_object_id, random_str)
        while True:
            augmented_metadata = self._augment_metadata(reference_object)
            augmented_metadata['object_id'] = new_object_id
            augmented_metadata['reference_object_id'] = ref_object_id
            observations = self._resample_light_curve(reference_object, augmented_metadata)
            if observations is not None:
                augmented_object = AstronomicalObject(augmented_metadata, observations)
                return augmented_object
            elif not force_success:
                return None
            else:
                logger.warn('Failed to generate a light curve for redshift %.2f. Retrying.' % augmented_metadata['redshift'])

class PlasticcAugmentor(Augmentor):
    """Implementation of an Augmentor for the PLAsTiCC dataset"""

    def _simulate_photoz(self, redshift):
        """Simulate the photoz determination for a lightcurve using the test
        set as a reference.

        I apply the observed differences between photo-zs and spec-zs directly
        to the new redshifts. This does not capture all of the intricacies of
        photo-zs, but it does ensure that we cover all of the available
        parameter space with at least some simulations.

        Parameters
        ----------
        redshift : float
            The new true redshift of the object.

        Returns
        -------
        host_photoz : float
            The simulated photoz of the host.

        host_photoz_error : float
            The simulated photoz error of the host.
        """
        photoz_reference = self._load_photoz_reference()
        while True:
            ref_idx = self.rng.choice(len(photoz_reference))
            ref_specz, ref_photoz, ref_photoz_err = photoz_reference[ref_idx]
            new_diff = (ref_photoz - ref_specz) * self.rng.choice([-1, 1])
            new_photoz = redshift + new_diff
            if new_photoz < 0:
                continue
            new_photoz_err = ref_photoz_err * self.rng.normal(1, 0.05)
            break
        return (new_photoz, new_photoz_err)

    def _augment_redshift(self, reference_object, augmented_metadata):
        """Choose a new redshift and simulate the photometric redshift for an
        augmented object

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        augmented_metadata : dict
            The augmented metadata to add the new redshift too. This will be
            updated in place.
        """
        if reference_object.metadata['galactic']:
            augmented_metadata['redshift'] = 0
            augmented_metadata['host_specz'] = 0
            augmented_metadata['host_photoz'] = 0
            augmented_metadata['host_photoz_error'] = 0
            augmented_metadata['augment_brightness'] = self.rng.normal(0.5, 0.5)
        else:
            template_redshift = reference_object.metadata['redshift']
            min_redshift = 0.95 * template_redshift
            max_redshift = 5 * template_redshift
            max_redshift = np.min([max_redshift, 1.5 * (1 + template_redshift) - 1])
            aug_redshift = np.exp(self.rng.uniform(np.log(min_redshift), np.log(max_redshift)))
            aug_photoz, aug_photoz_error = self._simulate_photoz(aug_redshift)
            aug_distmod = self.cosmology.distmod(aug_photoz).value
            augmented_metadata['redshift'] = aug_redshift
            augmented_metadata['host_specz'] = aug_redshift
            augmented_metadata['host_photoz'] = aug_photoz
            augmented_metadata['host_photoz_error'] = aug_photoz_error
            augmented_metadata['augment_brightness'] = 0.0

    def _augment_metadata(self, reference_object):
        """Generate new metadata for the augmented object.

        This method needs to be implemented in survey-specific subclasses of
        this class. The new redshift, photoz, coordinates, etc. should be
        chosen in this method.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """
        augmented_metadata = reference_object.metadata.copy()
        self._augment_redshift(reference_object, augmented_metadata)
        if reference_object.metadata['ddf']:
            augmented_metadata['ddf'] = self.rng.rand() > 0.8
        else:
            augmented_metadata['ddf'] = False
        augmented_metadata['mwebv'] *= self.rng.normal(1, 0.1)
        return augmented_metadata

    def _choose_target_observation_count(self, augmented_metadata):
        """Choose the target number of observations for a new augmented light
        curve.

        We use a functional form that roughly maps out the number of
        observations in the PLAsTiCC test dataset for each of the DDF and WFD
        samples.

        Parameters
        ----------
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        target_observation_count : int
            The target number of observations in the new light curve.
        """
        if augmented_metadata['ddf']:
            target_observation_count = int(self.rng.normal(330, 30))
        else:
            gauss_choice = self.rng.choice(3, p=[0.05, 0.4, 0.55])
            if gauss_choice == 0:
                mu = 95
                sigma = 20
            elif gauss_choice == 1:
                mu = 115
                sigma = 8
            elif gauss_choice == 2:
                mu = 138
                sigma = 8
            target_observation_count = int(np.clip(self.rng.normal(mu, sigma), 50, None))
        return target_observation_count

    def _simulate_light_curve_uncertainties(self, observations, augmented_metadata):
        """Simulate the observation-related noise and detections for a light
        curve.

        For the PLAsTiCC dataset, we estimate the measurement uncertainties for
        each band with a lognormal distribution for both the WFD and DDF
        surveys. Those measurement uncertainties are added to the simulated
        observations.

        Parameters
        ----------
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        observations = observations.copy()
        if len(observations) == 0:
            return observations
        if augmented_metadata['ddf']:
            band_noises = {'lsstu': (0.68, 0.26), 'lsstg': (0.25, 0.5), 'lsstr': (0.16, 0.36), 'lssti': (0.53, 0.27), 'lsstz': (0.88, 0.22), 'lssty': (1.76, 0.23)}
        else:
            band_noises = {'lsstu': (2.34, 0.43), 'lsstg': (0.94, 0.41), 'lsstr': (1.3, 0.41), 'lssti': (1.82, 0.42), 'lsstz': (2.56, 0.36), 'lssty': (3.33, 0.37)}
        lognormal_parameters = []
        for band in observations['band']:
            try:
                lognormal_parameters.append(band_noises[band])
            except KeyError:
                raise AvocadoException('Noise properties of band %s not known, add them in PlasticcAugmentor._simulate_light_curve_uncertainties.' % band)
        lognormal_parameters = np.array(lognormal_parameters)
        add_stds = self.rng.lognormal(lognormal_parameters[:, 0], lognormal_parameters[:, 1])
        noise_add = self.rng.normal(loc=0.0, scale=add_stds)
        observations['flux'] += noise_add
        observations['flux_error'] = np.sqrt(observations['flux_error'] ** 2 + add_stds ** 2)
        return observations

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.

        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(observations['flux']) / observations['flux_error']
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        observations['detected'] = self.rng.rand(len(s2n)) < prob_detected
        pass_detection = np.sum(observations['detected']) >= 2
        return (observations, pass_detection)

    def __init__(self, photoz_reference, *, augment_retries, seed):
        super().__init__()
        reference = np.asarray(photoz_reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
            raise ValueError("Expected nonempty empirical reference with three values per row")
        if not np.isfinite(reference).all() or (reference[:, 0] <= 0).any() or (reference[:, 1:] < 0).any():
            raise ValueError("Invalid empirical reference values")
        if isinstance(augment_retries, bool) or not isinstance(augment_retries, int) or augment_retries < 1:
            raise ValueError("augment_retries must be a positive integer")
        self._photoz_reference = reference.copy()
        self.augment_retries = augment_retries
        self.rng = np.random.RandomState(seed)

    def _load_photoz_reference(self):
        return self._photoz_reference

