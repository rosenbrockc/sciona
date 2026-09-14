"""Avocado numerical fold assignment and training weights.

Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Source kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Source settings are explicit caller-supplied keyword arguments; no file access.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .plasticc_observations import AvocadoException

def evaluate_weights_flat(dataset, class_weights=None):
    """Evaluate the weights to use for classification on a dataset.

    The weights are set to normalize each class to have same weight with the
    same weight for each object in a class. If class weights are set, those
    weights are applied after normalization.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to evaluate weights on.
    class_weights : dict (optional)
        Weights to use for each class. If not set, equal weights are assumed
        for each class.

    Returns
    -------
    weights : `pandas.Series`
        The weights that should be used for classification.
    """
    use_metadata = dataset.metadata
    object_classes = use_metadata['class']
    class_counts = object_classes.value_counts()
    norm_class_weights = {}
    for class_name, class_count in class_counts.items():
        if class_weights is not None:
            class_weight = class_weights[class_name]
        else:
            class_weight = 1
        norm_class_weights[class_name] = class_weight * len(object_classes) / class_count
    weights = object_classes.map(norm_class_weights)
    return weights

def evaluate_weights_redshift(dataset, class_weights=None, group_key=None, min_redshift=None, max_redshift=None, num_bins=None, min_bin_count=None, redshift_key=None, *, settings):
    """Evaluate redshift-weighted weights to use to generate a
    rates-independent classifier.

    The redshift range is divided into logarithmically-spaced bins. Each class
    is given the same weights in each bin so that the rates information in the
    training set doesn't affect the classification. A classifier trained using
    these weights will produce a "rates-independent" classification.

    The redshift bins to use are set using a logarithmic range between
    min_redshift and max_redshift with a total of num_bins. Any objects that
    spill out of these bins are included in the first and last bins. A separate
    bin is included for galactic objects at redshift exactly 0.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to evaluate weights on.
    class_weights : dict (optional)
        Weights to use for each class. If not set, equal weights are assumed
        for each class.
    group_key : str (optional)
        If set, the group of each object will be loaded using group_key as the
        key in the dataset's metadata. The weights will be calculated
        independently for each group. This can be useful if there are multiple
        very different survey strategies in the same dataset, all of which have
        their own selection efficiencies. By default,
        settings['redshift_weighting_group_key'] will be used.
    min_redshift : float (optional)
        The minimum redshift bin to use. By default,
        settings['redshift_weighting_min_redshift'] will be used.
    max_redshift : float (optional)
        The maximum redshift bin to use. By default,
        settings['redshift_weighting_max_redshift'] will be used.
    num_bins : int (optional)
        The number of redshift bins to use. By default,
        settings['redshift_weighting_num_bins'] will be used.
    min_bin_count : int (optional)
        The minimum number of counts in each redshift bin. Tis is used to avoid
        having poorly sampled objects in the training set blow up the metric.
        By default, settings['redshift_weighting_min_bin_count'] will be used.
    redshift_key : str (optional)
        The key to use for determining the redshift. When training a
        classifier, this should typically be the spectroscopic redshift of the
        host galaxy because that is the measured "true" redshift for real
        samples. When evaluating on simulated data without spectroscopic
        redshifts, this might need to be changed to the true simulated
        redshift. By default, settings['redshift_weighting_redshift_key'] will
        be used.

    Returns
    -------
    weights : `pandas.Series`
        The weights that should be used for classification.
    """
    if group_key is None:
        group_key = settings['redshift_weighting_group_key']
    if min_redshift is None:
        min_redshift = settings['redshift_weighting_min_redshift']
    if max_redshift is None:
        max_redshift = settings['redshift_weighting_max_redshift']
    if num_bins is None:
        num_bins = settings['redshift_weighting_num_bins']
    if min_bin_count is None:
        min_bin_count = settings['redshift_weighting_min_bin_count']
    if redshift_key is None:
        redshift_key = settings['redshift_weighting_redshift_key']
    use_metadata = dataset.metadata
    redshift_bins = np.logspace(np.log10(min_redshift), np.log10(max_redshift), num_bins + 1)
    redshift_bins[0] = 1e-99
    redshift_bins[-1] = 1e+99
    redshift_bins = np.hstack([-1e+99, redshift_bins])
    redshift_indices = np.searchsorted(redshift_bins, use_metadata[redshift_key]) - 1
    object_classes = use_metadata['class']
    class_names = np.unique(object_classes)
    class_map = {class_name: i for i, class_name in enumerate(class_names)}
    class_indices = [class_map[i] for i in object_classes]
    if group_key is not None:
        groups = use_metadata[group_key]
        group_names = np.unique(groups)
        group_map = {group_name: i for i, group_name in enumerate(group_names)}
        group_indices = [group_map[i] for i in groups]
    else:
        group_names = ['default']
        group_indices = np.zeros(len(use_metadata), dtype=int)
    counts = np.zeros((len(group_names), len(redshift_bins) - 1, len(class_names)))
    for group_index, redshift_index, class_index in zip(group_indices, redshift_indices, class_indices):
        counts[group_index, redshift_index, class_index] += 1
    total_counts = np.sum(counts)
    num_extgal_bins = np.sum(counts[:, 1:, :] > 0.0001 * total_counts)
    class_extgal_counts = np.sum(np.sum(counts[:, 1:, :], axis=0), axis=0)
    class_gal_counts = np.sum(counts[:, 0, :], axis=0)
    extgal_mask = class_extgal_counts > class_gal_counts
    num_extgal_classes = np.sum(extgal_mask)
    extgal_scale = num_extgal_bins / num_extgal_classes
    floor_counts = np.clip(counts, min_bin_count, None)
    weights = total_counts / floor_counts
    weights[:, :, extgal_mask] /= extgal_scale
    if class_weights is not None:
        for class_idx, class_name in enumerate(class_names):
            weights[:, :, class_idx] *= class_weights[class_name]
    object_weights = weights[group_indices, redshift_indices, class_indices]
    object_weights = pd.Series(object_weights, index=use_metadata.index)
    return object_weights

def weighted_multi_logloss(true_classes, predictions, object_weights=None, class_weights=None, return_object_contributions=False):
    """Evaluate a weighted multi-class logloss function.

    Parameters
    ----------
    true_classes : `pandas.Series`
        A pandas series with the true class for each object
    predictions : `pandas.DataFrame`
        A pandas data frame with the predicted probabilities of each class for
        every object. There should be one column for each class.
    object_weights : dict (optional)
        The weights to use for each object. These are used to weight objects
        within a given class. The overall class weights will be normalized to
        the values set by class_weights. If not specified, flat weights are
        used.
    class_weights : dict (optional)
        The weights to use for each class. If not specified, flat weights are
        assumed for each class.
    return_object_contributions : bool (optional)
        If True, return a pandas Series with the individual contributions from
        each object. Otherwise, return the sum over all classes (default).

    Returns
    -------
    logloss : float or `pandas.Series`
        By default, return the weighted multi-class logloss over all classes.
        If return_object_contributions is True, this returns a pandas Series
        with the individual contributions to the logloss from each object
        instead.
    """
    object_loglosses = pd.Series(10000000000.0 * np.ones(len(true_classes)), index=true_classes.index)
    sum_class_weights = 0
    for class_name in np.unique(true_classes):
        class_mask = true_classes == class_name
        class_count = np.sum(class_mask)
        if object_weights is not None:
            class_object_weights = object_weights[class_mask]
        else:
            class_object_weights = np.ones(class_count)
        if class_weights is not None:
            class_weight = class_weights.get(class_name, 1)
        else:
            class_weight = 1
        if class_weight == 0:
            object_loglosses[class_mask] = 0
            continue
        if class_name not in predictions.columns:
            raise AvocadoException('No predictions available for class %s! Either compute them or set the weight for that class to 0.' % class_name)
        class_predictions = predictions[class_name][class_mask]
        class_loglosses = -class_weight * class_object_weights * np.log(class_predictions) / np.sum(class_object_weights)
        object_loglosses[class_mask] = class_loglosses
        sum_class_weights += class_weight
    object_loglosses /= sum_class_weights
    if return_object_contributions:
        return object_loglosses
    else:
        return np.sum(object_loglosses)

def label_folds(dataset, num_folds=None, random_state=None, *, settings):
    """Separate the dataset into groups for k-folding

        This is only applicable to training datasets that have assigned
        classes.

        If the dataset is an augmented dataset, we ensure that the
        augmentations of the same object stay in the same fold.

        Parameters
        ----------
        num_folds : int (optional)
            The number of folds to use. Default: settings['num_folds']
        random_state : int (optional)
            The random number initializer to use for splitting the folds.
            Default: settings['fold_random_state'].

        Returns
        -------
        fold_indices : `pandas.Series`
            A pandas Series where each element is an integer representing the
            assigned fold for each object.
        """
    if num_folds is None:
        num_folds = settings['num_folds']
    if random_state is None:
        random_state = settings['fold_random_state']
    if 'class' not in dataset.metadata:
        raise AvocadoException("Dataset %s does not have labeled classes! Can't separate into folds." % dataset.name)
    if 'reference_object_id' in dataset.metadata:
        is_augmented = True
        reference_mask = dataset.metadata['reference_object_id'].isna()
        reference_metadata = dataset.metadata[reference_mask]
    else:
        is_augmented = False
        reference_metadata = dataset.metadata
    reference_classes = reference_metadata['class']
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_map = {}
    for fold_number, (fold_train, fold_val) in enumerate(folds.split(reference_classes, reference_classes)):
        for object_id in reference_metadata.index[fold_val]:
            fold_map[object_id] = fold_number
    if is_augmented:
        fold_indices = dataset.metadata['reference_object_id'].map(fold_map)
        fold_indices[reference_mask] = dataset.metadata.index.to_series().map(fold_map)
    else:
        fold_indices = dataset.metadata.index.to_series().map(fold_map)
    fold_indices = fold_indices.astype(int)
    return fold_indices
