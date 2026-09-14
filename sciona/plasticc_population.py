"""Avocado in-memory dataset and augmentation orchestration.

Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Source kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Numerical dataset methods retained; file persistence and plotting omitted.
Settings resolve to pinned numerical defaults, without working-directory overrides.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from .plasticc_defaults import DEFAULTS as settings
from .plasticc_observations import AstronomicalObject, AvocadoException
from .plasticc_augmentation import PlasticcAugmentor

class Dataset:

    def __init__(self, name, metadata, observations=None, objects=None, chunk=None, num_chunks=None, object_class=AstronomicalObject):
        """Create a new Dataset from a set of metadata and observations"""
        metadata = metadata.copy()
        if observations is not None:
            observations = observations.copy()
        self.name = name
        self.metadata = metadata
        self.chunk = chunk
        self.num_chunks = num_chunks
        self.object_class = object_class
        self.predictions = None
        self.classifier = None
        if observations is None:
            if objects is not None:
                self.objects = np.asarray(objects)
            else:
                self.objects = None
        else:
            self.objects = np.zeros(len(self.metadata), dtype=object)
            self.objects[:] = None
            meta_dicts = self.metadata.to_dict('records')
            for object_id, object_observations in observations.groupby('object_id'):
                meta_index = self.metadata.index.get_loc(object_id)
                if type(meta_index) != int:
                    raise AvocadoException("Error: found multiple metadata entries for object_id=%s! Can't handle." % object_id)
                object_metadata = meta_dicts[meta_index]
                object_metadata['object_id'] = object_id
                new_object = self.object_class(object_metadata, object_observations)
                self.objects[meta_index] = new_object
        self.raw_features = None
        self.features = None
        self.models = None

    def __len__(self):
        return len(self.metadata)

    @classmethod
    def from_objects(cls, name, objects, **kwargs):
        """Load a dataset from a list of AstronomicalObject instances.

        Parameters
        ----------
        objects : list
            A list of AstronomicalObject instances.
        name : str
            The name of the dataset.
        **kwargs
            Additional arguments to pass to Dataset()

        Returns
        -------
        dataset : :class:`Dataset`
            The loaded dataset.
        """
        metadata = pd.DataFrame([i.metadata for i in objects])
        metadata.set_index('object_id', inplace=True)
        dataset = cls(name, metadata, objects=objects, **kwargs)
        return dataset

    def label_folds(self, num_folds=None, random_state=None):
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
        if 'class' not in self.metadata:
            raise AvocadoException("Dataset %s does not have labeled classes! Can't separate into folds." % self.name)
        if 'reference_object_id' in self.metadata:
            is_augmented = True
            reference_mask = self.metadata['reference_object_id'].isna()
            reference_metadata = self.metadata[reference_mask]
        else:
            is_augmented = False
            reference_metadata = self.metadata
        reference_classes = reference_metadata['class']
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        fold_map = {}
        for fold_number, (fold_train, fold_val) in enumerate(folds.split(reference_classes, reference_classes)):
            for object_id in reference_metadata.index[fold_val]:
                fold_map[object_id] = fold_number
        if is_augmented:
            fold_indices = self.metadata['reference_object_id'].map(fold_map)
            fold_indices[reference_mask] = self.metadata.index.to_series().map(fold_map)
        else:
            fold_indices = self.metadata.index.to_series().map(fold_map)
        fold_indices = fold_indices.astype(int)
        return fold_indices

    def extract_raw_features(self, featurizer, keep_models=False):
        """Extract raw features from the dataset.

        The raw features are saved as `self.raw_features`.

        Parameters
        ----------
        featurizer : :class:`Featurizer`
            The featurizer that will be used to calculate the features.
        keep_models : bool
            If true, the models used for the features are kept and stored as
            Dataset.models. Note that not all featurizers support this.

        Returns
        -------
        raw_features : pandas.DataFrame
            The extracted raw features.
        """
        list_raw_features = []
        object_ids = []
        models = {}
        for obj in tqdm(self.objects, desc='Object', dynamic_ncols=True):
            obj_features = featurizer.extract_raw_features(obj, return_model=keep_models)
            if keep_models:
                obj_features, model = obj_features
                models[obj.metadata['object_id']] = model
            list_raw_features.append(obj_features.values())
            object_ids.append(obj.metadata['object_id'])
        keys = obj_features.keys()
        raw_features = pd.DataFrame(list_raw_features, index=object_ids, columns=keys)
        raw_features.index.name = 'object_id'
        self.raw_features = raw_features
        if keep_models:
            self.models = models
        return raw_features

    def select_features(self, featurizer):
        """Select features from the dataset for classification.

        This method assumes that the raw features have already been extracted
        for this dataset and are available with `self.raw_features`. Use
        `extract_raw_features` to calculate these from the data directly, or
        `load_features` to recover features that were previously stored on
        disk.

        The features are saved as `self.features`.

        Parameters
        ----------
        featurizer : :class:`Featurizer`
            The featurizer that will be used to select the features.

        Returns
        -------
        features : pandas.DataFrame
            The selected features.
        """
        if self.raw_features is None:
            raise AvocadoException('Must calculate raw features before selecting features!')
        features = featurizer.select_features(self.raw_features)
        self.features = features
        return features

    def predict(self, classifier):
        """Generate predictions using a classifier.

        Parameters
        ----------
        classifier : :class:`Classifier`
            The classifier to use.

        Returns
        -------
        predictions : :class:`pandas.DataFrame`
            A pandas Series with the predictions for each class.
        """
        self.predictions = classifier.predict(self)
        self.classifier = classifier
        return self.predictions

class PopulationAugmentor(PlasticcAugmentor):

    def augment_dataset(self, augment_name, dataset, num_augments, include_reference=True):
        """Generate augmented versions of all objects in a dataset.

        Parameters
        ==========
        augment_name : str
            The name of the augmented dataset.
        dataset : :class:`Dataset`
            The dataset to use as a reference for the augmentation.
        num_augments : int
            The number of times to use each object in the dataset as a
            reference for augmentation. Note that augmentation sometimes fails,
            so this is the number of tries, not the number of sucesses.
        include_reference : bool (optional)
            If True (default), the reference objects are included in the new
            augmented dataset. Otherwise they are dropped.

        Returns
        =======
        augmented_dataset : :class:`Dataset`
            The augmented dataset.
        """
        augmented_objects = []
        for reference_object in tqdm(dataset.objects, desc='Object', dynamic_ncols=True):
            if include_reference:
                augmented_objects.append(reference_object)
            for i in range(num_augments):
                augmented_object = self.augment_object(reference_object, force_success=False)
                if augmented_object is not None:
                    augmented_objects.append(augmented_object)
        augmented_dataset = Dataset.from_objects(augment_name, augmented_objects, chunk=dataset.chunk, num_chunks=dataset.num_chunks)
        return augmented_dataset
