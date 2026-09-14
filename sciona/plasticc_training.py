"""Avocado fold training and inference with current LightGBM callbacks.

Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Source kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
Numerical source defaults and fold/softmax averaging retained. Removed fit kwargs
verbose=100 and early_stopping_rounds=50 map to equivalent LightGBM callbacks.
No pickle/file persistence; callers supply a reviewed in-memory dataset interface.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .plasticc_weighting import evaluate_weights_flat, weighted_multi_logloss

class Classifier:
    def __init__(self, name):
        self.name = name

class LightGBMClassifier(Classifier):
    """Feature based classifier using LightGBM to classify objects.

    This uses a weighted multi-class logarithmic loss that normalizes for the
    total counts of each class. This classifier is optimized for the metric
    used in the PLAsTiCC Kaggle challenge.

    Parameters
    ----------
    featurizer : :class:`Featurizer`
        The featurizer to use to select features for classification.
    class_weights : dict (optional)
        Weights to use for each class. If not set, equal weights are assumed
        for each class.
    weighting_function : function (optional)
        Function to use to evaluate weights. By default,
        `evaluate_weights_flat` is used which normalizes the weights for each
        class so that their overall weight matches the one set by
        class_weights. Within each class, `evaluate_weights_flat` gives all
        objects equal weights. Any weights function can be used here as long as
        it has the same signature as `evaluate_weights_flat`.
    """

    def __init__(self, name, featurizer, class_weights=None, weighting_function=evaluate_weights_flat):
        super().__init__(name)
        self.featurizer = featurizer
        self.class_weights = class_weights
        self.weighting_function = weighting_function

    def train(self, dataset, num_folds=None, random_state=None, **kwargs):
        """Train the classifier on a dataset

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to use for training.
        num_folds : int (optional)
            The number of folds to use. Default: settings['num_folds']
        random_state : int (optional)
            The random number initializer to use for splitting the folds.
            Default: settings['fold_random_state']
        **kwargs
            Additional parameters to pass to the LightGBM classifier.
        """
        features = dataset.select_features(self.featurizer)
        folds = dataset.label_folds(num_folds, random_state)
        num_folds = np.max(folds) + 1
        object_weights = self.weighting_function(dataset, self.class_weights)
        object_classes = dataset.metadata['class']
        classes = np.unique(object_classes)
        importances = pd.DataFrame()
        predictions = pd.DataFrame(-1 * np.ones((len(object_classes), len(classes))), index=dataset.metadata.index, columns=classes)
        classifiers = []
        for fold in range(num_folds):
            print('Training fold %d.' % fold)
            train_mask = folds != fold
            validation_mask = folds == fold
            train_features = features[train_mask]
            train_classes = object_classes[train_mask]
            train_weights = object_weights[train_mask]
            validation_features = features[validation_mask]
            validation_classes = object_classes[validation_mask]
            validation_weights = object_weights[validation_mask]
            classifier = fit_lightgbm_classifier(train_features, train_classes, train_weights, validation_features, validation_classes, validation_weights, **kwargs)
            validation_predictions = classifier.predict_proba(validation_features, num_iteration=classifier.best_iteration_)
            predictions[validation_mask] = validation_predictions
            importance = pd.DataFrame()
            importance['feature'] = features.columns
            importance['gain'] = classifier.feature_importances_
            importance['fold'] = fold + 1
            importances = pd.concat([importances, importance], axis=0, sort=False)
            classifiers.append(classifier)
        total_logloss = weighted_multi_logloss(object_classes, predictions, object_weights=object_weights, class_weights=self.class_weights)
        unweighted_total_logloss = weighted_multi_logloss(object_classes, predictions, class_weights=self.class_weights)
        print('Weighted log-loss:')
        print('    With object weights:    %.5f' % total_logloss)
        print('    Without object weights: %.5f' % unweighted_total_logloss)
        if 'reference_object_id' in dataset.metadata:
            original_mask = dataset.metadata['reference_object_id'].isnull()
            original_logloss = weighted_multi_logloss(object_classes[original_mask], predictions[original_mask], object_weights=object_weights[original_mask], class_weights=self.class_weights)
            unweighted_original_logloss = weighted_multi_logloss(object_classes[original_mask], predictions[original_mask], class_weights=self.class_weights)
            print('Original un-augmented dataset weighted log-loss:')
            print('    With object weights:    %.5f' % original_logloss)
            print('    Without object weights: %.5f' % unweighted_original_logloss)
        self.importances = importances
        self.train_predictions = predictions
        self.train_classes = object_classes
        self.classifiers = classifiers
        return classifiers

    def predict(self, dataset):
        """Generate predictions for a dataset

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to generate predictions for.

        Returns
        -------
        predictions : :class:`pandas.DataFrame`
            A pandas Series with the predictions for each class.
        """
        features = dataset.select_features(self.featurizer)
        predictions = 0
        for classifier in tqdm(self.classifiers, desc='Classifier', dynamic_ncols=True):
            fold_scores = classifier.predict_proba(features, raw_score=True, num_iteration=classifier.best_iteration_)
            exp_scores = np.exp(fold_scores)
            fold_predictions = exp_scores / np.sum(exp_scores, axis=1)[:, None]
            predictions += fold_predictions
        predictions /= len(self.classifiers)
        predictions = pd.DataFrame(predictions, index=features.index, columns=self.train_predictions.columns)
        return predictions

def fit_lightgbm_classifier(train_features, train_classes, train_weights, validation_features, validation_classes, validation_weights, **kwargs):
    """Fit a LightGBM classifier

    Parameters
    ----------
    train_features : `pandas.DataFrame`
        The features of the training objects.
    train_classes : `pandas.Series`
        The classes of the training objects.
    train_weights : `pandas.Series`
        The weights of the training objects.
    validation_features : `pandas.DataFrame`
        The features of the validation objects.
    validation_classes : `pandas.Series`
        The classes of the validation objects.
    validation_weights : `pandas.Series`
        The weights of the validation objects.
    **kwargs
        Additional parameters to pass to the LightGBM classifier.

    Returns
    -------
    classifier : `lightgbm.LGBMClassifier`
        The fitted LightGBM classifier
    """
    import lightgbm as lgb
    lgb_params = {'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': len(np.unique(train_classes)), 'metric': 'multi_logloss', 'learning_rate': 0.05, 'colsample_bytree': 0.5, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'min_split_gain': 10.0, 'min_child_weight': 2000.0, 'n_estimators': 5000, 'silent': -1, 'verbose': -1, 'max_depth': 7, 'num_leaves': 50}
    lgb_params.update(kwargs)
    fit_params = {'callbacks': [lgb.log_evaluation(100), lgb.early_stopping(50)], 'sample_weight': train_weights}
    fit_params['eval_set'] = [(validation_features, validation_classes)]
    fit_params['eval_sample_weight'] = [validation_weights]
    classifier = lgb.LGBMClassifier(**lgb_params)
    classifier.fit(train_features, train_classes, **fit_params)
    return classifier
