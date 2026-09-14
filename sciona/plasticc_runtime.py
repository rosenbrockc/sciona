"""Versioned in-memory Avocado runtime boundary; Community review pending.

Runtime inputs only. No file reads, downloads, model pickle loading or exports.
Numerical components retain source behavior; validation rejects malformed inputs.
"""
from dataclasses import dataclass
from functools import partial
import math
import numpy as np
import pandas as pd
from .plasticc_defaults import DEFAULTS
from .plasticc_observations import AstronomicalObject
from .plasticc_population import Dataset, PopulationAugmentor
from .plasticc_features import PlasticcFeaturizer, plasticc_bands
from .plasticc_training import LightGBMClassifier
from .plasticc_weighting import evaluate_weights_flat, evaluate_weights_redshift
from .plasticc_predictions import create_kaggle_predictions

CLASSES = {6,15,16,42,52,53,62,64,65,67,88,90,92,95}
GALACTIC = {6,16,53,65,92}
TRAIN_PARAMETERS = {'n_estimators','learning_rate','colsample_bytree','reg_alpha','reg_lambda',
    'min_split_gain','min_child_weight','max_depth','num_leaves','min_child_samples','n_jobs','random_seed'}


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def decode_objects(items, training):
    if not isinstance(items,list) or not items:
        raise ValueError('Expected a nonempty object list')
    objects=[]; ids=set()
    keys={'object_id','host_specz','host_photoz','host_photoz_error','redshift','galactic','ddf','mwebv','ra','decl'}
    if training: keys=keys|{'class'}
    for item in items:
        if not isinstance(item,dict) or set(item)!={'metadata','observations'}:
            raise ValueError('Invalid object envelope')
        metadata=item['metadata']
        if not isinstance(metadata,dict) or set(metadata)!=keys:
            raise ValueError('Invalid object metadata contract')
        identifier=metadata['object_id']
        if not isinstance(identifier,str) or not identifier or identifier in ids:
            raise ValueError('Object identifiers must be unique nonempty strings')
        ids.add(identifier)
        if type(metadata['galactic']) is not bool or type(metadata['ddf']) is not bool:
            raise ValueError('Expected boolean survey and galactic flags')
        for key in keys-{'object_id','galactic','ddf','class'}:
            if not finite(metadata[key]): raise ValueError('Metadata numerics must be finite')
        if any(metadata[k]<0 for k in ['host_specz','host_photoz','host_photoz_error','redshift']):
            raise ValueError('Redshift values and errors must be nonnegative')
        if metadata['galactic']:
            if metadata['host_specz']!=0 or metadata['redshift']!=0:
                raise ValueError('Galactic reference redshift must be zero')
        elif training and (metadata['host_specz']<=0 or metadata['redshift']<=0):
            raise ValueError('Extragalactic augmentation requires positive reference redshift')
        if training and (type(metadata['class']) is not int or metadata['class'] not in CLASSES or
                         (metadata['class'] in GALACTIC)!=metadata['galactic']):
            raise ValueError('Invalid source class or galactic classification')
        rows=item['observations']
        if not isinstance(rows,list) or len(rows)<2:
            raise ValueError('At least two observations required')
        for row in rows:
            if not isinstance(row,list) or len(row)!=4 or row[1] not in plasticc_bands or not all(finite(row[i]) for i in [0,2,3]) or row[3]<=0:
                raise ValueError('Invalid numerical observation')
        frame=pd.DataFrame(rows,columns=['time','band','flux','flux_error'])
        if not frame.time.is_monotonic_increasing:
            raise ValueError('Observation times must be ordered')
        obj=AstronomicalObject(metadata,frame)
        if not np.any(obj.subtract_background().flux.to_numpy()!=0):
            raise ValueError('Background-subtracted signal cannot be identically zero')
        objects.append(obj)
    return objects


@dataclass
class Prepared:
    training: Dataset
    prediction: Dataset
    augmentor: PopulationAugmentor
    num_augments: int
    num_folds: int
    fold_seed: int
    weighting: str
    class_weighting: str
    training_parameters: dict


@dataclass
class Completed:
    classifier: LightGBMClassifier
    prediction: Dataset


def prepare_runtime(payload):
    if not isinstance(payload,dict) or set(payload)!={'version','training','prediction','photoz_reference','config'} or type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Expected version1 runtime payload')
    config=payload['config']
    allowed={'num_augments','num_folds','fold_seed','augmentation_seed','weighting','class_weighting','training_parameters'}
    if not isinstance(config,dict) or set(config)-allowed: raise ValueError('Unknown runtime configuration')
    augments=config.get('num_augments',100); folds=config.get('num_folds',DEFAULTS['num_folds'])
    fold_seed=config.get('fold_seed',DEFAULTS['fold_random_state']); seed=config.get('augmentation_seed',0)
    if not integer(augments) or not integer(folds,2) or not integer(fold_seed) or not integer(seed) or max(seed,fold_seed)>=2**32:
        raise ValueError('Invalid augmentation/fold count or seed')
    weighting=config.get('weighting','flat'); class_weighting=config.get('class_weighting','flat')
    if weighting not in ('flat','redshift') or class_weighting not in ('flat','kaggle'):
        raise ValueError('Unknown weighting mode')
    parameters=config.get('training_parameters',{})
    if not isinstance(parameters,dict) or set(parameters)-TRAIN_PARAMETERS or not all(finite(v) for v in parameters.values()):
        raise ValueError('Unsupported training parameter')
    for key in ('n_estimators','num_leaves','min_child_samples','n_jobs'):
        if key in parameters and not integer(parameters[key],2 if key=='num_leaves' else 1):
            raise ValueError('Training counts must be positive integers')
    if 'max_depth' in parameters and (type(parameters['max_depth']) is not int or parameters['max_depth']==0 or parameters['max_depth'] < -1):
        raise ValueError('max_depth must be positive or minus one')
    if 'random_seed' in parameters and (not integer(parameters['random_seed']) or parameters['random_seed']>=2**31):
        raise ValueError('Invalid model seed')
    for key in ('reg_alpha','reg_lambda','min_split_gain','min_child_weight'):
        if key in parameters and parameters[key]<0: raise ValueError('Regularization must be nonnegative')
    if 'learning_rate' in parameters and parameters['learning_rate']<=0:
        raise ValueError('Learning rate must be positive')
    if 'colsample_bytree' in parameters and not 0<parameters['colsample_bytree']<=1:
        raise ValueError('Feature fraction must be in (0,1]')
    train=decode_objects(payload['training'],True); predict=decode_objects(payload['prediction'],False)
    train_ids={o.metadata['object_id'] for o in train}
    if train_ids.intersection(o.metadata['object_id'] for o in predict):
        raise ValueError('Training and prediction identifiers must be disjoint')
    training=Dataset.from_objects('runtime-training',train)
    prediction=Dataset.from_objects('runtime-prediction',predict)
    counts=training.metadata['class'].value_counts()
    if not {42,52,62,95}.issubset(counts.index) or (counts<folds).any():
        raise ValueError('Required postprocessing classes and enough originals per fold are required')
    if prediction.metadata.galactic.any() and not set(counts.index).intersection(GALACTIC):
        raise ValueError('Galactic predictions require a galactic training class')
    augmentor=PopulationAugmentor(payload['photoz_reference'],seed=seed,augment_retries=DEFAULTS['augment_retries'])
    return Prepared(training,prediction,augmentor,augments,folds,fold_seed,weighting,class_weighting,dict(parameters))


def train_runtime(prepared):
    if not isinstance(prepared,Prepared): raise TypeError('Expected prepared runtime')
    population=prepared.augmentor.augment_dataset('runtime-augmented',prepared.training,prepared.num_augments)
    featurizer=PlasticcFeaturizer()
    population.extract_raw_features(featurizer)
    weighting=evaluate_weights_flat if prepared.weighting=='flat' else partial(evaluate_weights_redshift,settings=DEFAULTS)
    class_weights=None if prepared.class_weighting=='flat' else {c:2 if c in (15,64) else 1 for c in CLASSES}
    classifier=LightGBMClassifier('runtime-classifier',featurizer,class_weights=class_weights,weighting_function=weighting)
    classifier.train(population,num_folds=prepared.num_folds,random_state=prepared.fold_seed,**prepared.training_parameters)
    return Completed(classifier,prepared.prediction)


def predict_runtime(completed):
    if not isinstance(completed,Completed): raise TypeError('Expected completed runtime')
    population=completed.prediction
    population.extract_raw_features(completed.classifier.featurizer)
    population.predict(completed.classifier)
    final=create_kaggle_predictions(population)
    values=final.to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values.sum(axis=1),1):
        raise ValueError('Prediction produced invalid probability mass')
    return {'version':1,'object_ids':list(final.index),'classes':[int(c) for c in final.columns],
            'probabilities':values.tolist()}
