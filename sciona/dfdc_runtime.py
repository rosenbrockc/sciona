"""Versioned private DFDC input boundary and complete connected execution.

Inputs carry positional decoded RGB clips, explicit initialization and an
executable checkpoint plan. Native hull predictor paths are private runtime
configuration, never report fields. No weights or input data are downloaded.
"""
import random

import numpy as np
import torch

from sciona.dfdc_codec import decode_array, decode_state
from sciona.dfdc_selection import selection_plan
from sciona.dfdc_folds import assign_folds
from sciona.dfdc_records import build_records
from sciona.dfdc_training_crops import build_preprocessing_detector
from sciona.dfdc_detector import build_detector
from sciona.dfdc_hull_backend import load_hull_backend
from sciona.dfdc_ensemble_training import train_ensemble
from sciona.dfdc_ensemble_prediction import predict_selected


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value)-set(required)-set(optional):
        raise ValueError('invalid DFDC runtime fields')


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('invalid DFDC runtime integer')
    return value


def _frames(value, *, training):
    frames = decode_array(value)
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
            or min(frames.shape[1:3]) < 2 or training and len(frames) == 0):
        raise ValueError('runtime clips must contain decoded uint8 RGB frames')
    return frames


def prepare(payload):
    _keys(payload, ['version','training','prediction','detector','classifier','hull','plan','config'])
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('unsupported DFDC runtime version')
    config = payload['config']
    _keys(config, ['record_order_seed'], ['n_splits','batch_size','batches_per_epoch','test_every',
                                         'frames_per_video','precision'])
    settings = {'record_order_seed':_integer(config['record_order_seed'],0,2**32-1),
                'n_splits':_integer(config.get('n_splits',16),1,50),
                'batch_size':_integer(config.get('batch_size',12),1,2**31-1),
                'batches_per_epoch':_integer(config.get('batches_per_epoch',2500),1,2**31-1),
                'test_every':_integer(config.get('test_every',1),1,2**31-1),
                'frames_per_video':_integer(config.get('frames_per_video',32),1,2**31-1),
                'precision':config.get('precision','float16')}
    if settings['precision'] not in ('float16','float32'):
        raise ValueError('unsupported DFDC inference precision')
    _keys(payload['plan'], ['runs','requests'])
    plan = selection_plan(payload['plan']['runs'],payload['plan']['requests'])
    if any(row[1] >= settings['n_splits'] for row in plan['runs']):
        raise ValueError('run fold is outside configured folds')
    training = payload['training']; prediction = payload['prediction']
    if not isinstance(training,list) or not training or not isinstance(prediction,list) or not prediction:
        raise ValueError('runtime requires training and prediction populations')
    clips = []
    for row in training:
        _keys(row,['frames','part','original'])
        clips.append({'frames':_frames(row['frames'],training=True),
                      'part':_integer(row['part'],0,49),
                      'original':_integer(row['original'],0,len(training)-1)})
    folds=assign_folds([r['part'] for r in clips],[r['original'] for r in clips],n_splits=settings['n_splits'])
    # Only paired originals and their alterations survive source record discovery.
    altered=[i for i,r in enumerate(clips) if r['original']!=i]
    paired=set(altered)|{clips[i]['original'] for i in altered}
    for _,fold,_ in plan['runs']:
        for validation in (False,True):
            labels={int(clips[i]['original']!=i) for i in paired if (folds[i]==fold)==validation}
            if labels!={0,1}:
                raise ValueError('each run needs paired real/fake training and validation clips')
    predictions=[_frames(value,training=False) for value in prediction]
    detector=payload['detector'];_keys(detector,['policy','seed'],['state'])
    seed=_integer(detector['seed'],0,2**32-1)
    if detector['policy']=='random':
        if 'state' in detector:raise ValueError('random detector cannot receive state')
        detector_state=None
    elif detector['policy']=='state':
        detector_state=decode_state(detector.get('state'))
    else:raise ValueError('unknown detector initialization policy')
    classifier=payload['classifier'];_keys(classifier,['policy'],['states'])
    states=None
    if classifier['policy']=='random':
        if 'states' in classifier:raise ValueError('random classifier cannot receive state')
    elif classifier['policy'] in ('encoder','state'):
        entries=classifier.get('states')
        if not isinstance(entries,list):raise ValueError('per-run classifier states are required')
        states={}
        for entry in entries:
            _keys(entry,['seed','fold','state'])
            key=(_integer(entry['seed'],0,2**32-1),_integer(entry['fold'],0,49))
            if key in states:raise ValueError('duplicate classifier initialization state')
            states[key]=decode_state(entry['state'])
        if set(states)!={tuple(row[:2]) for row in plan['runs']}:
            raise ValueError('classifier states must match all runs exactly')
    else:raise ValueError('unknown classifier initialization policy')
    _keys(payload['hull'],['predictor_path'])
    path=payload['hull']['predictor_path']
    if not isinstance(path,str) or not path or '\x00' in path:
        raise ValueError('explicit native hull predictor path is required')
    return {'training':clips,'prediction':predictions,'plan':plan,'config':settings,
            'detector':{'initialization':detector['policy'],'seed':seed,'state':detector_state},
            'classifier':{'initialization':classifier['policy'],'states':states},'predictor_path':path}


def execute(payload):
    return execute_prepared(prepare(payload))


def execute_prepared(prepared):
    """Execute only an internal result of prepare; public JSON entrypoint is execute."""
    py_state,np_state=random.getstate(),np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            hull_detector,hull_predictor=load_hull_backend(predictor_path=prepared['predictor_path'])
            boxes=build_preprocessing_detector(stage='boxes',**prepared['detector'])
            landmarks=build_preprocessing_detector(stage='landmarks',**prepared['detector'])
            cfg=prepared['config']
            records=build_records(prepared['training'],box_detector=boxes,landmark_detector=landmarks,
                                  record_order_seed=cfg['record_order_seed'],n_splits=cfg['n_splits'])
            del boxes,landmarks
            bundle=train_ensemble(records,prepared['plan'],detector=hull_detector,predictor=hull_predictor,
                                  **prepared['classifier'],batch_size=cfg['batch_size'],
                                  batches_per_epoch=cfg['batches_per_epoch'],test_every=cfg['test_every'])
            inference=build_detector(**prepared['detector'])
            results=predict_selected(prepared['prediction'],bundle,detector=inference,
                                     frames_per_video=cfg['frames_per_video'],precision=cfg['precision'])
            return {'version':1,'plan':bundle['plan'],'training_records':len(records),
                    'runs':bundle['runs'],'predictions':results}
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
