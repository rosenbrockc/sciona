"""Versioned JSON boundary for the full Contrails training/inference lifecycle."""
import random

import numpy as np
import torch

from sciona.contrails_lifecycle import run_lifecycle
from sciona.contrails_population import FOLDS


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError('Invalid runtime object fields')


def _array(value, shape, *, target=False):
    try:
        result = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid numerical array') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid numerical array shape or values')
    if target and np.any((result < 0) | (result > 1)):
        raise ValueError('Targets must lie in [0,1]')
    return result


def _state(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('Expected tensor state mapping')
    result = {}
    for name, record in value.items():
        if not isinstance(name, str):
            raise ValueError('Invalid state key')
        _keys(record, ['dtype', 'values'])
        if record['dtype'] not in ('float32', 'int64'):
            raise ValueError('Unsupported state dtype')
        data = np.asarray(record['values'])
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError('Invalid state tensor')
        if record['dtype'] == 'int64' and (not np.issubdtype(data.dtype, np.integer) or data.dtype.kind == 'u' and np.any(data > np.iinfo(np.int64).max)):
            raise ValueError('Expected signed integer buffer values')
        converted = data.astype(record['dtype'])
        if not np.isfinite(converted).all():
            raise ValueError('State conversion overflow')
        result[name] = torch.from_numpy(converted.copy())
    return result


def prepare(payload):
    _keys(payload, ['version', 'training', 'scoring', 'prediction', 'initialization', 'config'])
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('Unsupported runtime version')
    config = payload['config']
    _keys(config, ['seed'], ['variant', 'batch_size', 'num_workers', 'epoch_limit'])
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    variant = config.get('variant', 'v47')
    if variant not in FOLDS:
        raise ValueError('Unknown variant')
    for key, low, high in [('batch_size', 1, None), ('num_workers', 0, None), ('epoch_limit', 1, 35)]:
        if key in config and (type(config[key]) is not int or config[key] < low or high is not None and config[key] > high):
            raise ValueError('Invalid execution setting')
    seen = set()
    populations = {}
    for population in ['training', 'scoring', 'prediction']:
        records = payload[population]
        if not isinstance(records, list) or len(records) < (10 if population == 'training' else 1):
            raise ValueError('Insufficient population')
        decoded = []
        for record in records:
            fields = ['key', 'thermal'] + ([] if population == 'prediction' else ['label', 'annotation_mean'])
            _keys(record, fields)
            key = record['key']
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError('Population keys must be nonempty and globally distinct')
            seen.add(key)
            thermal = _array(record['thermal'], (4, 3, 256, 256))
            if population == 'prediction':
                decoded.append(thermal)
            else:
                decoded.append(dict(thermal=thermal, label=_array(record['label'], (1, 256, 256), target=True),
                                    annotation_mean=_array(record['annotation_mean'], (1, 256, 256), target=True)))
        populations[population] = decoded
    initial = payload['initialization']
    _keys(initial, ['policy'], ['states'])
    policy = initial['policy']
    states = None
    if policy == 'random':
        if 'states' in initial:
            raise ValueError('Random initialization cannot supply states')
    elif policy in ('encoder', 'full_model'):
        _keys(initial.get('states'), ['single', 'temporal'])
        states = {}
        for branch, value in initial['states'].items():
            if policy == 'encoder':
                states[branch] = _state(value)
            else:
                _keys(value, [str(f) for f in FOLDS[variant][branch]])
                states[branch] = {int(f): _state(s) for f, s in value.items()}
    else:
        raise ValueError('Unknown initialization policy')
    return dict(**populations, initialization=policy, initial_states=states, config=dict(config))


def execute(payload):
    return execute_prepared(prepare(payload))


def execute_prepared(prepared):
    ready = dict(prepared)
    config = dict(ready.pop('config'))
    seed = config.pop('seed')
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            result = run_lifecycle(**ready, **config)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return dict(version=1, probabilities=result['probabilities'].tolist(), masks=result['masks'],
                variant=result['variant'], initialization=result['initialization'],
                seed=seed, epoch_limit=result['epoch_limit'],
                folds=[dict(branch=f['branch'], fold=f['fold'],
                            optimizer_updates=sum(e['optimizer_steps'] for e in f['training']['epochs']))
                       for f in result['folds']])
