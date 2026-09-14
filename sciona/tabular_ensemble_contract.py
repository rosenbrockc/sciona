"""Finite JSON boundary for group-isolated binary tabular stacking."""
from dataclasses import dataclass
import copy
import math
from sciona.tabular_ensemble_features import validate_table
from sciona.tabular_ensemble_training import binary_labels, group_ids, fit_stacker
from sciona.tabular_ensemble_calibration import calibrate_stack, predict_calibrated

@dataclass(frozen=True)
class Prepared:
    payload: dict

def _json(value):
    if value is None or type(value) in (str, int, bool): return
    if type(value) is float and math.isfinite(value): return
    if type(value) is list:
        for item in value: _json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values(): _json(item)
        return
    raise ValueError('Expected finite JSON values')

def prepare(payload):
    _json(payload)
    if type(payload) is not dict or set(payload) != {'version','training','calibration','query','controls'} or type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('Invalid tabular payload fields or version')
    controls = payload['controls']
    if type(controls) is not dict or set(controls) != {'seed','trees','max_depth','min_leaf','clip_low','clip_high'}:
        raise ValueError('Invalid control fields')
    for name in ('seed','trees','max_depth','min_leaf'):
        if type(controls[name]) is not int or controls[name] < (0 if name == 'seed' else 1):
            raise ValueError('Invalid integer control')
    if controls['seed'] >= 2**32: raise ValueError('Invalid random seed')
    if any(type(controls[n]) not in (int,float) for n in ('clip_low','clip_high')) or not 0 <= controls['clip_low'] < controls['clip_high'] <= 1:
        raise ValueError('Invalid clipping quantiles')
    populations = []; widths = []
    for name in ('training','calibration','query'):
        part = payload[name]
        fields = {'numeric','categorical','groups'}
        if name != 'query': fields.add('labels')
        if name == 'training': fields.add('folds')
        if type(part) is not dict or set(part) != fields: raise ValueError('Invalid population fields')
        numeric, categorical = validate_table(part['numeric'],part['categorical'])
        widths.append((numeric.shape[1],categorical.shape[1]))
        populations.append(group_ids(part['groups'],len(numeric)))
        if name != 'query': binary_labels(part['labels'],len(numeric))
    if len(set(widths)) != 1: raise ValueError('Population widths differ')
    if any(populations[a] & populations[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Overlapping population groups')
    training = payload['training']; folds = training['folds']
    if type(folds) is not list or len(folds) != len(training['labels']) or any(type(f) is not int or f < 0 for f in folds):
        raise ValueError('Invalid fold assignment')
    selected = set(folds)
    if len(selected) < 2 or selected != set(range(len(selected))): raise ValueError('Invalid fold indices')
    assignments = {}
    for group, fold in zip(training['groups'],folds):
        if group in assignments and assignments[group] != fold: raise ValueError('Group crosses folds')
        assignments[group] = fold
    for fold in selected:
        if {label for label, assigned in zip(training['labels'],folds) if assigned != fold} != {0,1}:
            raise ValueError('Each fitting fold requires both classes')
    return Prepared(copy.deepcopy(payload))

def execute(prepared):
    if not isinstance(prepared,Prepared): raise ValueError('Prepared input required')
    p = prepare(prepared.payload).payload
    fitted = fit_stacker(**p['training'],**p['controls'])
    calibrated = calibrate_stack(fitted,**p['calibration'])
    result = predict_calibrated(calibrated,**p['query'])
    result.update(training_rows=len(p['training']['numeric']),calibration_rows=len(p['calibration']['numeric']),
        query_rows=len(p['query']['numeric']),folds=len(fitted.fold_models),models=2,
        forest_trees=len(fitted.base.trees.estimators_),oof_rows=len(fitted.oof_predictions))
    _json(result)
    return result
