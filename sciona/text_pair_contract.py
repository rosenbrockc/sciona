"""Strict JSON boundary for the generic text-pair execution lifecycle."""
from dataclasses import dataclass
import copy
import math
from sciona.text_pair_features import normalize_pairs
from sciona.text_pair_training import _keys,_labels,fit_ensemble,calibrate_threshold,predict_pairs


@dataclass(frozen=True)
class Prepared:
    payload: dict


def _json(value):
    if type(value) in (str,int,bool) or value is None:return
    if type(value) is float and math.isfinite(value):return
    if type(value) is list:
        for item in value:_json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():_json(item)
        return
    raise ValueError('Expected finite JSON values')


def prepare(payload):
    _json(payload)
    fields={'version','training_pairs','training_labels','calibration_pairs','calibration_labels','query_pairs','controls'}
    if type(payload) is not dict or set(payload)!=fields or type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Invalid text-pair payload fields or version')
    controls=payload['controls']
    if type(controls) is not dict or set(controls)!={'components','seed','trees','max_depth','min_leaf'}:
        raise ValueError('Invalid training control fields')
    for name,value in controls.items():
        if type(value) is not int or value < (0 if name=='seed' else 1):raise ValueError('Invalid integer training control')
    if controls['seed']>=2**32:raise ValueError('Invalid random seed')
    keys=[]
    for name in ('training','calibration','query'):
        pairs=payload[name+'_pairs'];normalize_pairs(pairs);selected=_keys(pairs)
        if name!='query':
            if len(selected)!=len(pairs):raise ValueError('Duplicate fitted pair')
            _labels(payload[name+'_labels'],len(pairs))
        keys.append(selected)
    if any(keys[a]&keys[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Overlapping normalized pair populations')
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared input required')
    p=prepare(prepared.payload).payload
    fitted=fit_ensemble(p['training_pairs'],p['training_labels'],**p['controls'])
    calibrated=calibrate_threshold(fitted,p['calibration_pairs'],p['calibration_labels'])
    result=predict_pairs(calibrated,p['query_pairs'])
    result.update(training_rows=len(p['training_pairs']),calibration_rows=len(p['calibration_pairs']),
        query_rows=len(p['query_pairs']),feature_count=12,models=2,forest_trees=len(fitted.forest.estimators_))
    _json(result)
    return result
