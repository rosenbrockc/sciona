"""Explicit planar metric/seconds boundary for forward geotemporal regression."""
from dataclasses import dataclass
import copy
import math
from sciona.geotemporal_features import rows
from sciona.geotemporal_validation import forward_splits
from sciona.geotemporal_training import fit,controls_valid

@dataclass(frozen=True)
class Prepared:
    payload: dict


def finite_json(value):
    if value is None or type(value) in (str,int,bool):return
    if type(value) is float and math.isfinite(value):return
    if type(value) is list:
        for item in value:finite_json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():finite_json(item)
        return
    raise ValueError('Expected finite JSON values')


def prepare(payload):
    finite_json(payload)
    if type(payload) is not dict or set(payload)!={'version','reference','observations','context','blocks','query','controls'} or type(payload['version']) is not int or payload['version']!=1:raise ValueError('Invalid payload fields/version')
    reference=payload['reference']
    if type(reference) is not dict or set(reference)!={'coordinates','distance_unit','time_unit','shared_origin'} or reference['coordinates']!='planar' or reference['distance_unit']!='m' or reference['time_unit']!='s' or reference['shared_origin'] is not True:
        raise ValueError('Shared planar metric coordinates and seconds clock required')
    controls_valid(payload['controls'])
    rows(payload['observations'],('target',));rows(payload['context'],('value',));rows(payload['query'],())
    if any(r['target']<0 for r in payload['observations']):raise ValueError('Nonnegative targets required')
    for population in ('observations','context','query'):
        keys=[(r['x'],r['y'],r['time']) for r in payload[population]]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate space/time observation')
    forward_splits([r['time'] for r in payload['observations']],payload['blocks'],gap=payload['controls']['gap'])
    if min(r['time'] for r in payload['query'])<=max(r['time'] for r in payload['observations'])+payload['controls']['gap']:raise ValueError('Queries must follow training with required gap')
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared geotemporal input required')
    p=prepare(prepared.payload).payload
    fitted=fit(p['observations'],p['context'],p['blocks'],p['controls'])
    result=dict(predictions=fitted.predict(p['query']),training_rows=len(p['observations']),query_rows=len(p['query']),
        validation_rows=sum(v is not None for v in fitted.validation_predictions),warmup_rows=sum(v is None for v in fitted.validation_predictions),
        folds=len(fitted.folds),trees=len(fitted.estimator.estimators_),feature_count=11,validation_mse=fitted.validation_mse)
    finite_json(result)
    return result
