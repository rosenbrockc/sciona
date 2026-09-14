"""Strict JSON boundary for the generic chronological March pipeline."""
import copy
from dataclasses import dataclass
import json
from sciona.march_training import fit_predict
_FIELDS={'version','games','training','calibration','prediction','feature_controls','regularization','max_iterations','tolerance','seed'}
_CONTROLS={'free_throw_weight','elo_initial','elo_k','elo_scale','pagerank_alpha','pagerank_tolerance','pagerank_iterations'}


@dataclass(frozen=True)
class Prepared:
    payload: dict


def prepare(payload):
    if not isinstance(payload,dict) or set(payload)!=_FIELDS:raise ValueError('Exact March payload fields required')
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON payload required') from None
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported payload version')
    if not isinstance(payload['feature_controls'],dict) or set(payload['feature_controls'])!=_CONTROLS:raise ValueError('Exact feature controls required')
    for name in ('games','training','calibration','prediction'):
        if not isinstance(payload[name],list) or not payload[name]:raise ValueError('Explicit nonempty populations required')
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared March payload required')
    p=prepare(prepared.payload).payload;p.pop('version')
    result=fit_predict(**p);result['probabilities']=result['probabilities'].tolist()
    return result
