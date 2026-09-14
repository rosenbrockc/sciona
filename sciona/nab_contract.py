"""Serialized private boundary for the generic causal NAB reconstruction."""
import copy
from dataclasses import dataclass
import json
from sciona.nab_streaming import detect
from sciona.nab_scoring import evaluate

_FIELDS={'version','values','detector','windows','probation_percent','costs'}


@dataclass(frozen=True)
class Prepared:
    payload: dict


def prepare(payload):
    if not isinstance(payload,dict) or set(payload)!=_FIELDS:raise ValueError('Exact NAB payload fields required')
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON payload required') from None
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported payload version')
    values=payload['values']
    if not isinstance(values,list) or not values or any(type(v) not in (int,float) for v in values):raise ValueError('Explicit numeric observation list required')
    detect([0.],payload['detector'])
    evaluate([0.]*len(values),payload['windows'],threshold=.5,probation_percent=payload['probation_percent'],costs=payload['costs'])
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared NAB payload required')
    p=prepare(prepared.payload).payload
    detection=detect(p['values'],p['detector'])
    scoring=evaluate([float(flag) for flag in detection['detections']],p['windows'],threshold=.5,
        probation_percent=p['probation_percent'],costs=p['costs'])
    return dict(detection=detection,evaluation=scoring)
