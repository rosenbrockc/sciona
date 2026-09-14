"""Canonical private JSON boundary for the complete corrected uncertainty method."""
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import numpy as np
from sciona.m5u_hierarchy import aggregate
from sciona.m5u_cleaning import retained_rows


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration: str
    fingerprint: str


def fields(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('Execution fields differ from contract')


def integer(value):
    if type(value) is not int or not 0<=value<2**31:raise ValueError('Nonnegative bounded integer required')


def number(value):
    if value is None:return
    if type(value) not in (int,float) or not np.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative number or null required')


def matrix(value,element):
    if type(value) is not list or not value or any(type(row) is not list or not row for row in value):
        raise ValueError('Nonempty JSON matrix required')
    if len({len(row) for row in value})!=1:raise ValueError('Rectangular matrix required')
    for row in value:
        for item in row:element(item)


def unpack(payload):
    return dict(units=np.asarray(payload['units'],dtype=float),prices=np.asarray(payload['prices'],dtype=float),
                roles=np.asarray(payload['roles'],dtype=np.int64),
                category_partitions=dict(payload['partitions']),calendar=payload['calendar'],**payload['controls'])


def prepare(payload):
    fields(payload,('version','units','prices','roles','partitions','calendar','controls'))
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported version')
    matrix(payload['units'],number);matrix(payload['prices'],number);matrix(payload['roles'],integer)
    matrix(payload['partitions'],integer)
    partitions=payload['partitions']
    if (len(partitions)!=3 or any(len(row)!=2 for row in partitions)
            or len({row[0] for row in partitions})!=3 or {row[1] for row in partitions}!={13,14,15}):
        raise ValueError('Three unique category-to-source-partition pairs required')
    fields(payload['controls'],('minimum_day','seed'))
    for value in payload['controls'].values():integer(value)
    if not 1<=payload['controls']['seed']<2**31-2000:raise ValueError('Seed outside execution range')
    calendar=payload['calendar'];fields(calendar,('dates','holidays','state_codes','events'))
    if type(calendar['dates']) is not list or not calendar['dates'] or any(type(v) is not str for v in calendar['dates']):
        raise ValueError('ISO date list required')
    try:dates=[date.fromisoformat(v) for v in calendar['dates']]
    except ValueError:raise ValueError('Valid ISO dates required') from None
    if ([d.isoformat() for d in dates]!=calendar['dates']
            or any((b-a).days!=1 for a,b in zip(dates,dates[1:]))):
        raise ValueError('Consecutive canonical ISO dates required')
    matrix(calendar['holidays'],number);matrix(calendar['events'],integer)
    states=calendar['state_codes']
    if type(states) is not list or not states:raise ValueError('State code list required')
    for state in states:integer(state)
    if len(set(states))!=len(states):raise ValueError('Unique state codes required')
    if (len(calendar['holidays'])!=len(dates) or len(calendar['events'])!=len(dates)
            or len(calendar['events'][0])!=len(states)
            or any(value not in (0,1) for row in calendar['events'] for value in row)
            or any(value is None for row in calendar['holidays'] for value in row)):
        raise ValueError('Aligned finite holiday and binary event matrices required')
    try:encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON required') from None
    if len(encoded.encode())>256*1024*1024:raise ValueError('Private execution input exceeds byte limit')
    config=unpack(payload);units=config['units'];roles=config['roles']
    if config['prices'].shape!=units.shape or len(dates)<len(units)+28:
        raise ValueError('Aligned prices and complete future calendar required')
    aggregate(units,roles)  # Also validates hierarchy implications and base uniqueness.
    if set(roles[:,2])!=set(config['category_partitions']) or not set(roles[:,0]).issubset(states):
        raise ValueError('Complete category and state coverage required')
    selected=retained_rows(np.arange(len(units)),[d.year for d in dates],[d.month for d in dates],
                           payload['controls']['minimum_day'])
    if not len(selected) or selected[-1]!=len(units)-1:raise ValueError('Final day excluded by source cleaning')
    if len({dates[int(day)].year for day in selected})<3:raise ValueError('Three cleaned calendar groups required')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    try:payload=json.loads(prepared.configuration)
    except (ValueError,TypeError):raise ValueError('Canonical prepared input required') from None
    if prepare(payload)!=prepared:raise ValueError('Canonical prepared input required')
    from sciona.m5u_pipeline import run
    result=run(**unpack(payload))
    output=dict(forecast=result['predictions'].tolist(),levels=result['levels'].tolist(),
                hierarchy_indices=result['hierarchy_indices'].tolist(),quantiles=list(result['quantiles']),
                horizon=28,reports=result['reports'],normalization=result['normalization'],scope=result['scope'])
    json.dumps(output,allow_nan=False)
    return output
