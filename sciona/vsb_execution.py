"""Canonical private raw-signal boundary for the independent VSB pipeline."""
from dataclasses import dataclass,field
import hashlib
import json
import numpy as np
from sciona.vsb_threshold import repeated_folds,binary_labels


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration:str=field(repr=False)
    fingerprint:str=field(repr=False)


def fields(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('Invalid execution fields')


def prepare(payload):
    fields(payload,('version','training','query'))
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported version')
    fields(payload['training'],('signals','labels','identities'));fields(payload['query'],('signals','identities'))
    try:encoded=json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON input required') from None
    if len(encoded.encode())>64*1024*1024:raise ValueError('Execution input exceeds byte limit')
    p=json.loads(encoded);seen=set();lengths=[];sizes=[]
    for population in (p['training'],p['query']):
        x=np.asarray(population['signals'])
        if x.ndim!=2 or len(x)<3 or not x.shape[1] or x.shape[1]%3 or x.dtype.kind not in 'iuf' or not np.isfinite(x).all():raise ValueError('Finite samples by complete signal triples required')
        size=x.shape[1]//3;sizes.append(size);lengths.append(len(x));ids=population['identities']
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Unique disjoint opaque measurement identities required')
        seen.update(ids)
    if lengths[0]!=lengths[1]:raise ValueError('Aligned signal lengths required')
    raw_labels=p['training']['labels']
    if type(raw_labels) is not list or any(type(row) is not list or any(type(v) is not int or v not in (0,1) for v in row) for row in raw_labels):raise ValueError('Integer binary signal labels required')
    labels=np.asarray(p['training']['labels'])
    if labels.shape!=(sizes[0],3):raise ValueError('Three signal labels per measurement required')
    binary_labels(labels.reshape(-1))
    repeated_folds(labels.any(axis=1).astype(int),seed=123948,repetitions=25)
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint:raise ValueError('Unchanged prepared input required')
    p=json.loads(prepared.configuration)
    if prepare(p)!=prepared:raise ValueError('Canonical prepared input required')
    from sciona.vsb_measurement import measurement_features
    from sciona.vsb_training import fit
    training=measurement_features(p['training']['signals']);query=measurement_features(p['query']['signals'])
    result=fit(training,p['training']['labels'],query)
    out=dict(probabilities=result['probabilities'].tolist(),signal_decisions=result['signal_decisions'].tolist(),threshold=result['threshold'],models=result['models'],scope=result['scope'])
    json.dumps(out,allow_nan=False)
    return out
