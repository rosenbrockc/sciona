"""Canonical private execution boundary for the independent Porto ensemble."""
from dataclasses import dataclass,field
import hashlib
import json
import numpy as np
from sciona.porto_transforms import _matrix


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration: str=field(repr=False)
    fingerprint: str=field(repr=False)


def fields(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('Invalid execution fields')


def prepare(payload):
    fields(payload,('version','training','query','controls'))
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported execution version')
    fields(payload['training'],('values','labels','identities'));fields(payload['query'],('values','identities'))
    controls=payload['controls'];fields(controls,('preparation','neural_models','tree'))
    fields(controls['preparation'],('dropped_columns','categorical_columns','binary_columns'))
    models=controls['neural_models']
    if type(models) is not list or len(models)!=5:raise ValueError('Five explicit neural models required')
    for model in models:
        fields(model,('seed','dae_controls','neural_controls'))
        fields(model['dae_controls'],('hidden','feature_layers','epochs','batch_size','learning_rate','decay','swap_probability','momentum'))
        fields(model['neural_controls'],('hidden','epochs','batch_size','learning_rate','decay','l2','momentum','dropout','input_dropout','dropout_scaling'))
    fields(controls['tree'],('seed','controls'))
    fields(controls['tree']['controls'],('rounds','num_leaves','learning_rate','min_data_in_leaf','feature_fraction','bagging_fraction','bagging_freq','lambda_l2'))
    for model in models+[controls['tree']]:
        if type(model['seed']) is not int or not 0<=model['seed']<2**31:raise ValueError('Bounded integer seeds required')
    try:encoded=json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON execution input required') from None
    if len(encoded.encode())>64*1024*1024:raise ValueError('Execution input exceeds byte limit')
    p=json.loads(encoded);t=p['training'];q=p['query']
    x=_matrix(t['values']);v=_matrix(q['values'])
    if x.shape[1]!=v.shape[1]:raise ValueError('Aligned population widths required')
    y=t['labels']
    if type(y) is not list or len(y)!=len(x) or any(type(i) is not int or i not in (0,1) for i in y) or set(y)!={0,1}:raise ValueError('Aligned binary labels required')
    seen=set()
    for ids,size in ((t['identities'],len(x)),(q['identities'],len(v))):
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Unique disjoint opaque identities required')
        seen.update(ids)
    roles=p['controls']['preparation'];seen=set()
    for columns in roles.values():
        if type(columns) is not list or any(type(i) is not int or not 0<=i<x.shape[1] for i in columns) or len(set(columns))!=len(columns) or seen.intersection(columns):raise ValueError('Disjoint valid column roles required')
        seen.update(columns)
    if len(roles['dropped_columns'])==x.shape[1]:raise ValueError('No model features retained')
    for values in (x,v):
        if not np.isin(values[:,roles['binary_columns']],[0.,1.]).all():raise ValueError('Declared binary values must be zero or one')
    if len({json.dumps(m,sort_keys=True) for m in p['controls']['neural_models']})!=5:raise ValueError('Duplicate neural model configurations')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint:raise ValueError('Unchanged prepared input required')
    try:p=json.loads(prepared.configuration)
    except (ValueError,TypeError):raise ValueError('Invalid prepared JSON') from None
    if prepare(p)!=prepared:raise ValueError('Canonical prepared input required')
    from sciona.porto_ensemble import fit
    result=fit(p['training']['values'],p['training']['labels'],p['query']['values'],**p['controls'])
    output=dict(probabilities=result['probabilities'].tolist(),models=result['models'],dae_models=result['dae_models'],
        prepared_width=result['prepared_width'],learned_widths=[r['learned_width'] for r in result['neural_records']],
        score_kind='equal_probability_mean',scope=result['scope'])
    json.dumps(output,allow_nan=False)
    return output
