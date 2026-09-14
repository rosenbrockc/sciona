"""Strict private-input graph boundary for the complete Cornell lifecycle."""
import copy
from dataclasses import dataclass
import os
import numpy as np
import torch
from sciona.cornell_population import prepare_records,check_split


@dataclass(frozen=True)
class Prepared:
    payload: dict


def waveform(value,minimum=1,noise=False):
    array=np.asarray(value)
    if array.ndim!=1 or len(array)<minimum or array.dtype.kind not in 'fiu':raise ValueError('Invalid mono waveform')
    with np.errstate(over='ignore',invalid='ignore'):
        array=np.asarray(array,dtype=np.float32)
        rms=np.sqrt(np.mean(array*array)) if noise else 1.
    if not np.isfinite(array).all() or not np.isfinite(rms) or rms<=0:raise ValueError('Nonfinite or silent noise waveform')


def prepare(payload):
    fields={'version','populations','background','short_noises','inference','epochs','batch_size','seed','initialization'}
    if not isinstance(payload,dict) or set(payload)!=fields:raise ValueError('Unexpected Cornell payload fields')
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported Cornell version')
    if type(payload['epochs']) is not int or payload['epochs']<2:raise ValueError('At least two epochs required')
    size=payload['batch_size']
    if type(size) is not int or size<2 or size%2:raise ValueError('Expected even batch size')
    if type(payload['seed']) is not int or not 0<=payload['seed']<2**32-20000:raise ValueError('Invalid seed')
    if payload['initialization'] not in ('synthetic','provisioned_backbone'):raise ValueError('Explicit initialization required')
    if not isinstance(payload['populations'],list):raise ValueError('Expected fold list')
    expected={(group,i) for group,n in [('four',4),('five',5)] for i in range(n)};seen=set()
    for split in payload['populations']:
        if not isinstance(split,dict) or set(split)!={'group','fold','training','validation'}:raise ValueError('Invalid fold fields')
        if not isinstance(split['group'],str) or type(split['fold']) is not int:raise ValueError('Invalid fold identity')
        key=split['group'],split['fold']
        if key not in expected or key in seen:raise ValueError('Unexpected or duplicate fold')
        seen.add(key)
        train=prepare_records(split['training'],training=True);valid=prepare_records(split['validation'],training=False)
        check_split(train,valid)
        if len(train)<size:raise ValueError('Incomplete fold training batch')
    if seen!=expected:raise ValueError('Missing source folds')
    for name in ('background','short_noises'):
        if not isinstance(payload[name],(list,tuple)) or not payload[name]:raise ValueError('Required noise bank missing')
        for value in payload[name]:waveform(value,minimum=3200,noise=True)
    inference=payload['inference']
    if not isinstance(inference,dict) or set(inference)!={'sample_rate','waveform'} or type(inference['sample_rate']) is not int or inference['sample_rate']!=32000:
        raise ValueError('Inference requires32000Hz mono audio')
    waveform(inference['waveform'])
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared Cornell input required')
    payload=prepare(prepared.payload).payload
    source=os.environ.get('SCIONA_CORNELL_SOURCE_DIR');dependencies=os.environ.get('SCIONA_CORNELL_DEPENDENCY_DIR')
    if not source or not dependencies:raise ValueError('Provision Cornell source and historical dependencies')
    state=None
    if payload['initialization']=='provisioned_backbone':
        path=os.environ.get('SCIONA_CORNELL_BACKBONE_STATE')
        if not path:raise ValueError('Provision complete DenseNet121 backbone state')
        state=torch.load(path,map_location='cpu',weights_only=True)
    from sciona.cornell_lifecycle import execute as lifecycle
    populations={(p['group'],p['fold']):dict(training=p['training'],validation=p['validation']) for p in payload['populations']}
    result=lifecycle(source,dependencies,populations=populations,background=payload['background'],short_noises=payload['short_noises'],
        inference=payload['inference']['waveform'],epochs=payload['epochs'],batch_size=payload['batch_size'],seed=payload['seed'],
        backbone_state=state,synthetic=payload['initialization']=='synthetic')
    return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in result.items()}
