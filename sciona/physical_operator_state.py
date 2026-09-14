"""Explicit SI grid/parameter contract for periodic scalar diffusion."""
import copy
import math
import numpy as np


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


def array(value,ndim):
    def numeric(v):
        if type(v) is list:return all(numeric(x) for x in v)
        return type(v) in (int,float)
    if type(value) is not list or not numeric(value):raise ValueError('Expected numeric lists')
    try:result=np.asarray(value,dtype=np.float64)
    except (ValueError,OverflowError) as error:raise ValueError('Invalid numeric array') from error
    if result.ndim!=ndim or not all(result.shape) or not np.isfinite(result).all():raise ValueError('Expected finite nonempty numeric array')
    return result


def prepare(payload):
    finite_json(payload)
    if type(payload) is not dict or set(payload)!={'version','units','length','grid','training','validation','query','controls'} or type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Invalid physical operator payload fields')
    if payload['units']!={'length':'m','time':'s','diffusivity':'m^2/s','state':'dimensionless'}:raise ValueError('Explicit SI units required')
    length=payload['length']
    if type(length) not in (int,float) or length<=0:raise ValueError('Positive domain length required')
    grid=array(payload['grid'],1);n=len(grid)
    if n<4 or not np.allclose(grid/length,np.arange(n)/n,rtol=0,atol=1e-12):
        raise ValueError('Uniform endpoint-excluded periodic grid required')
    groups=[]
    for name in ('training','validation','query'):
        part=payload[name];fields={'states','diffusivity','elapsed','groups'}
        if name!='query':fields.add('targets')
        if type(part) is not dict or set(part)!=fields:raise ValueError('Invalid population fields')
        state=array(part['states'],2);diff=array(part['diffusivity'],1);elapsed=array(part['elapsed'],1)
        if state.shape[1]!=n:raise ValueError('Population shape mismatch')
        if diff.shape!=(len(state),) or elapsed.shape!=(len(state),):raise ValueError('Parameter shape mismatch')
        if (state<0).any() or (diff<0).any() or (elapsed<0).any():raise ValueError('Nonnegative states and diffusion parameters required')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            tau=(diff/length)*(elapsed/length)
        if not np.isfinite(tau).all():raise ValueError('Unrepresentable nondimensional diffusion time')
        if name!='query':
            target=array(part['targets'],2)
            if target.shape!=state.shape or (target<0).any():raise ValueError('Invalid diffusion targets')
            if not np.allclose(target.mean(1),state.mean(1),rtol=1e-9,atol=1e-12):raise ValueError('Targets must conserve mean state')
        ids=part['groups']
        if type(ids) is not list or len(ids)!=len(state) or any(type(g) is not str or not g for g in ids):raise ValueError('Aligned group identifiers required')
        groups.append(set(ids))
    if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Groups overlap populations')
    c=payload['controls']
    if type(c) is not dict or set(c)!={'seed','width','modes','depth','epochs','learning_rate','smoothing'}:raise ValueError('Invalid controls')
    for name in ('seed','width','modes','depth','epochs'):
        if type(c[name]) is not int or c[name]<(0 if name=='seed' else 1):raise ValueError('Invalid integer controls')
    if c['seed']>=2**32 or c['modes']>n//2+1:raise ValueError('Invalid seed or retained modes')
    if type(c['learning_rate']) not in (int,float) or c['learning_rate']<=0:raise ValueError('Positive learning rate required')
    if type(c['smoothing']) not in (int,float) or not 0<=c['smoothing']<=.5:raise ValueError('Invalid smoothing')
    return copy.deepcopy(payload)


def tensors(part,length):
    import torch
    states=torch.tensor(part['states'],dtype=torch.float64)
    tau=(np.asarray(part['diffusivity'])/length)*(np.asarray(part['elapsed'])/length)
    return states,torch.tensor(tau,dtype=torch.float64)
