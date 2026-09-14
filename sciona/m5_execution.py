"""Private canonical JSON boundary for the independent M5 full lifecycle."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from sciona.m5_preprocessing import prepare as preprocess


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration: str
    fingerprint: str


def fields(value,names):
    if type(value) is not dict or set(value)!=set(names):
        raise ValueError('Execution fields differ from contract')


def integer(value):
    if type(value) is not int or not -(2**31)<=value<2**31:
        raise ValueError('Bounded integer role or day required')


def numeric(value, *, nullable=False):
    if nullable and value is None:return
    if type(value) not in (int,float) or not np.isfinite(value) or value<0:
        raise ValueError('Nonnegative finite numeric value required')


def array(value,element):
    if type(value) is not list or not value:raise ValueError('Nonempty array required')
    for item in value:element(item)


def unpack(payload):
    config=dict(payload['configuration'])
    config['history']=np.asarray(config['history'],dtype=np.float64)
    config['prices']=dict(config['prices'])
    config['prices']['value']=np.asarray(config['prices']['value'],dtype=np.float64)
    return config


def prepare(payload):
    fields(payload,('version','identities','configuration','controls'))
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported version')
    fields(payload['configuration'],('history','first_day','pairs','roles','calendar','prices','groupings'))
    fields(payload['controls'],('recursive_first_day','nonrecursive_first_day'))
    try:encoded=json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON required') from None
    if len(encoded.encode())>256*1024*1024:raise ValueError('Execution input exceeds byte limit')
    p=json.loads(encoded);c=p['configuration']
    array(c['history'],lambda row:array(row,numeric))
    array(c['pairs'],integer);array(c['roles'],lambda row:array(row,integer))
    array(c['groupings'],lambda row:array(row,integer));integer(c['first_day'])
    fields(c['calendar'],('day','week','date','categories'))
    fields(c['prices'],('pair','outlet','product','week','value'))
    for name in ('day','week'):array(c['calendar'][name],integer)
    if type(c['calendar']['date']) is not list or any(type(v) is not str for v in c['calendar']['date']):
        raise ValueError('ISO date strings required')
    categories=c['calendar']['categories']
    if type(categories) is not list or any(type(row) is not list or any(v is not None and type(v) not in (str,int,float) for v in row) for row in categories):
        raise ValueError('Calendar categories must be JSON scalars or null')
    for name in ('pair','outlet','product','week'):array(c['prices'][name],integer)
    array(c['prices']['value'],lambda v:numeric(v,nullable=True))
    for value in p['controls'].values():integer(value)
    identities=p['identities']
    if (type(identities) is not list or len(identities)!=len(c['history'])
            or any(type(v) is not str or not v for v in identities) or len(set(identities))!=len(identities)):
        raise ValueError('Aligned unique opaque identities required')
    if len({len(row) for row in c['history']})!=1:raise ValueError('Rectangular history required')
    config=unpack(p)
    blocks=preprocess(**config)
    cutoff=c['first_day']+len(c['history'][0])-1
    future=blocks['grid']['day']>cutoff
    counts=np.bincount(blocks['grid']['series'][future],minlength=len(identities))
    if not np.all(counts==28):raise ValueError('Every input series needs 28 forecast rows after release filtering')
    for start in p['controls'].values():
        if start>cutoff:raise ValueError('Training starts after history ends')
    mapping={int(pair):(int(role[1]),int(role[4])) for pair,role in zip(c['pairs'],c['roles'])}
    for pair,outlet,product in zip(c['prices']['pair'],c['prices']['outlet'],c['prices']['product']):
        if pair in mapping and mapping[pair]!=(outlet,product):raise ValueError('Price and hierarchy pair roles disagree')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if (type(prepared) is not Prepared or type(prepared.configuration) is not str
            or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint):
        raise ValueError('Unchanged prepared input required')
    p=json.loads(prepared.configuration)
    if prepare(p)!=prepared:raise ValueError('Canonical prepared input required')
    from sciona.m5_pooled_training import train
    from sciona.m5_forecast import predict
    config=unpack(p);blocks=preprocess(**config)
    cutoff=config['first_day']+len(config['history'][0])-1
    models=train(blocks,cutoff,**p['controls'])
    result=predict(blocks,models,cutoff)
    output=dict(forecast=result['forecast'].tolist(),models=result['models'],horizon=28,
                model_families=6,scope=result['scope'])
    json.dumps(output,allow_nan=False)
    return output
